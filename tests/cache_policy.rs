use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const REPO_ID: &str = "openai-community/gpt2";
const DATA_FILE: &str = "model.safetensors";
const READ_LEN: usize = 4096;
const OFFSETS: [u64; 3] = [0, 160 * 1024 * 1024, 320 * 1024 * 1024];
const LRU_CACHE_SIZE_CANDIDATES: &[u64] = &[
    256 * 1024 * 1024,
    192 * 1024 * 1024,
    160 * 1024 * 1024,
    128 * 1024 * 1024,
    384 * 1024 * 1024,
];
const LFU_CACHE_SIZE_CANDIDATES: &[u64] = &[
    192 * 1024 * 1024,
    256 * 1024 * 1024,
    160 * 1024 * 1024,
    128 * 1024 * 1024,
    384 * 1024 * 1024,
];
const CACHE_POLL_INTERVAL: Duration = Duration::from_millis(200);
const CACHE_STABLE_ROUNDS: usize = 15;

#[test]
fn repo_chunk_cache_eviction_follows_lru_and_lfu_policy() {
    let binary = hf_mount_fuse_binary();
    if !binary.exists() {
        eprintln!("Skipping: {:?} not found; run with `--features fuse`", binary);
        return;
    }

    run_policy_case(&binary, "lru", ExpectedPolicy::Lru, LRU_CACHE_SIZE_CANDIDATES);
    run_policy_case(&binary, "lfu", ExpectedPolicy::Lfu, LFU_CACHE_SIZE_CANDIDATES);
}

#[derive(Clone, Copy)]
enum ExpectedPolicy {
    Lru,
    Lfu,
}

impl ExpectedPolicy {
    fn is_satisfied(self, a_survival: f64, b_survival: f64) -> bool {
        match self {
            Self::Lru => a_survival < b_survival,
            Self::Lfu => b_survival < a_survival,
        }
    }

    fn expectation(self) -> &'static str {
        match self {
            Self::Lru => "LRU should evict more of older A than newer B",
            Self::Lfu => "LFU should evict more of once-read B than twice-read A",
        }
    }
}

fn run_policy_case(binary: &Path, policy: &str, expected: ExpectedPolicy, cache_size_candidates: &[u64]) {
    let mut retry_reasons = Vec::new();
    for cache_size in cache_size_candidates {
        eprintln!("Running cache policy case: {policy}, cache_size={cache_size}");
        match try_policy_case(binary, policy, expected, *cache_size) {
            AttemptResult::Passed => {
                eprintln!("  {policy}: eviction assertions passed with cache_size={cache_size}");
                return;
            }
            AttemptResult::Retry(reason) => {
                eprintln!("  {policy}: retrying with another cache size: {reason}");
                retry_reasons.push(format!("cache_size={cache_size}: {reason}"));
            }
            AttemptResult::Failed(reason) => panic!("cache policy {policy} failed: {reason}"),
        }
    }

    panic!(
        "cache policy {policy} did not produce a stable eviction comparison. Attempts:\n{}",
        retry_reasons.join("\n")
    );
}

enum AttemptResult {
    Passed,
    Retry(String),
    Failed(String),
}

fn try_policy_case(binary: &Path, policy: &str, expected: ExpectedPolicy, cache_size: u64) -> AttemptResult {
    let suffix = unique_suffix(policy);
    let mount_point = PathBuf::from(format!("/tmp/hf-cache-policy-{suffix}-mnt"));
    let cache_dir = PathBuf::from(format!("/tmp/hf-cache-policy-{suffix}-cache"));
    let child = mount_repo(
        binary,
        &mount_point,
        &cache_dir,
        &[
            "--cache-size",
            &cache_size.to_string(),
            "--cache-policy",
            policy,
            "--direct-io",
            "--fuse-owner-only",
        ],
    );
    let guard = MountGuard::new(mount_point.clone(), cache_dir.clone(), child);
    let data_path = mount_point.join(DATA_FILE);
    assert_offsets_are_readable(&data_path);

    eprintln!("  {policy}: read A at offset {}", OFFSETS[0]);
    if let Err(reason) = read_at(&data_path, OFFSETS[0]) {
        drop(guard);
        return AttemptResult::Retry(format!("A read failed: {reason}"));
    }
    let after_a = match wait_for_cache_settle(&cache_dir, 1) {
        Ok(entries) => entries,
        Err(reason) => {
            drop(guard);
            return AttemptResult::Retry(format!("A read did not produce stable cache entries: {reason}"));
        }
    };
    let a_entries = entry_keys(&after_a);
    if a_entries.is_empty() {
        drop(guard);
        return AttemptResult::Retry("A read produced no cache entries".to_string());
    }

    eprintln!("  {policy}: read A again to raise frequency/recency");
    if let Err(reason) = read_at(&data_path, OFFSETS[0]) {
        drop(guard);
        return AttemptResult::Retry(format!("second A read failed: {reason}"));
    }
    let after_a_again = match wait_for_cache_contains(&cache_dir, &a_entries) {
        Ok(entries) => entries,
        Err(reason) => {
            drop(guard);
            return AttemptResult::Retry(format!("second A read did not keep A entries stable: {reason}"));
        }
    };

    eprintln!("  {policy}: read B at offset {}", OFFSETS[1]);
    if let Err(reason) = read_at(&data_path, OFFSETS[1]) {
        drop(guard);
        return AttemptResult::Retry(format!("B read failed: {reason}"));
    }
    let after_b = match wait_for_cache_change_and_settle(&cache_dir, &after_a_again, cache_size) {
        Ok(entries) => entries,
        Err(reason) => {
            drop(guard);
            return AttemptResult::Retry(format!("B read did not produce a stable cache change: {reason}"));
        }
    };
    let b_entries = diff_keys(&after_b, &after_a_again);
    if b_entries.is_empty() {
        drop(guard);
        return AttemptResult::Retry("B read did not add distinct cache entries".to_string());
    }
    if !contains_all(&after_b, &a_entries) {
        drop(guard);
        return AttemptResult::Retry("cache too small to keep A and B before C read".to_string());
    }

    eprintln!("  {policy}: read C at offset {} to force eviction", OFFSETS[2]);
    if let Err(reason) = read_at(&data_path, OFFSETS[2]) {
        drop(guard);
        return AttemptResult::Retry(format!("C read failed: {reason}"));
    }
    let final_entries = match wait_for_cache_change_and_settle(&cache_dir, &after_b, cache_size) {
        Ok(entries) => entries,
        Err(reason) => {
            drop(guard);
            return AttemptResult::Retry(format!("C read did not produce a stable cache change: {reason}"));
        }
    };
    let c_entries = diff_keys(&final_entries, &after_b);
    if c_entries.is_empty() {
        drop(guard);
        return AttemptResult::Retry("C read did not add distinct cache entries".to_string());
    }

    let a_survivors = count_present(&final_entries, &a_entries);
    let b_survivors = count_present(&final_entries, &b_entries);
    if a_survivors == a_entries.len() && b_survivors == b_entries.len() {
        drop(guard);
        return AttemptResult::Retry("C read did not evict any A/B entries".to_string());
    }

    let a_survival = a_survivors as f64 / a_entries.len() as f64;
    let b_survival = b_survivors as f64 / b_entries.len() as f64;
    eprintln!(
        "  {policy}: survival A={}/{} ({:.2}), B={}/{} ({:.2}), C_new={}",
        a_survivors,
        a_entries.len(),
        a_survival,
        b_survivors,
        b_entries.len(),
        b_survival,
        c_entries.len(),
    );

    drop(guard);

    if expected.is_satisfied(a_survival, b_survival) {
        AttemptResult::Passed
    } else if a_survivors == 0 && b_survivors == 0 {
        AttemptResult::Retry("C read evicted all A/B entries; cache too small for comparison".to_string())
    } else {
        AttemptResult::Failed(format!(
            "{}; observed survival A={:.2}, B={:.2}",
            expected.expectation(),
            a_survival,
            b_survival
        ))
    }
}

fn mount_repo(binary: &Path, mount_point: &Path, cache_dir: &Path, extra_args: &[&str]) -> Child {
    fs::create_dir_all(mount_point).expect("create mount point");
    fs::create_dir_all(cache_dir).expect("create cache dir");

    let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    let mut command = Command::new(binary);
    command.env(
        "RUST_LOG",
        std::env::var("RUST_LOG").unwrap_or_else(|_| "hf_mount=warn".to_string()),
    );
    command.envs([
        ("HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY", "1"),
        ("HF_XET_RECONSTRUCTION_MIN_RECONSTRUCTION_FETCH_SIZE", "4096"),
        ("HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER", "4096"),
        ("HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE", "1048576"),
        ("HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT", "4194304"),
    ]);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        command.args(["--hf-token", &token]);
    }

    let child = command
        .args([
            "--hub-endpoint",
            &endpoint,
            "--cache-dir",
            cache_dir.to_str().expect("cache dir should be valid UTF-8"),
            "--poll-interval-secs",
            "0",
        ])
        .args(extra_args)
        .args([
            "repo",
            REPO_ID,
            mount_point.to_str().expect("mount point should be valid UTF-8"),
        ])
        .spawn()
        .expect("spawn hf-mount-fuse");

    wait_for_mount(mount_point);
    child
}

fn assert_offsets_are_readable(path: &Path) {
    let file_len = fs::metadata(path).expect("stat data file").len();
    let required_len = OFFSETS.last().copied().unwrap() + READ_LEN as u64;
    assert!(
        file_len >= required_len,
        "{DATA_FILE} in {REPO_ID} is too small: {file_len} < {required_len}"
    );
}

fn read_at(path: &Path, offset: u64) -> Result<(), String> {
    assert_eq!(offset % READ_LEN as u64, 0, "offset must align to READ_LEN");
    let skip = (offset / READ_LEN as u64).to_string();
    let block_size = READ_LEN.to_string();
    let status = Command::new("timeout")
        .args([
            "45s",
            "dd",
            &format!("if={}", path.display()),
            "of=/dev/null",
            &format!("bs={block_size}"),
            "count=1",
            &format!("skip={skip}"),
            "status=none",
        ])
        .status()
        .map_err(|err| format!("spawn timeout/dd: {err}"))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("timeout/dd exited with status {status}"))
    }
}

fn wait_for_mount(mount_point: &Path) {
    let mount_point = mount_point.to_string_lossy();
    for _ in 0..60 {
        if let Ok(mounts) = fs::read_to_string("/proc/mounts") {
            if mounts.lines().any(|line| line.contains(mount_point.as_ref())) {
                return;
            }
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    panic!("mount point {mount_point} did not appear in /proc/mounts");
}

fn wait_for_cache_settle(cache_dir: &Path, min_entries: usize) -> Result<BTreeMap<PathBuf, u64>, String> {
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    let mut last = BTreeMap::new();
    let mut stable_rounds = 0;

    while std::time::Instant::now() < deadline {
        let current = cache_entries(cache_dir).expect("read cache entries");
        if current == last && current.len() >= min_entries {
            stable_rounds += 1;
            if stable_rounds >= CACHE_STABLE_ROUNDS {
                return Ok(current);
            }
        } else {
            stable_rounds = 0;
            last = current;
        }
        std::thread::sleep(CACHE_POLL_INTERVAL);
    }

    Err(format!(
        "cache did not settle with at least {min_entries} entries; last entries: {:?}",
        last.keys().collect::<Vec<_>>()
    ))
}

fn wait_for_cache_contains(cache_dir: &Path, expected: &BTreeSet<PathBuf>) -> Result<BTreeMap<PathBuf, u64>, String> {
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    while std::time::Instant::now() < deadline {
        let entries = cache_entries(cache_dir).expect("read cache entries");
        if contains_all(&entries, expected) {
            return wait_for_cache_settle(cache_dir, entries.len());
        }
        std::thread::sleep(CACHE_POLL_INTERVAL);
    }
    Err(format!("cache did not contain expected entries: {expected:?}"))
}

fn wait_for_cache_change_and_settle(
    cache_dir: &Path,
    before: &BTreeMap<PathBuf, u64>,
    cache_size: u64,
) -> Result<BTreeMap<PathBuf, u64>, String> {
    let deadline = std::time::Instant::now() + Duration::from_secs(90);
    let mut saw_change = false;
    let mut last = BTreeMap::new();
    let mut stable_rounds = 0;

    while std::time::Instant::now() < deadline {
        let current = cache_entries(cache_dir).expect("read cache entries");
        if current != *before {
            saw_change = true;
        }

        if saw_change && current == last && total_size(&current) <= cache_size {
            stable_rounds += 1;
            if stable_rounds >= CACHE_STABLE_ROUNDS {
                return Ok(current);
            }
        } else {
            stable_rounds = 0;
            last = current;
        }

        std::thread::sleep(CACHE_POLL_INTERVAL);
    }

    Err(format!(
        "cache did not change and settle; cache_size={}, before={:?}, last_size={}, last={:?}",
        cache_size,
        before.keys().collect::<Vec<_>>(),
        total_size(&last),
        last.keys().collect::<Vec<_>>()
    ))
}

fn cache_entries(cache_dir: &Path) -> std::io::Result<BTreeMap<PathBuf, u64>> {
    let root = cache_dir.join("xorbs");
    let mut entries = BTreeMap::new();
    if !root.exists() {
        return Ok(entries);
    }
    collect_cache_entries(&root, &root, &mut entries)?;
    Ok(entries)
}

fn collect_cache_entries(root: &Path, dir: &Path, entries: &mut BTreeMap<PathBuf, u64>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            collect_cache_entries(root, &path, entries)?;
        } else if metadata.is_file() && metadata.len() > 0 {
            let rel = path
                .strip_prefix(root)
                .expect("cache path should be under root")
                .to_path_buf();
            entries.insert(rel, metadata.len());
        }
    }
    Ok(())
}

fn entry_keys(entries: &BTreeMap<PathBuf, u64>) -> BTreeSet<PathBuf> {
    entries.keys().cloned().collect()
}

fn diff_keys(after: &BTreeMap<PathBuf, u64>, before: &BTreeMap<PathBuf, u64>) -> BTreeSet<PathBuf> {
    after
        .keys()
        .filter(|entry| !before.contains_key(*entry))
        .cloned()
        .collect()
}

fn contains_all(actual: &BTreeMap<PathBuf, u64>, expected: &BTreeSet<PathBuf>) -> bool {
    expected.iter().all(|entry| actual.contains_key(entry))
}

fn count_present(actual: &BTreeMap<PathBuf, u64>, expected: &BTreeSet<PathBuf>) -> usize {
    expected.iter().filter(|entry| actual.contains_key(*entry)).count()
}

fn total_size(entries: &BTreeMap<PathBuf, u64>) -> u64 {
    entries.values().sum()
}

fn hf_mount_fuse_binary() -> PathBuf {
    std::env::current_exe()
        .expect("current exe")
        .parent()
        .expect("test deps dir")
        .parent()
        .expect("target debug dir")
        .join("hf-mount-fuse")
}

fn unique_suffix(label: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{}-{}-{}", label, std::process::id(), nanos)
}

struct MountGuard {
    mount_point: PathBuf,
    cache_dir: PathBuf,
    child: Option<Child>,
}

impl MountGuard {
    fn new(mount_point: PathBuf, cache_dir: PathBuf, child: Child) -> Self {
        Self {
            mount_point,
            cache_dir,
            child: Some(child),
        }
    }
}

impl Drop for MountGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            unmount(&self.mount_point);
            for _ in 0..10 {
                if let Ok(Some(_)) = child.try_wait() {
                    break;
                }
                std::thread::sleep(Duration::from_secs(1));
            }
            if child.try_wait().ok().flatten().is_none() {
                child.kill().ok();
                child.wait().ok();
            }
        }
        fs::remove_dir_all(&self.mount_point).ok();
        fs::remove_dir_all(&self.cache_dir).ok();
    }
}

fn unmount(mount_point: &Path) {
    let commands: &[(&str, &[&str])] = &[("fusermount", &["-u"]), ("fusermount3", &["-u"]), ("umount", &[])];

    for (cmd, args) in commands {
        if let Ok(status) = Command::new(cmd).args(*args).arg(mount_point).status() {
            if status.success() {
                return;
            }
        }
    }
}
