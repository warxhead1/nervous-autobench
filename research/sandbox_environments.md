# Sandbox Environment Research for Autobench

This document covers execution environments suitable for multi-language code testing with sub-second latency requirements.

---

## 1. gVisor Analysis

### Architecture

gVisor (`runsc`) implements a Linux kernel interface in userspace. It intercepts syscalls via a "Sentry" process:

```
Untrusted App → Platform Syscall Switcher → Sentry (userspace kernel) → Gofer (I/O handler) → Host OS
```

Two platforms:
- **KVM**: Uses hardware virtualization (~830ns/getpid overhead)
- **ptrace**: Software interception (~6249ns/getpid) — slower but works without KVM

### Cold Start Latency

- gVisor containers start in **milliseconds**
- Filter precompilation (2024) eliminated ~10ms startup delay from cBPF compilation
- Native getpid: ~62ns, gVisor-KVM: ~830ns (13x overhead)
- For short-lived processes (<100ms), syscall interception overhead is measurable but acceptable

### Fastest Execution Path

1. Pre-compiled seccomp filters (now default in `runsc`)
2. Use `--platform=kvm` flag (requires KVM hardware)
3. Avoid rootfs tar annotation if possible — direct OCI image pull is faster

### Pre-warming Strategy

gVisor supports **checkpoint/restore**:
```bash
# Checkpoint a running container
runsc checkpoint --state-file=/tmp/saved-state <container_id>

# Restore instantly
runsc restore --detach=true --state-file=/tmp/saved-state <container_id>
```

Warm container pools are possible: keep N containers paused/restored, clone via `runsc clone`.

---

## 2. Firecracker

### MicroVM Overhead

- **Startup time**: <125ms (boot + userspace ready)
- **Memory footprint**: ~5MiB for the VMM itself
- **Hardware isolation**: Full KVM virtualization — stronger than gVisor

### Snapshot/Restore Speed

- Snapshot creation pauses the VM and serializes state + memory
- **Restore is nearly instant** — under 125ms total for resume
- Memory page faults on restore: host kernel handles on-demand (can be slow with cgroups v1, use v2)
- Network interfaces can be remapped on restore for tap device pooling

### Firecracker vs gVisor for Short-Lived Processes

| Metric | gVisor (KVM) | Firecracker |
|--------|-------------|-------------|
| Startup | ~50-100ms | ~100-125ms |
| Memory overhead | ~10-50MB | ~5MB VMM + guest memory |
| Isolation | Syscall filtering | Hardware (KVM) |
| Per-process overhead | ~830ns/syscall | Minimal (native) |
| Snapshot/restore | Not supported natively | Full snapshot support |

**Verdict**: For code that makes many syscalls, gVisor's per-call overhead dominates. For compute-heavy code with minimal syscalls, Firecracker wins. For autobench (short-lived, syscall-light), **Firecracker snapshot restore** is likely faster.

---

## 3. WebAssembly (Wasmtime/Wasmer)

### Isolation Performance for Compiled Languages

WASM runtimes execute compiled languages with hardware-assisted sandboxing (WASM spec's memory safety).

| Runtime | Cold start | Execution overhead |
|---------|------------|-------------------|
| Wasmtime (Cranelift) | ~1-10ms JIT | ~90-110% native (CPU-bound) |
| Wasmer | ~1-5ms (JIT disabled) | ~100-130% native |
| WAMR (embedded) | <1ms | ~80-100% native |

### Compile-to-WASM Overhead

- **Rust**: `cargo build --target wasm32-wasi` — compile time comparable to native, output ~2x binary size
- **C++**: `clang --target=wasm32 -c` then link with `lld` — compilation is standard
- **Go**: `tinygo build -target=wasm wasi` — slower compile but smaller binary
- **Overhead**: WASM compilation adds ~0.5-2s to build pipeline for typical packages

### Practical Assessment

WASM is **NOT faster than native execution** — it adds overhead for:
- Garbage collection (if used)
- WASM ↔ host syscall translation
- Linear memory model with bounds checking

WASM's advantage is **strong isolation + portability**, not speed. For autobench requiring sub-second compile+run, WASM doesn't help for Go/Rust/C++ (those languages compile to native faster than WASM overhead).

---

## 4. Namespace Containers (unshare/chroot)

### Cold Start Time

Raw Linux namespaces have the **fastest cold start**:
- `unshare --user --mount --pid --net --cgroup --ipc --uts` — ~1-5ms
- `chroot` to pivot_root — ~5-20ms depending on filesystem

### Namespace Pooling

Can be combined with user namespaces for efficient pooling:
```bash
# Create a fresh namespace with user mapping (no root)
unshare --user --map-root-user --mount --pid

# Or use systemd-run for transient scopes
systemd-run --scope --uid=1000 -- python3 script.py
```

### Verdict

**Fastest option for local isolation**. Good for:
- Pre-flight checks that don't need hardware virtualization
- Quick unit tests before shipping to heavier sandbox

Not suitable for untrusted code (no syscall filtering — just namespace isolation).

---

## 5. Existing Autobench Systems

### SWE-Bench

**Execution**: Docker containers
- Each instance runs in an OCI container with Ubuntu base
- Python code executed via `python3 -c` or pytest
- **Runtime constraints**: 8 CPU cores, 16GB RAM recommended, 120GB storage
- Timeout: `max_workers = min(0.75 * os.cpu_count(), 24)` — worker limits

**Sandbox**: Standard Docker isolation (cgroups + namespaces)
- ARM support is experimental
- Uses `docker build` for custom environments

### LiveCodeBench

**Execution**: Code is compiled/run with strict constraints:
- **Time limit**: Enforced via `ulimit -t` (CPU time) or `timeout` command
- **Memory limit**: `ulimit -m` or cgroup memory限制
- **Languages**: Multiple (Python, Java, C++, etc.)

**Verification**: Runs code against test cases, compares output.

### HumanEval/CodexEval

**Execution**: Python `exec()` with timeout wrapper:
```python
import signal
def timeout_handler(signum, frame):
    raise TimeoutError
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(TIME_LIMIT)
result = exec(code, globals)
```

**Constraints**: Soft timeouts via `signal.SIGALRM`, memory via `resource.setrlimit`.

**Verification**: stdout comparison against expected outputs.

---

## 6. Language Detection

### File Pattern Detection

```python
"""
Language detection for autobench.
Works on real project structures (monorepos, mixed-language repos).
"""

import re
from pathlib import Path
from typing import Optional


# Extensions per language
EXTENSIONS = {
    "python": {".py", ".pyw", ".pyi"},
    "go": {".go"},
    "rust": {".rs"},
    "c": {".c", ".h"},
    "cpp": {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"},
    "java": {".java"},
    "typescript": {".ts", ".tsx"},
    "javascript": {".js", ".jsx"},
    "ruby": {".rb"},
    "kotlin": {".kt", ".kts"},
    "swift": {".swift"},
    "zig": {".zig"},
}

# Heuristic markers for build systems / project structure
MARKERS = {
    "python": ["setup.py", "pyproject.toml", "requirements.txt", "Pipfile", "__init__.py"],
    "go": ["go.mod", "go.sum"],
    "rust": ["Cargo.toml", "Cargo.lock"],
    "c": ["Makefile", ".cmake"],
    "cpp": ["CMakeLists.txt"],
    "java": ["pom.xml", "build.gradle"],
    "typescript": ["package.json", "tsconfig.json"],
    "javascript": ["package.json"],
    "ruby": ["Gemfile"],
    "kotlin": ["build.gradle.kts"],
    "swift": ["Package.swift"],
    "zig": ["build.zig"],
}

# Shebang patterns
SHEBANG = {
    "python": re.compile(r"^#!.*python"),
    "bash": re.compile(r"^#!.*(?:ba)?sh"),
    "ruby": re.compile(r"^#!.*ruby"),
    "perl": re.compile(r"^#!.*perl"),
}


def detect_language_from_path(path: Path) -> Optional[str]:
    """
    Detect language from file path. Returns language name or None.
    """
    # Check shebang first (for scripts)
    if path.is_file():
        try:
            with open(path, "rb") as f:
                first_line = f.readline().decode("utf-8", errors="ignore")
            for lang, pattern in SHEBANG.items():
                if pattern.match(first_line.strip()):
                    return lang
        except Exception:
            pass

    # Check extension
    suffix = path.suffix.lower()
    for lang, exts in EXTENSIONS.items():
        if suffix in exts:
            return lang

    # Check directory markers
    parent = path.parent
    for lang, markers in MARKERS.items():
        for marker in markers:
            if (parent / marker).exists():
                return lang

    return None


def detect_language_from_content(code: str) -> Optional[str]:
    """
    Detect language from code content heuristics.
    """
    code_lower = code.lower()

    # Python indicators
    if "import " in code and ("def " in code or "class " in code):
        if not re.search(r"package\s+\w+", code):  # Not Java
            return "python"

    # Go indicators
    if re.search(r"^package\s+\w+", code, re.MULTILINE) and "func " in code:
        return "go"

    # Rust indicators
    if re.search(r"^use\s+\w+", code, re.MULTILINE) and ("fn " in code or "impl " in code):
        return "rust"

    # C/C++ indicators
    if "#include" in code and ("int main(" in code or "void " in code):
        return "cpp"

    # Java indicators
    if re.search(r"^public\s+(class|interface|enum)", code, re.MULTILINE):
        return "java"

    return None


def detect_language(project_root: Path, target_path: Path) -> Optional[str]:
    """
    Full language detection: path first, then content fallback.
    """
    lang = detect_language_from_path(target_path)
    if lang:
        return lang

    # Try content-based detection
    try:
        code = target_path.read_text(errors="ignore")
        lang = detect_language_from_content(code)
        if lang:
            return lang
    except Exception:
        pass

    # Scan project for dominant language
    lang_counts = {}
    for path in project_root.rglob("*"):
        if path.is_file() and not any(p in path.parts for p in [".git", "node_modules", "target", "__pycache__"]):
            lang = detect_language_from_path(path)
            if lang:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

    if lang_counts:
        return max(lang_counts, key=lang_counts.get)
    return None
```

---

## 7. Sub-Second Compilation Strategies

### Python: < 0.1s

No compilation needed. Interpreter startup is the bottleneck.

```python
import subprocess
import time

def run_python(code: str, timeout: float = 1.0) -> dict:
    start = time.perf_counter()
    result = subprocess.run(
        ["python3", "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout
    )
    elapsed = time.perf_counter() - start
    return {
        "elapsed_s": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }
```

Typical latency: **~20-50ms** for simple scripts.

### Go: ~0.3s for Small Packages

Go compilation is fast. `go run` handles single-file programs:

```bash
# Single file: ~0.1-0.2s
go run main.go

# Small package: ~0.2-0.3s
go build -o /tmp/out ./cmd/mylib && /tmp/out
```

Caching: Go modules are cached in `$(go env GOMODCACHE)`. Rebuild is fast if unchanged.

### Rust: ~2s — Build Caching Required

First build is slow (~2-10s). Subsequent builds with unchanged deps are fast (~0.5s via Cargo incremental).

**Strategy**: Pre-compile common crates, use `-C incremental`:
```bash
# First: compile dependencies once
cargo build --release --target wasm32-wasi 2>/dev/null || true

# Then: fast rebuild via cargo
cargo build  # Uses cached deps
```

For autobench: **cache compiled binaries**, don't recompile unless code changes.

### C++: Varies Wildly

| Project Size | Compile Time |
|--------------|-------------|
| Single header-only file | ~0.05s |
| Small CMake project | ~1-5s |
| Large project (FFmpeg scale) | minutes |

**Minimum viable strategy**:
- Use `clang++ -O0 -std=c++17 -c` for fast compile (no linking)
- Link with `lld -fast` or traditional linker
- For tests: use header-only libs (nlohmann/json, fmt) — no external compile needed

---

## 8. Verdict Detection

### Exit Codes and stderr Pattern Recognition

```python
"""
Verdict detection for code execution results.
Maps exit codes and stderr patterns to verdict categories.
"""

import re
from enum import Enum
from typing import Optional


class Verdict(Enum):
    ACCEPTED = "AC"
    COMPILE_ERROR = "CE"
    RUNTIME_ERROR = "RE"
    TIME_LIMIT_EXCEEDED = "TLE"
    MEMORY_LIMIT_EXCEEDED = "MLE"
    WRONG_ANSWER = "WA"
    INTERNAL_ERROR = "IE"
    TIMEOUT = "TO"  # Hard timeout (signal)


# Signal codes for Unix
SIGNALS = {
    152: "XCPU",    # SIGXCPU (exceeded CPU time)
    159: "SIGSYS",  # SIGSYS (bad syscall)
    138: "SIGALRM", # SIGALRM (timeout)
    139: "SIGSEGV", # SIGSEGV (segfault)
    136: "SIGFPE",  # SIGFPE (float error)
    137: "SIGKILL", # SIGKILL (OOM or killed)
    134: "SIGABRT", # SIGABRT (abort)
}


def detect_verdict(
    returncode: int,
    stderr: str,
    stdout: str,
    has_output: bool = False,
    expected_output: Optional[str] = None,
) -> Verdict:
    """
    Detect verdict from process exit state.

    Args:
        returncode: Process exit code
        stderr: Standard error output
        stdout: Standard output
        has_output: Whether the process produced output before exiting
        expected_output: Expected output for WA comparison (optional)

    Returns:
        Verdict enum value
    """
    stderr_lower = stderr.lower()

    # Hard timeout (signal 124 from timeout command)
    if returncode == 124:
        return Verdict.TIMEOUT

    # Signal-based exits
    if returncode < 0:
        signal_num = -returncode
        if signal_num in SIGNALS:
            sig_name = SIGNALS[signal_num]
            if sig_name == "XCPU":
                return Verdict.TIME_LIMIT_EXCEEDED
            if sig_name == "SIGKILL":
                return Verdict.MEMORY_LIMIT_EXCEEDED  # Often OOM
            if sig_name in ("SIGSEGV", "SIGFPE", "SIGABRT"):
                return Verdict.RUNTIME_ERROR

    # Compile errors
    if returncode != 0:
        compile_error_patterns = [
            r"error:",
            r"fatal error:",
            r"compilation failed",
            r"cannot find",
            r"undefined reference",
            r"no such file or directory",
            r"syntax error",
            r"traceback (most recent call last)",
            r"error\[",
            r"failed to compile",
            r"compilation error",
            r"^error",
            r"^fatal",
            r"^syntax error",
        ]
        for pattern in compile_error_patterns:
            if re.search(pattern, stderr, re.IGNORECASE | re.MULTILINE):
                return Verdict.COMPILE_ERROR

        # Runtime error (non-zero exit without compile error patterns)
        runtime_error_patterns = [
            r"exception",
            r"traceback",
            r"panic",
            r"runtime error",
            r"segmentation fault",
            r"core dumped",
            r"aborted",
            r"signal",
            r"killed",
        ]
        for pattern in runtime_error_patterns:
            if re.search(pattern, stderr, re.IGNORECASE):
                return Verdict.RUNTIME_ERROR

        return Verdict.RUNTIME_ERROR

    # Zero exit — check output
    if returncode == 0:
        if expected_output is not None:
            if stdout.strip() != expected_output.strip():
                return Verdict.WRONG_ANSWER
            return Verdict.ACCEPTED

        if has_output:
            return Verdict.ACCEPTED

        return Verdict.INTERNAL_ERROR

    return Verdict.INTERNAL_ERROR


def detect_language_errors(stderr: str) -> Optional[str]:
    """
    Detect which language's error format is in stderr.
    """
    if re.search(r"Traceback \(most recent call last\)", stderr):
        return "python"
    if re.search(r"error\[E\d+\]", stderr):
        return "rust"
    if re.search(r"error:", stderr) and re.search(r"--> \d+:\d+", stderr):
        return "cpp"
    if re.search(r"Error:", stderr) and re.search(r"at ", stderr):
        return "java"
    if re.search(r"# error", stderr):
        return "c"
    if re.search(r"cannot find", stderr):
        return "go"
    return None
```

### Key Pattern Summary

| Verdict | Detection Method |
|---------|-----------------|
| **CE** | stderr contains "error:", "fatal error", "compilation failed", "undefined reference" |
| **RE** | Non-zero exit + runtime crash patterns (segfault, panic, exception) |
| **TLE** | Signal XCPU or returncode 124 |
| **MLE** | Signal SIGKILL or "killed" in stderr |
| **WA** | Zero exit but stdout mismatch |
| **AC** | Zero exit + stdout match |

---

## Summary: Practical Autobench Architecture

For sub-second code testing:

1. **Language detection**: File extension + shebang + content heuristics
2. **Execution path**:
   - Python: `subprocess.run(["python3", "-c", code])` — ~30ms
   - Go: `go run` or cached build — ~200ms
   - Rust: Pre-cached builds, rebuild via cargo — ~500ms (cached)
   - C++: Clang compile + link — ~1-5s, use precompiled headers
3. **Isolation choices**:
   - For untrusted code: **gVisor** (syscall filtering) or **Firecracker** (hardware isolation + snapshots)
   - For trusted code (pre-approved): **namespace containers** (fastest, ~5ms startup)
4. **Verdict detection**: Exit code + stderr pattern matching as above

**Recommended stack**: gVisor + namespace pooling for warm containers, with Firecracker snapshot restore for heavy workloads.