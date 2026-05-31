"""Multi-language sandboxed execution for autobench.

Classes:
    SandboxedExecutor — runs code in gVisor/Firecracker namespace
    detect_language(project_path) — detect language stack from files
    compile_and_run(code, language, constraints) — returns (output, verdict, latency)
    verify_output(actual, expected, constraints) — returns pass/fail with verdict type

Sandbox types:
    - subprocess: raw subprocess with psutil RSS tracking and optional cgroup2 enforcement
    - gvisor: gVisor (runsc) if available on disk, falls back to subprocess
    - firecracker: reserved for future Firecracker microVM integration
    - namespace: reserved for Linux namespace isolation

cgroup2 enforcement:
    If use_cgroup=True and cgcreate is permitted, creates a temporary cgroup
    with memory.max set to the limit. Falls back to psutil RSS tracking if
    cgcreate fails with PermissionError.
"""

from __future__ import annotations

import os
import random
import re
import shutil
import select
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .core import Verdict
from .observability import AutobenchObservability


try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


LANGUAGE_MAP: dict[str, list[str]] = {
    "python": ["py", "python", "pyi"],
    "rust": ["rs"],
    "go": ["go"],
    "javascript": ["js", "mjs", "cjs"],
    "typescript": ["ts", "tsx"],
    "java": ["java"],
    "c": ["c", "h"],
    "cpp": ["cpp", "hpp", "cc", "cxx"],
    "ruby": ["rb"],
    "bash": ["sh", "bash", "zsh"],
    "php": ["php"],
    "swift": ["swift"],
    "kotlin": ["kt", "kts"],
    "zig": ["zig"],
}

LANGUAGE_RUNNERS: dict[str, dict[str, Any]] = {
    "python": {
        "cmd": ["python3"],
        "compile_cmd": None,
        "file_ext": "py",
        "timeout_default": 10,
    },
    "rust": {
        "cmd": None,  # needs compile first
        "compile_cmd": ["cargo", "build", "--release", "--quiet"],
        "file_ext": "rs",
        "timeout_default": 60,
        "run_binary": "{binary}",
    },
    "go": {
        "cmd": None,
        "compile_cmd": ["go", "build", "-o", "{binary}", "{source}"],
        "file_ext": "go",
        "timeout_default": 30,
        "run_binary": "{binary}",
    },
    "javascript": {
        "cmd": ["node"],
        "compile_cmd": None,
        "file_ext": "js",
        "timeout_default": 10,
    },
    "typescript": {
        "cmd": ["npx", "ts-node"],
        "compile_cmd": None,
        "file_ext": "ts",
        "timeout_default": 15,
    },
    "java": {
        "cmd": None,
        "compile_cmd": ["javac", "{source}"],
        "file_ext": "java",
        "timeout_default": 30,
        "run_binary": ["java", "-cp", "{dirname}", "{classname}"],
    },
    "c": {
        "cmd": None,
        "compile_cmd": ["gcc", "-o", "{binary}", "{source}", "-lm"],
        "file_ext": "c",
        "timeout_default": 30,
        "run_binary": "{binary}",
    },
    "cpp": {
        "cmd": None,
        "compile_cmd": ["g++", "-o", "{binary}", "{source}", "-std=c++17"],
        "file_ext": "cpp",
        "timeout_default": 30,
        "run_binary": "{binary}",
    },
    "ruby": {
        "cmd": ["ruby"],
        "compile_cmd": None,
        "file_ext": "rb",
        "timeout_default": 10,
    },
    "bash": {
        "cmd": ["bash"],
        "compile_cmd": None,
        "file_ext": "sh",
        "timeout_default": 30,
    },
    "php": {
        "cmd": ["php"],
        "compile_cmd": None,
        "file_ext": "php",
        "timeout_default": 10,
    },
    "swift": {
        "cmd": ["swift"],
        "compile_cmd": None,
        "file_ext": "swift",
        "timeout_default": 30,
    },
    "kotlin": {
        "cmd": None,
        "compile_cmd": ["kotlinc", "{source}", "-include-runtime", "-d", "{binary}.jar"],
        "file_ext": "kt",
        "timeout_default": 60,
        "run_binary": ["java", "-jar", "{binary}.jar"],
    },
    "zig": {
        "cmd": None,
        "compile_cmd": ["zig", "build-exe", "-O ReleaseFast", "{source}"],
        "file_ext": "zig",
        "timeout_default": 60,
        "run_binary": "{binary}",
    },
}


def cleanup_cgroup(cgroup_name: str) -> None:
    """Clean up a cgroup after subprocess completes."""
    cgroup_path = f"/sys/fs/cgroup/{cgroup_name}"
    try:
        shutil.rmtree(cgroup_path, ignore_errors=True)
    except Exception:
        pass


@dataclass
class ExecutionResult:
    """Result of a sandboxed execution."""

    stdout: str = ""
    stderr: str = ""
    verdict: Verdict = Verdict.OK
    latency_ms: float = 0.0
    exit_code: int = 0
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SandboxedExecutor:
    """Runs code in a sandboxed environment.

    Uses gVisor if available, falls back to Firecracker, then raw subprocess
    with resource limits via POSIX ulimit / cgroups.

    The executor:
    1. Writes code to a temp file
    2. Compiles if needed (Rust, Go, C, C++, Java, Kotlin, Zig)
    3. Runs with timeout and memory limits
    4. Returns output + verdict

    Sandbox types:
        - subprocess: raw subprocess with psutil RSS tracking and optional cgroup2
        - gvisor: gVisor (runsc) if available, falls back to subprocess
        - firecracker: reserved for future Firecracker microVM integration
        - namespace: reserved for Linux namespace isolation
    """

    def __init__(
        self,
        sandbox_type: Literal["subprocess", "gvisor", "firecracker", "namespace"] = "subprocess",
        max_memory_mb: int = 512,
        cpu_limit: int = 2,
        use_cgroup: bool = False,
        obs: AutobenchObservability | None = None,
    ):
        self.sandbox_type = sandbox_type
        self.max_memory_mb = max_memory_mb
        self.cpu_limit = cpu_limit
        self.use_cgroup = use_cgroup
        self.obs = obs
        self._tmpdir = tempfile.mkdtemp(prefix="autobench_sandbox_")

        # Check for gVisor availability
        self._runsc_path: str | None = None
        if self.sandbox_type == "gvisor":
            self._runsc_path = self._find_runsc()
            available, version, issues = self.check_gvisor()
            if not available:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"gVisor unavailable: {issues}. Falling back to subprocess.")
                self.sandbox_type = "subprocess"
                self._runsc_path = None

        # Check for Firecracker / KVM
        self._firecracker_pool = None
        if self.sandbox_type == "firecracker":
            if os.path.exists("/dev/kvm") and os.access("/dev/kvm", os.R_OK | os.W_OK):
                try:
                    from .firecracker_vm import FirecrackerPool
                    self._firecracker_pool = FirecrackerPool(pool_size=4)
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Firecracker pool init failed: {e}. Falling back to subprocess.")
                    self.sandbox_type = "subprocess"
            else:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("Firecracker requires /dev/kvm. Falling back to subprocess.")
                self.sandbox_type = "subprocess"

    def _gvisor_cmd_prefix(self) -> list[str]:
        """Build the runsc command prefix used for every gVisor invocation.

        Single source of truth — both check_gvisor() and _run_subprocess()
        must use this so a passing health check actually validates the
        same code path used for real executions.

        We use the ptrace platform (not KVM) because it works without
        /dev/kvm and without root privileges. ptrace is slower than KVM
        (~10x for syscall-heavy workloads) but adequate for short
        benchmark runs and removes a dependency on host KVM access.

        --rootless        — run as the invoking user, no setuid required
        --network=none    — no external network access
        --ignore-cgroups  — don't try to manage cgroups (would require root)
        --platform=ptrace — ptrace-based syscall interception (no KVM needed)

        A finer-grained seccomp filter would require a real OCI seccomp
        profile (JSON) passed via the OCI spec — the prior `.conf` DSL
        was not a real runsc input format and silently did nothing.
        Leaving that as a follow-up (see beads); gVisor itself already
        applies a minimal Sentry seccomp filter to the host kernel.
        """
        return [
            self._runsc_path,
            "--rootless",
            "--network=none",
            "--ignore-cgroups",
            "--platform=ptrace",
            "do",
            "--",
        ]

    def check_gvisor(self) -> tuple[bool, str, list[str]]:
        """Check gVisor availability and health.

        Returns:
            Tuple of (available: bool, version: str, issues: list[str]).
            available is True only if runsc is present AND can execute a
            simple command AND can execute python (the most common runtime).
        """
        issues = []
        if not self._runsc_path:
            return False, "", ["runsc binary not found at /usr/local/bin/runsc or /usr/bin/runsc"]

        # Check runsc --version
        try:
            result = subprocess.run(
                [self._runsc_path, "--version"],
                timeout=5,
                capture_output=True,
            )
            version = result.stdout.strip() or result.stderr.strip() or "unknown"
            if isinstance(version, bytes):
                version = version.decode(errors="replace")
        except Exception as e:
            issues.append(f"runsc --version failed: {e}")
            return False, "", issues

        # Quick echo exec test — uses the SAME prefix as _run_subprocess
        try:
            cmd = self._gvisor_cmd_prefix() + ["echo", "ok"]
            result = subprocess.run(cmd, timeout=10, capture_output=True)
            stdout = result.stdout.decode() if isinstance(result.stdout, bytes) else result.stdout
            if result.returncode != 0 or 'ok' not in stdout:
                stderr = result.stderr.decode() if isinstance(result.stderr, bytes) else result.stderr
                issues.append(f'runsc echo test failed (exit {result.returncode}): {stderr[:200]}')
                return False, version, issues
        except Exception as e:
            issues.append(f"runsc echo test failed: {e}")
            return False, version, issues

        # Python exec test — verifies language-level execution path, not
        # just echo. Catches missing /usr/bin/python3 in the sandbox view,
        # broken filesystem mounts, etc.
        py_ok, py_err = self._test_gvisor_python()
        if not py_ok:
            issues.append(py_err)
            return False, version, issues

        return True, version, []

    def _test_gvisor_python(self) -> tuple[bool, str]:
        """Run a trivial python3 command under runsc to verify execution.

        Returns (True, "") on success, (False, reason) on failure.
        """
        try:
            cmd = self._gvisor_cmd_prefix() + ["python3", "-c", "print('ok')"]
            result = subprocess.run(cmd, timeout=15, capture_output=True)
            stdout = result.stdout.decode() if isinstance(result.stdout, bytes) else result.stdout
            if result.returncode != 0 or 'ok' not in stdout:
                stderr = result.stderr.decode() if isinstance(result.stderr, bytes) else result.stderr
                return False, f"runsc python test failed (exit {result.returncode}): {stderr[:200]}"
        except Exception as e:
            return False, f"runsc python test failed: {e}"
        return True, ""

    def _find_runsc(self) -> str | None:
        """Locate runsc binary if present on disk."""
        for path in ["/usr/local/bin/runsc", "/usr/bin/runsc"]:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    # TODO(Firecracker): Deferred. Firecracker requires KVM device nodes and ~200ms
    # cold-boot cost — too slow for RSI loops that run 30-200 iterations per
    # convergence window. Revisit after gVisor is validated. See: arXiv 2604.14004
    # for cross-model harness transfer benchmarks.
    def __del__(self):
        # Cleanup temp directory
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def execute(
        self,
        code: str,
        language: str,
        constraints: dict[str, Any] | None = None,
        stdin: str = "",
        max_memory_mb: int | None = None,
        case_id: str | None = None,
    ) -> ExecutionResult:
        """Execute code in the sandbox.

        Args:
            code: Source code to execute.
            language: Language identifier (python, rust, go, etc.).
            constraints: Dict with max_time_seconds, max_memory_mb, expected_output.
            stdin: Optional stdin input.
            max_memory_mb: Override for max memory limit in MB.
            case_id: Optional case identifier for observability emission.

        Returns:
            ExecutionResult with stdout, stderr, verdict, latency_ms.
        """
        constraints = constraints or {}
        lang_lower = language.lower()
        lang_config = LANGUAGE_RUNNERS.get(lang_lower, LANGUAGE_RUNNERS["python"])
        emit_case_id = case_id if case_id is not None else ""

        if self.obs is not None:
            self.obs.sandbox_dispatch(
                case_id=emit_case_id,
                language=lang_lower,
                sandbox_type=self.sandbox_type,
            )

        result = self._execute_inner(
            code=code,
            lang_lower=lang_lower,
            lang_config=lang_config,
            constraints=constraints,
            stdin=stdin,
            max_memory_mb=max_memory_mb,
        )

        if self.obs is not None:
            self.obs.sandbox_complete(
                case_id=emit_case_id,
                verdict=result.verdict.value,
                latency_ms=result.latency_ms,
                exit_code=result.exit_code,
                language=lang_lower,
                sandbox_type=self.sandbox_type,
            )
        return result

    def _execute_inner(
        self,
        code: str,
        lang_lower: str,
        lang_config: dict,
        constraints: dict[str, Any],
        stdin: str,
        max_memory_mb: int | None,
    ) -> ExecutionResult:
        """Core execute body — split out so observability emits wrap it."""

        ext = lang_config["file_ext"]
        source_path = os.path.join(self._tmpdir, f"main.{ext}")
        binary_path = os.path.join(self._tmpdir, "main")
        if lang_lower == "java":
            classname = self._extract_java_classname(code)
            source_path = os.path.join(self._tmpdir, f"{classname}.java")

        # Write source
        Path(source_path).write_text(code)

        max_time = constraints.get("max_time_seconds", lang_config.get("timeout_default", 30))
        effective_max_memory = max_memory_mb if max_memory_mb is not None else constraints.get("max_memory_mb", self.max_memory_mb)

        start = time.perf_counter()
        verdict = Verdict.OK
        stderr = ""
        exit_code = 0

        # Compile if needed
        if lang_config.get("compile_cmd"):
            compile_cmd = self._expand_cmd(lang_config["compile_cmd"], source_path, binary_path)
            try:
                result = self._run_subprocess(
                    compile_cmd,
                    timeout=max(max_time, 60),
                    max_memory_mb=effective_max_memory,
                    sandboxed=False,  # compile on host; run the binary under gVisor
                )
                stderr += result.stderr
                if result.returncode != 0:
                    return ExecutionResult(
                        stderr=stderr,
                        verdict=Verdict.CE,
                        latency_ms=(time.perf_counter() - start) * 1000,
                        exit_code=result.returncode,
                    )
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    stderr="Compilation timed out",
                    verdict=Verdict.TLE,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    exit_code=-1,
                )

        # Run
        run_cmd = self._build_run_cmd(lang_config, source_path, binary_path)

        # Firecracker path — use VM pool for execution
        if self.sandbox_type == "firecracker" and self._firecracker_pool is not None:
            try:
                start_vm = time.perf_counter()
                fc_vm = self._firecracker_pool.acquire(timeout=max_time)
                fc_result = fc_vm.run(run_cmd, timeout=max_time)
                self._firecracker_pool.release(fc_vm)
                elapsed = (time.perf_counter() - start_vm) * 1000
                return ExecutionResult(
                    stdout=fc_result.stdout,
                    stderr=fc_result.stderr,
                    verdict=Verdict.OK if fc_result.returncode == 0 else Verdict.RE,
                    latency_ms=elapsed,
                    exit_code=fc_result.returncode,
                )
            except Exception as e:
                # Firecracker failed — fall back to subprocess
                import logging
                logging.getLogger(__name__).warning(f"Firecracker execution failed: {e}. Falling back to subprocess.")

        try:
            result = self._run_subprocess(
                run_cmd,
                timeout=max_time,
                max_memory_mb=effective_max_memory,
                stdin_input=stdin,
            )
            stdout = result.stdout
            stderr += result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stderr="Execution timed out",
                verdict=Verdict.TLE,
                latency_ms=(time.perf_counter() - start) * 1000,
                exit_code=-1,
            )

        # Check for MLE from RSS kill before other verdict logic
        # When _run_subprocess kills due to memory, it returns returncode=-1
        # and stderr containing "Memory limit exceeded"
        if "Memory limit exceeded" in stderr:
            verdict = Verdict.MLE

        # Detect runtime errors from stderr (only if not already MLE)
        elif exit_code != 0 and verdict == Verdict.OK:
            if self._is_runtime_error(stderr, lang_lower):
                verdict = Verdict.RE
            else:
                verdict = Verdict.WA

        # Check for memory issues from other sources (OOM killer, etc.)
        elif self._looks_like_memory_error(stderr):
            verdict = Verdict.MLE

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            verdict=verdict,
            latency_ms=(time.perf_counter() - start) * 1000,
            exit_code=exit_code,
        )

    def _build_run_cmd(self, lang_config: dict, source_path: str, binary_path: str) -> list[str]:
        """Build the run command for a language config."""
        cmd_template = lang_config.get("run_binary") or lang_config.get("cmd")
        if isinstance(cmd_template, list):
            expanded = self._expand_cmd(cmd_template, source_path, binary_path)
            # For interpreted languages (no compile_cmd), the source must be passed
            # to the interpreter. If no placeholder was used, append source_path.
            if lang_config.get("compile_cmd") is None and source_path not in expanded:
                expanded = expanded + [source_path]
            return expanded
        elif isinstance(cmd_template, str):
            expanded = cmd_template.format(binary=binary_path, source=source_path)
            return expanded.split()
        else:
            # Fallback: direct interpreter
            return lang_config["cmd"]

    def _expand_cmd(
        self,
        cmd: list[str] | str,
        source_path: str,
        binary_path: str,
    ) -> list[str]:
        """Expand {source}, {binary}, {dirname}, {classname} placeholders."""
        dirname = os.path.dirname(source_path)
        classname = self._extract_java_classname(Path(source_path).read_text()) if "java" in str(cmd) else ""

        if isinstance(cmd, list):
            result = []
            for token in cmd:
                token = token.replace("{source}", source_path)
                token = token.replace("{binary}", binary_path)
                token = token.replace("{dirname}", dirname)
                token = token.replace("{classname}", classname)
                result.append(token)
            return result
        else:
            cmd = cmd.replace("{source}", source_path)
            cmd = cmd.replace("{binary}", binary_path)
            cmd = cmd.replace("{dirname}", dirname)
            cmd = cmd.replace("{classname}", classname)
            return cmd.split()

    def _run_subprocess(
        self,
        cmd: list[str],
        timeout: int,
        max_memory_mb: int,
        stdin_input: str = "",
        sandboxed: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a command with resource limits.

        Supports:
        - psutil RSS tracking: monitors process RSS and kills on MLE if exceeded
        - cgroup2 enforcement: if use_cgroup=True and cgcreate succeeds, hard memory limit
        - gVisor: if sandbox_type=gvisor and runsc found, run via runsc

        ``sandboxed`` controls whether the gVisor wrap is applied. The COMPILE
        step passes sandboxed=False: each ``runsc do`` is an independent sandbox
        with an ephemeral overlay filesystem, so a binary produced by g++ inside
        one ``runsc do`` does NOT persist to the host for a separate run-step
        ``runsc do`` to load (it fails with "failed to load main: no such file").
        We therefore compile on the host — the untrusted input is source-only and
        the compiler flags are fixed (no -fplugin injection), g++ does not execute
        the program at compile time, and the produced binary is then EXECUTED
        under gVisor with --network=none. Resource limits + timeout still apply to
        the host compile, bounding include/template bombs.
        """
        env = os.environ.copy()
        env["RUST_BACKTRACES"] = "1"

        # Check for gVisor
        #
        # We deliberately do NOT pass --seccomp-policy. The previous
        # autobench/gvisor_policy.conf used a DSL (`%nestsandbox`, bare
        # syscall names) that runsc does not consume — it expects an OCI
        # spec seccomp profile in JSON. Passing the .conf file silently
        # did nothing. Isolation today comes from runsc's built-in Sentry
        # + --network=none + ptrace platform. A real seccomp profile is a
        # follow-up; see _gvisor_cmd_prefix() docstring.
        #
        # We do NOT allocate a PTY: `runsc do` works fine with plain pipes
        # for the workloads autobench runs (one-shot python/rust/go
        # processes that print to stdout and exit). PTYs complicate stdin
        # routing and timeout cleanup, so dropping them keeps the path
        # identical to the subprocess fallback.
        use_pty = False
        if sandboxed and self.sandbox_type == "gvisor" and self._runsc_path:
            cmd = self._gvisor_cmd_prefix() + cmd
            import logging
            logging.getLogger(__name__).debug(
                "gvisor exec: %s", " ".join(cmd)
            )

        cgroup_cleanup = None
        cgroup_prepared_path = None  # path of created cgroup, before child PID is assigned
        proc = None

        # PTY setup for gVisor
        master_fd = None
        slave_stdin = None
        if use_pty:
            import pty
            master_fd, slave_stdin = pty.openpty()

        try:
            # Prepare cgroup2 (create + set memory.max) — but DO NOT add parent PID.
            # Adding os.getpid() here would put the Python test runner under the
            # unit-under-test's memory limit, causing the runner itself to be
            # OOM-killed under load (Angela's audit, vmho).
            if self.use_cgroup and HAS_PSUTIL:
                cgroup_name = None
                try:
                    cgroup_name = f"autobench_{random.randint(0, 0xFFFFFFFFFF):010x}"
                    cgroup_path = f"/sys/fs/cgroup/{cgroup_name}"
                    os.makedirs(cgroup_path, exist_ok=True)

                    with open(f"{cgroup_path}/memory.max", "w") as f:
                        f.write(str(max_memory_mb * 1024 * 1024))

                    cgroup_prepared_path = cgroup_path
                    cgroup_cleanup = cgroup_name
                    cgroup_name = None  # success — no cleanup needed on exception path
                except Exception:
                    # Cleanup partial cgroup on any failure after makedirs succeeded
                    if cgroup_name:
                        try:
                            shutil.rmtree(f"/sys/fs/cgroup/{cgroup_name}", ignore_errors=True)
                        except Exception:
                            pass
                    # Fall back to psutil tracking only
                    cgroup_cleanup = None
                    cgroup_prepared_path = None

            # Start subprocess. In the non-PTY path we open a stdin PIPE so the
            # caller's stdin_input is actually delivered (previously stdin was
            # None and stdin_input was silently dropped — see below).
            stdin_fd = slave_stdin if use_pty else subprocess.PIPE
            proc = subprocess.Popen(
                cmd,
                stdin=stdin_fd,
                stdout=subprocess.PIPE if not use_pty else slave_stdin,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True,
            )

            # Deliver stdin and close it (EOF) BEFORE the RSS-polling loop, so a
            # process that reads stdin can proceed. We write inline rather than at
            # communicate() time because communicate() runs only after the poll
            # loop, which would deadlock a stdin-reading child. This assumes the
            # child drains stdin before emitting a stdout pipe-buffer's worth of
            # output (true for autobench's read-all-then-solve workloads).
            if not use_pty and proc.stdin is not None:
                try:
                    if stdin_input:
                        proc.stdin.write(stdin_input)
                    proc.stdin.close()
                except (BrokenPipeError, ValueError, OSError):
                    pass
                # Detach so the later communicate() doesn't flush a closed handle.
                proc.stdin = None

            # Assign the CHILD PID (not parent) to the cgroup. There is a small
            # race window between Popen()/fork+exec and this write where the
            # child runs uncontained — for autobench's tier-1 CodeForces
            # solutions this is sub-millisecond and negligible. psutil RSS
            # tracking below still catches runaway children if the write fails.
            if cgroup_prepared_path is not None:
                try:
                    with open(f"{cgroup_prepared_path}/cgroup.procs", "w") as f:
                        f.write(str(proc.pid))
                except Exception:
                    # Cgroup assignment failed — fall back to psutil-only enforcement.
                    # Don't kill the child; psutil polling below will enforce the limit.
                    pass

            # Close slave PTY in parent — only needed in child
            if use_pty and slave_stdin is not None:
                os.close(slave_stdin)

            # Track RSS via psutil if available
            peak_rss = 0
            ps_proc = None
            if HAS_PSUTIL:
                try:
                    ps_proc = psutil.Process(proc.pid)
                except psutil.NoSuchProcess:
                    pass

            # Poll with timeout
            start_time = time.time()
            while True:
                # Check if process exited
                retcode = proc.poll()
                if retcode is not None:
                    break

                # Check timeout
                if time.time() - start_time > timeout:
                    proc.kill()
                    proc.wait()
                    raise subprocess.TimeoutExpired(cmd, timeout)

                # Check RSS via psutil
                if ps_proc is not None:
                    try:
                        mem_info = ps_proc.memory_info()
                        rss = mem_info.rss
                        peak_rss = max(peak_rss, rss)

                        # Convert MB for comparison
                        rss_mb = rss / (1024 * 1024)
                        if rss_mb > max_memory_mb:
                            proc.kill()
                            proc.wait()
                            # Return a fake CompletedProcess to indicate MLE
                            return subprocess.CompletedProcess(
                                args=cmd,
                                returncode=-1,
                                stdout="",
                                stderr=f"Memory limit exceeded: {rss_mb:.1f}MB > {max_memory_mb}MB (RSS)",
                            )
                    except psutil.NoSuchProcess:
                        # Process died between checks
                        break

                time.sleep(0.05)

            # Wait for process to finish
            if not use_pty:
                stdout, stderr = proc.communicate()
            else:
                # Read stdout from PTY master fd
                stdout_chunks = []
                while True:
                    retcode = proc.poll()
                    try:
                        r, _, _ = select.select([master_fd], [], [], 0.1)
                        if r:
                            data = os.read(master_fd, 4096)
                            if data:
                                stdout_chunks.append(data)
                            else:
                                break
                        elif retcode is not None:
                            # Process exited and no more data pending - drain any final data
                            break
                    except OSError:
                        break
                    time.sleep(0.05)

                # Drain any remaining data after process exits
                proc.wait()
                try:
                    while True:
                        r, _, _ = select.select([master_fd], [], [], 0.1)
                        if not r:
                            break
                        data = os.read(master_fd, 4096)
                        if not data:
                            break
                        stdout_chunks.append(data)
                except OSError:
                    pass

                stdout = b"".join(stdout_chunks).decode()
                _, stderr = proc.communicate()  # stderr still via pipe

            # Check peak RSS after completion
            if peak_rss > max_memory_mb * 1024 * 1024 and proc.returncode == 0:
                # It was killed but we didn't get the returncode - could be OOM killed externally
                pass

            return subprocess.CompletedProcess(
                args=cmd,
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
            )

        finally:
            # Cleanup cgroup
            if cgroup_cleanup:
                try:
                    shutil.rmtree(f"/sys/fs/cgroup/{cgroup_cleanup}", ignore_errors=True)
                except Exception:
                    pass

            # Close PTY master
            if master_fd is not None:
                try:
                    os.close(master_fd)
                except Exception:
                    pass

            # Ensure process is dead
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()

    def _extract_java_classname(self, source: str) -> str:
        """Extract public class name from Java source."""
        match = re.search(r"public\s+class\s+(\w+)", source)
        if match:
            return match.group(1)
        return "Main"

    def _is_runtime_error(self, stderr: str, language: str) -> bool:
        """Heuristic check for runtime error patterns in stderr."""
        patterns = {
            "python": [
                r"Traceback \(most recent call last\)",
                r"RuntimeError",
                r"TypeError",
                r"ValueError",
                r"AttributeError",
                r"IndexError",
                r"KeyError",
                r"NameError",
            ],
            "rust": [
                r"thread '.*' panicked",
                r"panicked at",
                r"error\[E\d+\]",
                r"rust_backtrace",
            ],
            "go": [
                r"panic:",
                r"runtime error:",
                r"fatal error:",
            ],
            "javascript": [
                r"ReferenceError:",
                r"TypeError:",
                r"SyntaxError:",
                r"RangeError:",
            ],
            "java": [
                r"Exception in thread",
                r"java\.\w+Exception",
                r"Error:",
            ],
            "c": [
                r"Segmentation fault",
                r"core dumped",
                r"SIGSEGV",
            ],
            "cpp": [
                r"Segmentation fault",
                r"core dumped",
                r"SIGSEGV",
                r"std::exception",
                r"terminate called",
            ],
        }
        lang_patterns = patterns.get(language, patterns["python"])
        for pat in lang_patterns:
            if re.search(pat, stderr, re.IGNORECASE):
                return True
        return False

    def _looks_like_memory_error(self, stderr: str) -> bool:
        """Heuristic check for memory limit exceeded in output."""
        mem_patterns = [
            r"out of memory",
            r"MemoryError",
            r"cannot allocate",
            r"fatal: unable to allocate",
            r"std::bad_alloc",
            r"OutOfMemoryError",
            r"Heap OutOfMemoryError",
            r"Killed",
            r"oom_kill",
        ]
        for pat in mem_patterns:
            if re.search(pat, stderr, re.IGNORECASE):
                return True
        return False


def detect_language(project_path: str | Path) -> str | None:
    """Detect the primary language of a project from its files.

    Uses a weighted extension count. Directories are recursed up to
    a depth of 3 to find the most common language marker.

    Returns:
        Language name (python, rust, go, etc.) or None if undetected.
    """
    project_path = Path(project_path)
    if not project_path.exists():
        return None

    ext_counts: dict[str, int] = {}
    dirs_to_scan = [project_path]
    max_depth = 3

    for _ in range(max_depth):
        next_dirs = []
        for d in dirs_to_scan:
            if not d.exists():
                continue
            try:
                for entry in os.scandir(d):
                    if entry.is_dir():
                        # Skip common non-source dirs
                        if entry.name in {".git", "node_modules", "__pycache__", "target", ".venv", "venv"}:
                            continue
                        next_dirs.append(Path(entry.path))
                    elif entry.is_file():
                        ext = entry.name.rsplit(".", 1)[-1] if "." in entry.name else ""
                        for lang, exts in LANGUAGE_MAP.items():
                            if ext in exts:
                                ext_counts[lang] = ext_counts.get(lang, 0) + 1
            except PermissionError:
                continue
        dirs_to_scan = next_dirs

    if not ext_counts:
        return None

    return max(ext_counts, key=ext_counts.get)


def compile_and_run(
    code: str,
    language: str,
    constraints: dict[str, Any] | None = None,
    stdin: str = "",
    executor: SandboxedExecutor | None = None,
) -> tuple[str, Verdict, float]:
    """Convenience function wrapping SandboxedExecutor.execute().

    Args:
        code: Source code.
        language: Language identifier.
        constraints: max_time_seconds, max_memory_mb, expected_output.
        stdin: Optional stdin.
        executor: Optional pre-created executor (reuses temp dir).

    Returns:
        Tuple of (stdout, verdict, latency_ms).
    """
    if executor is None:
        executor = SandboxedExecutor()

    result = executor.execute(code, language, constraints, stdin)
    return result.stdout, result.verdict, result.latency_ms


def verify_output(
    actual: str,
    expected: str,
    constraints: dict[str, Any] | None = None,
) -> tuple[bool, Verdict]:
    """Verify actual output against expected output.

    Args:
        actual: stdout from execution.
        expected: Expected output (exact string, or JSON/YAML for structured comparison).
        constraints: Options like ignore_whitespace, normalize_float, exact_match.

    Returns:
        Tuple of (pass: bool, verdict: Verdict).
    """
    constraints = constraints or {}
    verdict = Verdict.OK

    if constraints.get("ignore_whitespace", False):
        actual = " ".join(actual.split())
        expected = " ".join(expected.split())

    if constraints.get("normalize_float", False):
        actual = _normalize_floats(actual)
        expected = _normalize_floats(expected)

    if constraints.get("parse_json", False):
        try:
            import json

            actual = json.loads(actual)
            expected = json.loads(expected)
            matches = actual == expected
        except json.JSONDecodeError:
            matches = False
    elif constraints.get("parse_yaml", False):
        try:
            import yaml

            actual = yaml.safe_load(actual)
            expected = yaml.safe_load(expected)
            matches = actual == expected
        except ImportError:
            # Fallback: string compare
            matches = actual.strip() == expected.strip()
    else:
        matches = actual.strip() == expected.strip()

    if not matches:
        verdict = Verdict.WA

    return matches, verdict


def _normalize_floats(s: str, decimals: int = 6) -> str:
    """Replace floating point numbers with normalized form for comparison."""
    return re.sub(r"\d+\.\d+", lambda m: f"{float(m.group()):.{decimals}g}", s)
