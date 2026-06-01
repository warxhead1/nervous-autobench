"""Integration layer with deer-flow for autobench."""

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RoleSpec:
    """Role specification for conditional routing."""
    name: str
    condition: Callable[[dict], bool] = field(default=lambda _: True)
    capability: str = ""

    def matches(self, context: dict) -> bool:
        """Check if this role matches the given context."""
        return self.condition(context)


@dataclass
class BudgetViolation:
    """Represents a budget violation event."""
    timestamp: str
    agent_id: str
    budget_type: str
    limit: float
    actual: float
    excess_ratio: float
    policy: str = "default"


def activation_predicate(roles: list[RoleSpec], autobench_result: dict) -> list[str]:
    """
    Build conditional routing predicate from autobench results.

    Given a list of RoleSpecs and an autobench result, returns the list of
    role names that should be activated based on the result.

    Args:
        roles: List of RoleSpec objects defining available roles
        autobench_result: Dict containing autobench evaluation results

    Returns:
        List of role names that match the current context
    """
    activated = []
    context = _build_context(autobench_result)

    for role in roles:
        if role.matches(context):
            activated.append(role.name)

    return activated


def _build_context(result: dict) -> dict:
    """Build routing context from autobench result."""
    return {
        "verdict": result.get("verdict", "unknown"),
        "quality": result.get("quality", 0.0),
        "cost": result.get("cost", 0.0),
        "time": result.get("time", 0.0),
        "change_type": result.get("change_type", "unknown"),
        "utilization": result.get("utilization", {}),
    }


class BudgetViolationMiddleware:
    """
    Middleware to detect token excess and emit bus events.

    Monitors agent execution for budget violations (token limit exceeded,
    time limit exceeded, etc.) and logs policy violations while optionally
    emitting events to the nervous-bus.
    """

    def __init__(
        self,
        token_limit: Optional[float] = None,
        time_limit: Optional[float] = None,
        cost_limit: Optional[float] = None,
        bus_publisher: Optional["NervousBusPublisher"] = None,
        policy: str = "default",
    ):
        """
        Initialize budget violation middleware.

        Args:
            token_limit: Maximum tokens allowed (None = unlimited)
            time_limit: Maximum time in seconds (None = unlimited)
            cost_limit: Maximum cost in dollars (None = unlimited)
            bus_publisher: Optional publisher for bus events
            policy: Policy name for violation events
        """
        self.token_limit = token_limit
        self.time_limit = time_limit
        self.cost_limit = cost_limit
        self.bus_publisher = bus_publisher
        self.policy = policy
        self._violations: list[BudgetViolation] = []

    def check(self, agent_id: str, metrics: dict) -> Optional[BudgetViolation]:
        """
        Check for budget violations.

        Args:
            agent_id: Identifier for the agent being checked
            metrics: Dict containing usage metrics (tokens, time, cost)

        Returns:
            BudgetViolation if detected, None otherwise
        """
        violation = None

        if self.token_limit is not None:
            tokens = metrics.get("tokens", 0)
            if tokens > self.token_limit:
                excess_ratio = (tokens - self.token_limit) / self.token_limit
                violation = BudgetViolation(
                    timestamp=_rfc3339_now(),
                    agent_id=agent_id,
                    budget_type="tokens",
                    limit=self.token_limit,
                    actual=tokens,
                    excess_ratio=excess_ratio,
                    policy=self.policy,
                )

        if self.time_limit is not None:
            elapsed = metrics.get("elapsed_seconds", 0)
            if elapsed > self.time_limit:
                excess_ratio = (elapsed - self.time_limit) / self.time_limit
                violation = BudgetViolation(
                    timestamp=_rfc3339_now(),
                    agent_id=agent_id,
                    budget_type="time",
                    limit=self.time_limit,
                    actual=elapsed,
                    excess_ratio=excess_ratio,
                    policy=self.policy,
                )

        if self.cost_limit is not None:
            cost = metrics.get("cost", 0.0)
            if cost > self.cost_limit:
                excess_ratio = (cost - self.cost_limit) / self.cost_limit
                violation = BudgetViolation(
                    timestamp=_rfc3339_now(),
                    agent_id=agent_id,
                    budget_type="cost",
                    limit=self.cost_limit,
                    actual=cost,
                    excess_ratio=excess_ratio,
                    policy=self.policy,
                )

        if violation:
            self._violations.append(violation)
            self._log_violation(violation)
            if self.bus_publisher:
                self._emit_violation(violation)

        return violation

    def _log_violation(self, violation: BudgetViolation) -> None:
        """Log a policy violation."""
        logger.warning(
            f"Budget violation [{violation.policy}]: "
            f"{violation.agent_id} exceeded {violation.budget_type} "
            f"(limit={violation.limit}, actual={violation.actual:.2f}, "
            f"excess={violation.excess_ratio:.1%})"
        )

    def _emit_violation(self, violation: BudgetViolation) -> None:
        """Emit violation event to nervous-bus."""
        if self.bus_publisher:
            try:
                self.bus_publisher.publish(
                    channel="autobench.budget.violation",
                    event={
                        "type": "autobench.budget.violation",
                        "timestamp": violation.timestamp,
                        "agent_id": violation.agent_id,
                        "budget_type": violation.budget_type,
                        "limit": violation.limit,
                        "actual": violation.actual,
                        "excess_ratio": violation.excess_ratio,
                        "policy": violation.policy,
                    },
                )
            except Exception as e:
                logger.error(f"Failed to emit violation to bus: {e}")

    @property
    def violations(self) -> list[BudgetViolation]:
        """Return all recorded violations."""
        return self._violations.copy()


class DeerFlowEvaluator:
    """
    Wraps deer-flow supervisor→subagent topology as harness executor.

    Provides a harness execution interface that translates autobench
    concepts to deer-flow's multi-agent topology for recursive
    self-improvement evaluation.
    """

    def __init__(
        self,
        supervisor_config: Optional[dict] = None,
        subagent_configs: Optional[list[dict]] = None,
        project_root: Optional[Path] = None,
    ):
        """
        Initialize deer-flow evaluator.

        Args:
            supervisor_config: Configuration for supervisor agent
            subagent_configs: List of subagent configurations
            project_root: Root path for project context
        """
        self.supervisor_config = supervisor_config or {}
        self.subagent_configs = subagent_configs or []
        self.project_root = project_root or Path.cwd()
        self._results: list[dict] = []

    def run_harness(
        self,
        benchmark_path: Path,
        harness_config: dict,
    ) -> dict[str, Any]:
        """
        Execute a harness on a benchmark using deer-flow topology.

        Args:
            benchmark_path: Path to benchmark definition
            harness_config: Harness configuration

        Returns:
            Dict containing execution results and metrics
        """
        benchmark = json.load(open(benchmark_path)) if benchmark_path.exists() else {}

        print(f"DeerFlow evaluator: running harness on {benchmark_path}")
        print(f"  Supervisor: {self.supervisor_config.get('model', 'default')}")
        print(f"  Subagents: {len(self.subagent_configs)}")

        start_time = time.time()

        cycle_result = self._run_deer_cycle(benchmark, harness_config)

        elapsed = time.time() - start_time

        result = {
            "verdict": cycle_result.get("verdict", "unknown"),
            "quality": cycle_result.get("quality", 0.0),
            "cost": cycle_result.get("cost", 0.0),
            "time": elapsed,
            "utilization": cycle_result.get("utilization", {}),
            "iterations": cycle_result.get("iterations", 1),
            "benchmark": str(benchmark_path),
        }

        self._results.append(result)
        return result

    def _run_deer_cycle(self, benchmark: dict, harness_config: dict) -> dict:
        """
        Run a deer-flow cycle for the benchmark.

        Invokes deer-flow's meta-probe cycle to perform recursive
        self-improvement on the given benchmark.
        """
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "deer_flow.meta_probe",
                    "--benchmark", json.dumps(benchmark),
                    "--harness", json.dumps(harness_config),
                ],
                capture_output=True,
                text=True,
                timeout=harness_config.get("timeout", 300),
            )

            if result.returncode == 0:
                return json.loads(result.stdout) if result.stdout else {}
            else:
                return {
                    "verdict": "RE",
                    "quality": 0.0,
                    "cost": 0.0,
                    "utilization": {},
                    "error": result.stderr,
                }
        except subprocess.TimeoutExpired:
            return {"verdict": "TLE", "quality": 0.0, "cost": 0.0, "utilization": {}}
        except FileNotFoundError:
            return self._run_fallback(benchmark, harness_config)
        except json.JSONDecodeError:
            return {"verdict": "RE", "quality": 0.0, "cost": 0.0, "utilization": {}}

    def _run_fallback(self, benchmark: dict, harness_config: dict) -> dict:
        """Fallback when deer-flow is not available."""
        return {
            "verdict": "unknown",
            "quality": 0.0,
            "cost": 0.0,
            "utilization": {},
            "error": "deer-flow not available",
        }

    @property
    def results(self) -> list[dict]:
        """Return all harness execution results."""
        return self._results.copy()


class NervousBusPublisher:
    """
    Publishes autobench results to nervous-bus.

    Translates autobench evaluation results into CloudEvents-lite
    format and publishes them to the nervous-bus plugin via
    the shell SDK.
    """

    def __init__(self, sdk_path: Optional[Path] = None, project: str = "autobench"):
        """
        Initialize nervous-bus publisher.

        Args:
            sdk_path: Path to nervous shell SDK (auto-detected if None)
            project: Project name for source URI
        """
        self.sdk_path = sdk_path or self._detect_sdk_path()
        self.project = project
        self._channel = "autobench.result"

    def _detect_sdk_path(self) -> Path:
        """Detect the nervous shell SDK path."""
        import shutil
        # Prefer `nervous` on PATH (installed via nervous setup or ~/.local/bin symlink).
        if shutil.which("nervous"):
            return Path(shutil.which("nervous"))
        # Fall back to sibling nervous-bus checkout (submodule layout).
        candidates = [
            Path(__file__).parent.parent / "sdk" / "shell" / "nervous",
            Path("sdk/shell/nervous"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def publish(
        self,
        channel: str,
        event: dict,
        source_uri: Optional[str] = None,
    ) -> bool:
        """
        Publish an event to the nervous-bus.

        Args:
            channel: Channel name (e.g., 'autobench.result')
            event: Event data dict
            source_uri: Optional source URI override

        Returns:
            True if published successfully, False otherwise
        """
        source_uri = source_uri or f"/{self.project}/result"

        cloud_event = {
            "specversion": "1.0",
            "type": channel,
            "source": source_uri,
            "id": _generate_ulid(),
            "time": _rfc3339_now(),
            "data": event,
        }

        jsonl = json.dumps(cloud_event)

        try:
            result = subprocess.run(
                [str(self.sdk_path), "publish", "--channel", channel],
                input=jsonl,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to publish to nervous-bus: {e}")
            return False

    def publish_result(self, result: dict) -> bool:
        """
        Publish an autobench result event.

        Args:
            result: Dict containing result fields (verdict, quality, cost, time, etc.)

        Returns:
            True if published successfully
        """
        return self.publish(
            channel="autobench.result",
            event={
                "type": "autobench.result",
                "verdict": result.get("verdict", "unknown"),
                "quality": result.get("quality", 0.0),
                "cost": result.get("cost", 0.0),
                "time": result.get("time", 0.0),
                "utilization": result.get("utilization", {}),
                "iterations": result.get("iterations", 1),
                "benchmark": result.get("benchmark", ""),
            },
        )

    def publish_cycle(self, cycle_data: dict) -> bool:
        """
        Publish a metaprobe cycle completion event.

        Similar to deer-flow's deer-flow.metaprobe.cycle event.

        Args:
            cycle_data: Dict containing cycle metrics

        Returns:
            True if published successfully
        """
        return self.publish(
            channel="autobench.metaprobe.cycle",
            event={
                "type": "autobench.metaprobe.cycle",
                "iterations": cycle_data.get("iterations", 0),
                "quality_delta": cycle_data.get("quality_delta", 0.0),
                "total_cost": cycle_data.get("total_cost", 0.0),
                "total_time": cycle_data.get("total_time", 0.0),
                "verdict": cycle_data.get("verdict", "unknown"),
            },
        )


def _rfc3339_now() -> str:
    """Return current time in RFC3339 format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _generate_ulid() -> str:
    """Generate a ULID-like identifier."""
    import random
    import time
    timestamp = int(time.time() * 1000)
    random_part = random.randint(0, 2**80)
    return f"{timestamp:013d}{random_part:015d}"