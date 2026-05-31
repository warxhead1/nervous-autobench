"""Metrics collection and reporting for autobench."""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

Verdict = str
VerdictDistribution = dict[Verdict, int]


@dataclass
class AutobenchMetrics:
    """
    Metrics tracking for autobench harness executions.

    Attributes:
        total_runs: Total number of benchmark runs executed
        verdict_distribution: Count of each verdict type (CE/RE/TLE/MLE/WA/OK)
        avg_quality: Average quality score across all runs
        avg_cost: Average cost in dollars across all runs
        avg_time: Average execution time in seconds
        avg_tokens: Average tokens used per run
    """
    total_runs: int = 0
    verdict_distribution: VerdictDistribution = field(default_factory=dict)
    avg_quality: float = 0.0
    avg_cost: float = 0.0
    avg_time: float = 0.0
    avg_tokens: float = 0.0

    def record(self, result: dict) -> None:
        """
        Record a single benchmark result.

        Args:
            result: Dict containing verdict, quality, cost, time, tokens, etc.
        """
        self.total_runs += 1

        verdict = result.get("verdict", "unknown")
        self.verdict_distribution[verdict] = self.verdict_distribution.get(verdict, 0) + 1

        quality = result.get("quality", 0.0)
        cost = result.get("cost", 0.0)
        time = result.get("time", 0.0)
        tokens = result.get("tokens", 0)

        n = self.total_runs
        self.avg_quality = ((n - 1) * self.avg_quality + quality) / n
        self.avg_cost = ((n - 1) * self.avg_cost + cost) / n
        self.avg_time = ((n - 1) * self.avg_time + time) / n
        self.avg_tokens = ((n - 1) * self.avg_tokens + tokens) / n

    def merge(self, other: "AutobenchMetrics") -> None:
        """
        Merge another AutobenchMetrics into this one.

        Args:
            other: Another AutobenchMetrics to merge
        """
        if other.total_runs == 0:
            return

        total = self.total_runs + other.total_runs
        if total == 0:
            return

        for verdict, count in other.verdict_distribution.items():
            self.verdict_distribution[verdict] = (
                self.verdict_distribution.get(verdict, 0) + count
            )

        self.avg_quality = (
            (self.avg_quality * self.total_runs + other.avg_quality * other.total_runs)
            / total
        )
        self.avg_cost = (
            (self.avg_cost * self.total_runs + other.avg_cost * other.total_runs)
            / total
        )
        self.avg_time = (
            (self.avg_time * self.total_runs + other.avg_time * other.total_runs)
            / total
        )
        self.avg_tokens = (
            (self.avg_tokens * self.total_runs + other.avg_tokens * other.total_runs)
            / total
        )
        self.total_runs = total


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""
    quality: float
    cost: float
    speed: float
    index: int

    def dominates(self, other: "ParetoPoint") -> bool:
        """Check if this point dominates another (all attributes >= and one >)."""
        better = False
        if self.quality >= other.quality and self.cost >= other.cost and self.speed >= other.speed:
            if self.quality > other.quality or self.cost > other.cost or self.speed > other.speed:
                better = True
        return better


def compute_pareto_frontier(points: list[tuple[float, float, float]]) -> list[int]:
    """
    Compute the Pareto-optimal frontier from a list of (quality, cost, speed) tuples.

    Args:
        points: List of (quality, cost, speed) tuples.
                Higher quality and speed are better.
                Lower cost is better (so we invert cost for dominance check).

    Returns:
        List of indices into the input points that are on the Pareto frontier.
        Higher quality and speed = better, lower cost = better.
    """
    if not points:
        return []

    pareto_points = []
    n = len(points)

    for i in range(n):
        quality, cost, speed = points[i]
        is_pareto = True

        for j in range(n):
            if i == j:
                continue
            q2, c2, s2 = points[j]

            if q2 >= quality and c2 <= cost and s2 >= speed:
                if q2 > quality or c2 < cost or s2 > speed:
                    is_pareto = False
                    break

        if is_pareto:
            pareto_points.append(i)

    return pareto_points


def report(metrics: AutobenchMetrics, verbose: bool = False) -> str:
    """
    Generate a pretty-printed metrics report.

    Args:
        metrics: AutobenchMetrics to report
        verbose: Include detailed per-run breakdown

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("AUTOBENCH METRICS")
    lines.append("=" * 60)
    lines.append(f"Total Runs:     {metrics.total_runs}")
    lines.append(f"Avg Quality:    {metrics.avg_quality:.4f}")
    lines.append(f"Avg Cost:       ${metrics.avg_cost:.4f}")
    lines.append(f"Avg Time:       {metrics.avg_time:.2f}s")
    lines.append(f"Avg Tokens:     {metrics.avg_tokens:.0f}")
    lines.append("")
    lines.append("Verdict Distribution:")
    lines.append("-" * 40)

    verdicts_order = ["OK", "CE", "RE", "TLE", "MLE", "WA"]
    total_verdicts = sum(metrics.verdict_distribution.values())

    for verdict in verdicts_order:
        count = metrics.verdict_distribution.get(verdict, 0)
        if count > 0:
            pct = (count / total_verdicts * 100) if total_verdicts > 0 else 0
            lines.append(f"  {verdict:4s}: {count:5d} ({pct:5.1f}%)")

    for verdict, count in sorted(metrics.verdict_distribution.items()):
        if verdict not in verdicts_order:
            pct = (count / total_verdicts * 100) if total_verdicts > 0 else 0
            lines.append(f"  {verdict:4s}: {count:5d} ({pct:5.1f}%)")

    lines.append("-" * 40)
    lines.append("=" * 60)
    return "\n".join(lines)


def to_json(metrics: AutobenchMetrics) -> str:
    """
    Serialize metrics to JSON string.

    Args:
        metrics: AutobenchMetrics to serialize

    Returns:
        JSON string representation
    """
    return json.dumps(asdict(metrics), indent=2)


def from_json(json_str: str) -> AutobenchMetrics:
    """
    Deserialize metrics from JSON string.

    Args:
        json_str: JSON string

    Returns:
        AutobenchMetrics instance
    """
    data = json.loads(json_str)
    dist = data.pop("verdict_distribution", {})
    metrics = AutobenchMetrics(**data)
    metrics.verdict_distribution = dist
    return metrics


def to_file(metrics: AutobenchMetrics, path: str) -> None:
    """Write metrics to a JSON file."""
    with open(path, "w") as f:
        f.write(to_json(metrics))


def from_file(path: str) -> AutobenchMetrics:
    """Read metrics from a JSON file."""
    with open(path) as f:
        return from_json(f.read())