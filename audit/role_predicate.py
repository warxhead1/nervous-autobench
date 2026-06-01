"""RoleSpec activation predicate builder for autobench deer-flow integration.

Builds deer-flow RoleSpec activation_predicate fields from autobench
HarnessResult verdict signals for conditional agent routing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from autobench.core import HarnessResult, Verdict


# Operators supported in ActivationPredicate
PredicateOperator = Literal["eq", "ne", "gt", "lt", "contains", "regex"]


@dataclass
class ActivationPredicate:
    """A predicate for conditional RoleSpec activation.

    Attributes:
        field: Dot-notation path to the context field to evaluate.
               e.g., "verdict", "quality", "metadata.error_type"
        operator: Comparison operator (eq, ne, gt, lt, contains, regex).
        value: The value to compare against.
    """

    field: str
    operator: PredicateOperator
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActivationPredicate:
        return cls(
            field=data["field"],
            operator=data["operator"],
            value=data["value"],
        )


def evaluate_predicate(predicate: ActivationPredicate, context: dict[str, Any]) -> bool:
    """Evaluate a predicate against a routing context.

    Args:
        predicate: The ActivationPredicate to evaluate.
        context: Dict containing routing context (verdict, quality, etc.).

    Returns:
        True if the predicate matches the context, False otherwise.

    Raises:
        ValueError: If the operator is unknown or the field is missing.
    """
    # Resolve field path (supports dot notation)
    parts = predicate.field.split(".")
    value: Any = context
    for part in parts:
        if not isinstance(value, dict):
            return False
        if part not in value:
            return False
        value = value[part]

    op = predicate.operator

    if op == "eq":
        return value == predicate.value
    elif op == "ne":
        return value != predicate.value
    elif op == "gt":
        return float(value) > float(predicate.value)
    elif op == "lt":
        return float(value) < float(predicate.value)
    elif op == "contains":
        if isinstance(value, str):
            return predicate.value in value
        if isinstance(value, (list, tuple)):
            return predicate.value in value
        return False
    elif op == "regex":
        try:
            return bool(re.search(str(predicate.value), str(value)))
        except re.error:
            return False
    else:
        raise ValueError(f"Unknown operator: {op}")


# Mapping from Verdict to routing target
_VERDICT_ROUTING = {
    Verdict.CE: "error-handler",
    Verdict.RE: "error-handler",
    Verdict.TLE: "timeout-handler",
    Verdict.MLE: "resource-handler",
    Verdict.WA: "debug-agent",
    Verdict.OK: "success-handler",
    # VF (Visual Fidelity, continuous) — close enough to OK that a refiner
    # might polish it rather than a debug agent rewriting from scratch.
    Verdict.VF: "refiner-agent",
    # Tier-1 refactor verifier verdicts (see autobench/research/refactor_verifiers_2026.md §10).
    # RV is success-shaped (refactor verified, no semantic change).
    Verdict.RV: "success-handler",
    # RD/RT are failure-shaped: drift recovery for unexpected AST changes,
    # test-fail handler for regressed test suites.
    Verdict.RD: "refactor-drift-recovery",
    Verdict.RT: "refactor-test-fail-handler",
}


def build_predicate_from_autobench_result(
    result: HarnessResult,
    default_target: Optional[str] = None,
) -> ActivationPredicate:
    """Build an ActivationPredicate that routes based on verdict.

    Creates a predicate that evaluates the verdict field and can be
    used by deer-flow's RoleSpec activation_predicate to route to
    appropriate handlers.

    Args:
        result: The HarnessResult to build a predicate from.
        default_target: Default routing target (e.g., "worker").

    Returns:
        An ActivationPredicate that matches the result's verdict.
    """
    verdict_value = result.verdict.value if isinstance(result.verdict, Verdict) else str(result.verdict)

    if default_target:
        # Build a contains predicate that can match multiple targets
        # This allows the RoleSpec to route based on verdict patterns
        return ActivationPredicate(
            field="verdict",
            operator="eq",
            value=verdict_value,
        )

    return ActivationPredicate(
        field="verdict",
        operator="eq",
        value=verdict_value,
    )


def build_verdict_routing_predicates() -> dict[Verdict, ActivationPredicate]:
    """Build a complete set of verdict-based routing predicates.

    Returns a dict mapping each Verdict to its corresponding predicate
    that can be used to route to error-handler, timeout-handler, etc.
    """
    return {
        verdict: ActivationPredicate(
            field="verdict",
            operator="eq",
            value=v.value if isinstance(v, Verdict) else str(v),
        )
        for verdict, v in [
            (Verdict.CE, Verdict.CE),
            (Verdict.RE, Verdict.RE),
            (Verdict.TLE, Verdict.TLE),
            (Verdict.MLE, Verdict.MLE),
            (Verdict.WA, Verdict.WA),
            (Verdict.OK, Verdict.OK),
            (Verdict.VF, Verdict.VF),
            # Tier-1 refactor verifier verdicts (refactor_verifiers_2026.md §10).
            (Verdict.RV, Verdict.RV),
            (Verdict.RD, Verdict.RD),
            (Verdict.RT, Verdict.RT),
        ]
    }


@dataclass
class RoleSpecActivationBuilder:
    """Builds deer-flow RoleSpec objects with activation_predicate fields.

    Example usage::

        builder = RoleSpecActivationBuilder()
        spec = builder.build(
            name="error-handler",
            predicate=ActivationPredicate(field="verdict", operator="eq", value="CE"),
            capability="error_recovery",
        )
    """

    predicates: dict[str, ActivationPredicate] = field(default_factory=dict)

    def add_predicate(self, name: str, predicate: ActivationPredicate) -> None:
        """Register a named predicate for use in role specs."""
        self.predicates[name] = predicate

    def build(
        self,
        name: str,
        predicate: ActivationPredicate,
        capability: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build a RoleSpec-compatible dict with activation_predicate.

        Args:
            name: Role name (e.g., "error-handler", "worker").
            predicate: The ActivationPredicate that determines when this role activates.
            capability: Capability string for this role.
            metadata: Optional additional metadata.

        Returns:
            A dict suitable for deer-flow's RoleSpec format.
        """
        spec: dict[str, Any] = {
            "name": name,
            "activation_predicate": predicate.to_dict(),
            "capability": capability,
        }
        if metadata:
            spec["metadata"] = metadata
        return spec

    def build_from_result(
        self,
        result: HarnessResult,
        name: str,
        capability: str = "",
    ) -> dict[str, Any]:
        """Build a RoleSpec from a HarnessResult verdict.

        Routes to appropriate handler based on verdict:

        - CE, RE → error-handler
        - TLE → timeout-handler
        - MLE → resource-handler
        - WA → debug-agent
        - OK → success-handler (or worker)

        Args:
            result: The HarnessResult to build from.
            name: Role name to use.
            capability: Capability string.

        Returns:
            A RoleSpec-compatible dict.
        """
        predicate = build_predicate_from_autobench_result(result)
        return self.build(name=name, predicate=predicate, capability=capability)

    def build_error_handler_role(self) -> dict[str, Any]:
        """Build a role spec for error-handler (CE/RE verdicts)."""
        return self.build(
            name="error-handler",
            predicate=ActivationPredicate(
                field="verdict",
                operator="eq",
                value=Verdict.CE.value,
            ),
            capability="error_recovery",
            metadata={"handles": ["CE", "RE"]},
        )

    def build_timeout_handler_role(self) -> dict[str, Any]:
        """Build a role spec for timeout-handler (TLE verdict)."""
        return self.build(
            name="timeout-handler",
            predicate=ActivationPredicate(
                field="verdict",
                operator="eq",
                value=Verdict.TLE.value,
            ),
            capability="timeout_recovery",
            metadata={"handles": ["TLE"]},
        )

    def build_debug_agent_role(self) -> dict[str, Any]:
        """Build a role spec for debug-agent (WA verdict)."""
        return self.build(
            name="debug-agent",
            predicate=ActivationPredicate(
                field="verdict",
                operator="eq",
                value=Verdict.WA.value,
            ),
            capability="debug_assist",
            metadata={"handles": ["WA"]},
        )

    def build_refactor_drift_recovery_role(self) -> dict[str, Any]:
        """Build a role spec for refactor-drift-recovery (RD verdict).

        Activates when a tier-1 refactor verifier flags AST drift beyond
        the declared refactor scope. See
        autobench/research/refactor_verifiers_2026.md §10.
        """
        return self.build(
            name="refactor-drift-recovery",
            predicate=ActivationPredicate(
                field="verdict",
                operator="eq",
                value=Verdict.RD.value,
            ),
            capability="refactor_drift_recovery",
            metadata={"handles": ["RD"]},
        )

    def build_worker_role(self) -> dict[str, Any]:
        """Build a default worker role that activates on any context."""
        return self.build(
            name="worker",
            predicate=ActivationPredicate(
                field="verdict",
                operator="eq",
                value=Verdict.OK.value,
            ),
            capability="code_generation",
            metadata={"fallback": True},
        )

    def build_all_handler_roles(self) -> list[dict[str, Any]]:
        """Build all standard handler role specs.

        Returns a list containing error-handler, timeout-handler,
        debug-agent, and worker roles.
        """
        return [
            self.build_error_handler_role(),
            self.build_timeout_handler_role(),
            self.build_debug_agent_role(),
            self.build_worker_role(),
        ]