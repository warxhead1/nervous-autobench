"""Back-compat shim. Prefer ``autobench.audit.role_predicate``."""

from autobench.audit.role_predicate import (  # noqa: F401
    ActivationPredicate,
    PredicateOperator,
    RoleSpecActivationBuilder,
    build_predicate_from_autobench_result,
    build_verdict_routing_predicates,
    evaluate_predicate,
)

__all__ = [
    "ActivationPredicate",
    "PredicateOperator",
    "RoleSpecActivationBuilder",
    "build_predicate_from_autobench_result",
    "build_verdict_routing_predicates",
    "evaluate_predicate",
]
