"""autobench.audit — claim verification, refactor checks, budgets, calibration, state.

Phase 2C of the autobench restructuring. Eleven sibling modules that
used to live at the autobench package root are regrouped here:
claims_audit, refactor_verifier, budget_guard, oracle_calibration,
post_run_assess, harness_diff, session_state, role_predicate, ahe,
invalidation, repo_analyzer. Public re-exports are listed in ``__all__``.
"""

from __future__ import annotations

# ahe.py — Agent Harness Evolution prediction contract
from .ahe import (
    LiveRefutationStatus,
    PlannedCompaction,
    Prediction,
    PredictionVerification,
    clip_prediction_to_feasible,
    compact_predictions,
    invalidate_prior_predictions,
    parse_prediction_from_llm_response,
    prediction_fingerprint,
    refute_live,
    should_emit_warning,
    verify_prediction,
)

# budget_guard.py — cost + rate budget guards
from .budget_guard import (
    BudgetExceeded,
    BudgetGuard,
    CompositeBudgetGuard,
    RateBudgetExceeded,
    RateBudgetGuard,
)

# claims_audit.py — EDD claims verification
from .claims_audit import (
    ClaimsAuditor,
    NBClaimResult,
    NBClaimSpec,
    NBEvidenceRecord,
    NBObservabilityEmulator,
    NBPassCriteria,
)

# harness_diff.py — before/after HarnessConfig diff
from .harness_diff import diff_harnesses

# invalidation.py — Bitloops-style scope invalidation
from .invalidation import (
    InvalidationEngine,
    InvalidationResult,
    InvalidationStore,
    ahe_scope_key,
    bead_scope_key,
    get_invalidation_engine,
    history_source_scope_key,
    promotion_scope_key,
    schema_scope_key,
)

# oracle_calibration.py — T-vector parameter calibration
from .oracle_calibration import (
    calibrate_noise,
    calibrate_sdf_topology,
    load_calibration,
    save_calibration,
)

# post_run_assess.py — post-run observer assessment
from .post_run_assess import assess_run

# refactor_verifier.py — tier-1 symbol-rename verifier
from .refactor_verifier import (
    RenameVerifier,
    RenameVerdict,
    Verdict,
)

# repo_analyzer.py — repo metrics + change classification
from .repo_analyzer import (
    ChangeType,
    RepoMetrics,
    analyze_repo,
    classify_change,
    detect_architecture_shift,
)

# role_predicate.py — deer-flow RoleSpec predicates
from .role_predicate import (
    ActivationPredicate,
    RoleSpecActivationBuilder,
    build_predicate_from_autobench_result,
    build_verdict_routing_predicates,
    evaluate_predicate,
)

# session_state.py — ULID session lifecycle
from .session_state import (
    SessionState,
    finish_session,
    generate_ulid,
    is_session_complete,
    is_valid_ulid,
    parse_rfc3339,
    rfc3339_now,
    start_session,
)

__all__ = [
    # ahe
    "LiveRefutationStatus",
    "PlannedCompaction",
    "Prediction",
    "PredictionVerification",
    "clip_prediction_to_feasible",
    "compact_predictions",
    "invalidate_prior_predictions",
    "parse_prediction_from_llm_response",
    "prediction_fingerprint",
    "refute_live",
    "should_emit_warning",
    "verify_prediction",
    # budget_guard
    "BudgetExceeded",
    "BudgetGuard",
    "CompositeBudgetGuard",
    "RateBudgetExceeded",
    "RateBudgetGuard",
    # claims_audit
    "ClaimsAuditor",
    "NBClaimResult",
    "NBClaimSpec",
    "NBEvidenceRecord",
    "NBObservabilityEmulator",
    "NBPassCriteria",
    # harness_diff
    "diff_harnesses",
    # invalidation
    "InvalidationEngine",
    "InvalidationResult",
    "InvalidationStore",
    "ahe_scope_key",
    "bead_scope_key",
    "get_invalidation_engine",
    "history_source_scope_key",
    "promotion_scope_key",
    "schema_scope_key",
    # oracle_calibration
    "calibrate_noise",
    "calibrate_sdf_topology",
    "load_calibration",
    "save_calibration",
    # post_run_assess
    "assess_run",
    # refactor_verifier
    "RenameVerifier",
    "RenameVerdict",
    "Verdict",
    # repo_analyzer
    "ChangeType",
    "RepoMetrics",
    "analyze_repo",
    "classify_change",
    "detect_architecture_shift",
    # role_predicate
    "ActivationPredicate",
    "RoleSpecActivationBuilder",
    "build_predicate_from_autobench_result",
    "build_verdict_routing_predicates",
    "evaluate_predicate",
    # session_state
    "SessionState",
    "finish_session",
    "generate_ulid",
    "is_session_complete",
    "is_valid_ulid",
    "parse_rfc3339",
    "rfc3339_now",
    "start_session",
]
