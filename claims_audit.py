"""Back-compat shim. Prefer ``autobench.audit.claims_audit``."""

from autobench.audit.claims_audit import (  # noqa: F401
    ClaimsAuditor,
    NBClaimResult,
    NBClaimSpec,
    NBEvidenceRecord,
    NBObservabilityEmulator,
    NBPassCriteria,
    main,
    run_evaluation,
    watch_mode,
)

# `python -m autobench.claims_audit` compatibility.
if __name__ == "__main__":
    main()
