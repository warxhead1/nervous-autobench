"""Back-compat shim. Prefer ``autobench.audit.refactor_verifier``."""

from autobench.audit.refactor_verifier import (  # noqa: F401
    RenameVerifier,
    RenameVerdict,
    Verdict,
    _dump_normalized,
    _fallback_ast_drift,
    _python_identifier_index,
    _substitute_identifier,
    _which,
    run_ast_grep,
    run_difft,
    run_test_suite,
)

# Legacy alias — the original name was ``RefactorVerifier``; the source uses
# ``RenameVerifier`` for accuracy. Keep both working.
RefactorVerifier = RenameVerifier  # noqa: F405
