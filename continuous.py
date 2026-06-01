"""Back-compat shim. Prefer ``autobench.daemons.continuous``."""

from autobench.daemons import continuous as _mod

# Mirror the public surface of the source module so consumers using either
# path see identical behaviour. Importing the module (rather than picking
# names) lets ``monkeypatch.setattr("autobench.continuous.X", ...)`` keep
# working — the shim is a proxy.
from autobench.daemons.continuous import (  # noqa: F401
    BEAD_ID_ENV,
    CONFIDENT_THRESHOLD,
    ContinuousModeDaemon,
    DEFAULT_SESSIONS_PER_WINDOW,
    DEFAULT_WORKSPACE,
    Digest,
    PROMOTION_CONFIRM_ENV,
    PROMOTION_LEDGER_ENV,
    PROMOTION_REJECT_ENV,
    PromotionDecision,
    RECOMMENDED_RATE_MAX,
    RECOMMENDED_WINDOW_SECONDS,
    Surprise,
    SurpriseDigest,
    _iso_now,
    _serialise_harness,
    _resolve_promotion_ledger_path,
    main,
)

# Constants re-exported for direct import compatibility.
DEBUG_FILE = _mod.DEBUG_FILE

# `python -m autobench.continuous` compatibility: the source module's
# ``if __name__ == "__main__"`` guard was at the original location.
if __name__ == "__main__":
    import sys
    sys.exit(main())
