"""Back-compat shim. Prefer ``autobench.audit.post_run_assess``."""

from autobench.audit.post_run_assess import (  # noqa: F401
    assess_run,
    main,
)

# `python -m autobench.post_run_assess` compatibility.
if __name__ == "__main__":
    import sys
    sys.exit(main())
