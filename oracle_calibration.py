"""Back-compat shim. Prefer ``autobench.audit.oracle_calibration``."""

from autobench.audit.oracle_calibration import (  # noqa: F401
    calibrate_noise,
    calibrate_sdf_topology,
    load_calibration,
    main,
    save_calibration,
)

# `python -m autobench.oracle_calibration` compatibility.
if __name__ == "__main__":
    main()
