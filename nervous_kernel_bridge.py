"""Back-compat shim for the old ``autobench.nervous_kernel_bridge`` import path.

Phase 1 of the autobench kernels restructuring relocated this module into
``autobench.kernels.bridge``. The old path remains importable and re-exports
the same public names so that terrain_kernel and external scripts keep
working.

New code should import from ``autobench.kernels.bridge`` directly.
"""

from autobench.kernels.bridge import (  # noqa: F401
    BridgeError,
    NervousKernelBridge,
    _ROLLING_HILLS_C,
    TRANSPILE,
)

# __main__ smoke test: delegate to the new location
if __name__ == "__main__":
    import runpy
    runpy.run_module("autobench.kernels.bridge", run_name="__main__")
