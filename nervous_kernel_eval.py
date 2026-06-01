"""Back-compat shim for the old ``autobench.nervous_kernel_eval`` import path.

Phase 1 of the autobench kernels restructuring relocated this module into
``autobench.kernels.eval``. The old path remains importable and re-exports
the same public names.

New code should import from ``autobench.kernels.eval`` directly.
"""

from autobench.kernels.eval import NervousKernelEvaluator  # noqa: F401

# __main__ smoke test: delegate to the new location
if __name__ == "__main__":
    import runpy
    runpy.run_module("autobench.kernels.eval", run_name="__main__")
