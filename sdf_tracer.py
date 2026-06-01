"""Back-compat shim. Prefer `autobench.engines.sdf_tracer`.

Phase 2A of the autobench restructuring moved ``sdf_tracer.py`` into the
``autobench.engines`` subpackage. This module re-exports the public API
so legacy ``from autobench.sdf_tracer import …`` call sites keep working.
"""

from .engines.sdf_tracer import (  # noqa: F401
    render_sdf_cpp_to_png,
)
