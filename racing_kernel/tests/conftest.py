"""conftest.py — worktree import override for racing_kernel tests.

When running from the worktree under PYTHONPATH override, the editable install
finder routes autobench.racing_kernel to the main checkout (not the worktree).
This conftest patches autobench.racing_kernel.__path__ to point to the worktree
racing_kernel directory so new modules (loop_feedback) are found from the
worktree rather than the main checkout.

This file must NOT reach the main checkout (the feat/race-close-loop branch
is a throwaway worktree branch and is never merged per the bead contract).
"""
import importlib
import sys
from pathlib import Path

# Absolute path to this worktree's racing_kernel directory
_WORKTREE_RACING_KERNEL = str(Path(__file__).parent.parent.resolve())

# Patch autobench.racing_kernel.__path__ to include the worktree directory first.
# This ensures autobench.racing_kernel.loop_feedback is found in the worktree,
# while other submodules (oracle, instance, rollout_eval, etc.) still resolve
# from the editable install (main checkout) if they're not overridden.
try:
    import autobench.racing_kernel as _rk_pkg
    if _WORKTREE_RACING_KERNEL not in _rk_pkg.__path__:
        _rk_pkg.__path__.insert(0, _WORKTREE_RACING_KERNEL)
except ImportError:
    pass
