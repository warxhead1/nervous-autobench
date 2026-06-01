"""Back-compat shim. Prefer ``autobench.daemons.trigger_daemon``."""

from autobench.daemons.trigger_daemon import (  # noqa: F401
    CHANNEL_COMMAND,
    CHANNEL_COMMAND_ACK,
    DEFAULT_ADVERSARIAL_RATIO,
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_MAX_ITER,
    DEFAULT_N_ADVOCATES,
    DEFAULT_TARGET_SKILL,
    CycleConfig,
    TriggerDaemon,
    _run_cycle_with_population_runner,
    build_cycle_config,
    is_autobench_paused,
)

# `python -m autobench.trigger_daemon` compatibility.
if __name__ == "__main__":
    import sys
    from autobench.daemons.trigger_daemon import _main
    sys.exit(_main())
