"""greenhouse — goal-directed, budget-governed background scheduler for the
autobench FunSearch kernel loops, feeding Shader Garden.

Runs as a oneshot cycle (see ``greenhouse.cycle.run_cycle``), driven by a
goals manifest (``greenhouse.goals``) and a persistent request ledger
(``greenhouse.ledger``) that bounds spend against the shared plan budget
across process invocations. Validated GLSL candidates are exported
(``greenhouse.export``) to a drop directory Shader Garden can ingest, and
every cycle emits observability events on nervous-bus (``greenhouse.bus``).
"""
from __future__ import annotations
