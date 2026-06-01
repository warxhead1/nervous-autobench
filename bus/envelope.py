"""Unified CloudEvents-lite envelope builder for autobench bus events.

Phase 2A of the autobench restructuring. Prior to this phase, the autobench
bus surface had two divergent event-envelope shapes:

1. CloudEvents-lite (used by ``signal_bus.AutobenchResultPublisher`` and
   ``gpu_types.GPUResult.to_event``):
       {id, source, type, datacontenttype, time, data}

2. Nervous-bus-flavored CloudEvents (used by
   ``integration.NervousBusPublisher.publish``):
       {specversion, type, source, id, time, data}

The two diverge on two fields: CloudEvents-lite has ``datacontenttype``
(where CloudEvents says "application/json" by default), the nervous-bus
flavor has ``specversion`` (the CloudEvents core spec version). Both
encode the same logical envelope, but downstream parsers key off either
field depending on the consumer.

Phase 2A unifies on the CloudEvents-lite shape — it is the simpler of the
two, it is what most internal autobench code already produces, and it
matches the nervous shell SDK's reference envelope. ``NervousBusPublisher``
is updated to emit this shape; ``specversion`` is dropped; ``datacontenttype``
is added.

If you need to wrap a payload in a bus event, use ``build_event`` here.
"""

from __future__ import annotations

from typing import Any

from .idgen import iso_now, ulid


def build_event(source: str, type_: str, data: dict[str, Any]) -> dict[str, Any]:
    """Wrap a payload in a CloudEvents-lite envelope.

    Returns a dict with the canonical autobench event shape::

        {
            "id": <ULID>,
            "source": <source URI>,
            "type": <event type, e.g. "autobench.result.v1">,
            "datacontenttype": "application/json",
            "time": <RFC3339 timestamp>,
            "data": <payload dict>,
        }

    No `specversion` field — Phase 2A deliberately omits it; downstream
    consumers should key off `id`/`source`/`type`/`time`/`data`.
    """
    return {
        "id": ulid(),
        "source": source,
        "type": type_,
        "datacontenttype": "application/json",
        "time": iso_now(),
        "data": data,
    }
