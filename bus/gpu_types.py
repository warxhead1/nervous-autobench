"""GPU shader job and result types for autobench.

Classes:
    GPUJob — input job for the GPU shader broker
    GPUResult — result emitted after GPU shader execution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .idgen import iso_now as _iso_now
from .idgen import ulid as _ulid


@dataclass
class GPUJob:
    silo_id: str
    shader_artifact_path: str
    case_id: str
    frames: int = 60
    timeout_s: float = 120.0
    reference_image_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "silo_id": self.silo_id,
            "shader_artifact_path": self.shader_artifact_path,
            "case_id": self.case_id,
            "frames": self.frames,
            "timeout_s": self.timeout_s,
            "reference_image_path": self.reference_image_path,
        }


@dataclass
class GPUResult:
    case_id: str
    silo_id: str
    verdict: str  # OK|RE|TLE|WA|VF|CE
    fps_mean: float = 0.0
    frame_ms_p99: float = 0.0
    frames_rendered: int = 0
    frames_requested: int = 0
    frame_uri: str = ""
    latency_ms: float = 0.0
    silo_tester_report: dict = field(default_factory=dict)
    error: str = ""

    def to_event(self) -> dict[str, Any]:
        """CloudEvents-lite envelope for nervous-bus emission."""
        return {
            "id": _ulid(),
            "source": "/autobench/gpu_executor",
            "type": "autobench.gpu_result.v1",
            "datacontenttype": "application/json",
            "time": _iso_now(),
            "data": {k: v for k, v in vars(self).items()},
        }