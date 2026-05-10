from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..config import ViewerConfig
from ..models import FlightSample


@dataclass
class AdapterContext:
    """Execution context for a reconstruction-method adapter."""

    method_id: str
    method_label: str
    source_path: Path
    config: ViewerConfig


@dataclass
class NormalizedTrack:
    """Method-neutral data passed from adapters into the viewer payload builder.

    Adapters may parse any method-specific JSON schema internally, but the rest
    of the viewer only sees this object.
    """

    track_json: dict[str, Any]
    raw_position_points: list[dict[str, Any]]
    samples: list[FlightSample]
    track_id: str = "track"
    source_schema: str = "unknown"
    interpolate_samples: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class MethodAdapter(Protocol):
    """Interface for reconstruction-method adapters."""

    adapter_id: str
    label: str

    def can_load(self, data: dict[str, Any], context: AdapterContext) -> bool:
        ...

    def normalize(self, data: dict[str, Any], context: AdapterContext) -> NormalizedTrack:
        ...
