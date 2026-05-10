from __future__ import annotations

from typing import Any

from .adapters import AdapterContext, MethodAdapter, RawConverterJsonAdapter, ViewerPayloadAdapter


class NoMethodAdapterError(RuntimeError):
    pass


class MethodRegistry:
    """Registry that resolves reconstruction method JSON to an adapter.

    The registry is the only place where method/schema dispatch happens. The
    Cesium viewer and dashboard consume only the normalized payload produced
    after this step.
    """

    def __init__(self) -> None:
        self._adapters: list[MethodAdapter] = []

    def register(self, adapter: MethodAdapter) -> None:
        self._adapters.append(adapter)

    def resolve(self, data: dict[str, Any], context: AdapterContext) -> MethodAdapter:
        for adapter in self._adapters:
            if adapter.can_load(data, context):
                return adapter

        known = ", ".join(getattr(adapter, "adapter_id", adapter.__class__.__name__) for adapter in self._adapters)
        raise NoMethodAdapterError(
            f"No viewer adapter could load method '{context.method_id}' from {context.source_path}. "
            f"Add an adapter in adsb_viewer/adapters/ and register it in method_registry.py. "
            f"Registered adapters: {known or 'none'}."
        )


def default_method_registry() -> MethodRegistry:
    registry = MethodRegistry()

    # Already-normalized payload files first, so method-specific preprocessors can
    # bypass converter-schema parsing by emitting the browser viewer contract.
    registry.register(ViewerPayloadAdapter())

    # Current JsonTrackConverter output: raw_keyframes/render_keyframes.
    registry.register(RawConverterJsonAdapter())

    return registry
