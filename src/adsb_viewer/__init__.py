"""ADS-B 3D flight viewer package."""

from .config import ViewerConfig
from .payload_builder import TrackPayloadBuilder


def create_dash_app(*args, **kwargs):
    """Import Dash app creation lazily so data utilities do not require Dash."""
    from .dash_app import create_dash_app as _create_dash_app

    return _create_dash_app(*args, **kwargs)


__all__ = ["ViewerConfig", "TrackPayloadBuilder", "create_dash_app"]
