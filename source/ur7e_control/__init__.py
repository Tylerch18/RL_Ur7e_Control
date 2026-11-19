"""MuJoCo + Gymnasium package for UR7e control."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ur7e-control")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

# Register Gymnasium envs on import.
from ur7e_control import envs as _envs  # noqa: F401

__all__ = ["__version__"]
