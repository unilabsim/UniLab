"""Environment registry and base classes."""

# Re-export from unilab.base for backward compatibility
from unilab.base import EnvCfg, NpEnv, registry, EpisodeLengthTracker, PenaltyCurriculum

__all__ = ["EnvCfg", "NpEnv", "registry", "EpisodeLengthTracker", "PenaltyCurriculum"]
