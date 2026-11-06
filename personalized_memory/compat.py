"""Read renamed configuration while supporting existing installations."""

import os


def getenv(name: str, default: str | None = None) -> str | None:
    """Prefer the current variable, falling back to its original name."""
    value = os.getenv(name)
    if value is not None:
        return value
    return os.getenv("ASTRO_" + name.removeprefix("PMVA_"), default)
