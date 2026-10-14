"""Settings for the in-memory detection buffer."""
from dataclasses import dataclass
from typing import ClassVar, Dict

from config.loader import PROJECT_ROOT


@dataclass
class StorageConfig:
    PATH_FIELDS: ClassVar[Dict[str, str]] = {"output_dir": "root"}

    output_dir: str = str(PROJECT_ROOT / "output")
