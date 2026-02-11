"""Project configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path
    solution_name: str
    definition: str
    author: str
    language: str
    entry_point: str

    @property
    def kernel_path(self) -> Path:
        if self.language != "triton":
            raise ValueError("KernelForge currently supports Triton-first flow only.")
        return self.project_root / "solution" / "triton" / "kernel.py"


def load_project_config(project_root: Path) -> ProjectConfig:
    config_path = project_root / "config.toml"
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    solution = raw["solution"]
    build = raw["build"]
    return ProjectConfig(
        project_root=project_root,
        solution_name=solution["name"],
        definition=solution["definition"],
        author=solution["author"],
        language=build["language"],
        entry_point=build["entry_point"],
    )

