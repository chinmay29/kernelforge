"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
(Triton or CUDA) into a Solution JSON file for submission.

This version constructs the JSON manually so it works on macOS without
triton/CUDA dependencies.  The JSON is validated by flashinfer_bench only
inside the Modal container.
"""

import json
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def load_config() -> dict:
    """Load configuration from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _collect_sources(source_dir: Path) -> list[dict]:
    """Recursively collect all source files under *source_dir*."""
    sources: list[dict] = []
    for p in sorted(source_dir.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            rel = p.relative_to(source_dir)
            sources.append({
                "path": str(rel),
                "content": p.read_text(encoding="utf-8"),
            })
    return sources


def pack_solution(output_path: Path | None = None) -> Path:
    """Pack solution files into a Solution JSON (no flashinfer_bench needed)."""
    config = load_config()

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]

    # Determine source directory based on language
    if language == "triton":
        source_dir = PROJECT_ROOT / "solution" / "triton"
    elif language == "cuda":
        source_dir = PROJECT_ROOT / "solution" / "cuda"
    else:
        raise ValueError(f"Unsupported language: {language}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    sources = _collect_sources(source_dir)
    if not sources:
        raise ValueError(f"No source files found in {source_dir}")

    # Build entry_point in the format flashinfer_bench expects:
    #   "<relative_file>::<function_name>"
    # If the user only specified the function name, prepend the first .py file.
    if "::" not in entry_point:
        py_files = [s["path"] for s in sources if s["path"].endswith(".py")]
        if py_files:
            entry_point = f"{py_files[0]}::{entry_point}"
        else:
            entry_point = f"{sources[0]['path']}::{entry_point}"

    solution_dict = {
        "name": solution_config["name"],
        "definition": solution_config["definition"],
        "author": solution_config["author"],
        "spec": {
            "language": language,
            "target_hardware": ["cuda"],
            "entry_point": entry_point,
            "dependencies": [],
        },
        "sources": sources,
    }

    # Write to output file
    if output_path is None:
        output_path = PROJECT_ROOT / "solution.json"

    output_path.write_text(json.dumps(solution_dict, indent=2), encoding="utf-8")
    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution_dict['name']}")
    print(f"  Definition: {solution_dict['definition']}")
    print(f"  Author: {solution_dict['author']}")
    print(f"  Language: {language}")
    print(f"  Entry: {entry_point}")

    return output_path


def main():
    """Entry point for pack_solution script."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack solution files into solution.json")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: ./solution.json)",
    )
    args = parser.parse_args()

    try:
        pack_solution(args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
