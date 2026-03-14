"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
(Triton or CUDA) into a Solution JSON file for submission.

This version constructs the JSON manually so it works on macOS without
triton/CUDA dependencies. The JSON is validated by flashinfer_bench only
inside the Modal container.
"""

import ast
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
    binary_suffixes = {".pyc", ".pyo", ".so", ".dylib", ".o", ".a"}
    for p in sorted(source_dir.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        if "__pycache__" in p.parts:
            continue
        if p.suffix.lower() in binary_suffixes:
            continue
        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Skip non-UTF8 artifacts that can appear during local dev (e.g. .pyc).
            continue
        rel = p.relative_to(source_dir)
        sources.append({
            "path": str(rel),
            "content": content,
        })
    return sources


def _normalize_list(value: object, *, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    return list(value)


def _infer_destination_passing_style(source: str, fn_name: str) -> bool | None:
    """Infer destination-passing style from the kernel return behavior.

    Returns:
        True if the kernel looks destination-passing style (no value-return),
        False if it appears value-returning, or None if inference fails.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    target_fn: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            target_fn = node
            break
    if target_fn is None:
        return None

    class _ReturnValueVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.returns_value = False

        def visit_Return(self, node: ast.Return) -> None:
            if node.value is not None:
                self.returns_value = True

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is target_fn:
                self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node is target_fn:
                self.generic_visit(node)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

    visitor = _ReturnValueVisitor()
    visitor.visit(target_fn)
    return not visitor.returns_value


def pack_solution(output_path: Path | None = None) -> Path:
    """Pack solution files into a Solution JSON (no flashinfer_bench needed)."""
    config = load_config()

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]
    target_hardware = _normalize_list(
        build_config.get("target_hardware"),
        default=["cuda"],
    )
    dependencies = _normalize_list(
        build_config.get("dependencies"),
        default=[],
    )
    binding = build_config.get("binding")

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

    entry_rel_path, _, entry_fn = entry_point.partition("::")
    source_map = {row["path"]: row["content"] for row in sources}

    dps_override = build_config.get("destination_passing_style")
    if dps_override is None and entry_rel_path in source_map and entry_fn:
        destination_passing_style = _infer_destination_passing_style(
            source_map[entry_rel_path],
            entry_fn,
        )
    elif dps_override is None:
        destination_passing_style = None
    else:
        destination_passing_style = bool(dps_override)

    spec = {
        "language": language,
        "target_hardware": target_hardware,
        "entry_point": entry_point,
        "dependencies": dependencies,
    }
    if binding:
        spec["binding"] = binding
    if destination_passing_style is not None:
        spec["destination_passing_style"] = destination_passing_style

    solution_dict = {
        "name": solution_config["name"],
        "definition": solution_config["definition"],
        "author": solution_config["author"],
        "spec": spec,
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
    if destination_passing_style is not None:
        print(f"  Destination Passing Style: {destination_passing_style}")

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
