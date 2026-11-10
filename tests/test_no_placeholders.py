from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_sources_have_no_scaffold_placeholders() -> None:
    forbidden = (
        "NotImplementedError",
        "UnsupportedOperationException",
        "Headless simulator is not implemented",
    )
    source_paths = list((ROOT / "src/graphwar_rl").rglob("*.py"))
    source_paths.append(
        ROOT / "java-sim/src/main/java/graphwarrl/HeadlessSimulator.java"
    )

    for source_path in source_paths:
        contents = source_path.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in contents, f"{marker} remains in {source_path}"
