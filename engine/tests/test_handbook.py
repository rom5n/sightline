"""The Handbook: every chapter exists in both languages with all sections, in-text links point to
real chapters and anchors, and every Python example runs (it is shown in the app as-is)."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src" / "handbook"
CHAPTERS = ["intro", "LinearRegression", "LogisticRegression", "Ridge", "Lasso", "ElasticNet", "SGDClassifier",
            "DecisionTreeClassifier", "RandomForestClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier",
            "HistGradientBoostingClassifier", "AdaBoostClassifier", "BaggingClassifier", "CatBoost", "SVC",
            "KNeighborsClassifier", "GaussianNB", "MLPClassifier", "KMeans", "DBSCAN", "PCA"]
SECTIONS = ["idea", "steps", "formulas", "where", "params", "code"]


def anchors(md: str) -> set[str]:
    return set(re.findall(r"^#{2,3} .+ \{#([\w-]+)\}\s*$", md, flags=re.M))


def test_every_chapter_in_both_languages() -> None:
    for lang in ("en", "ru"):
        have = {p.stem for p in (ROOT / lang).glob("*.md")}
        assert have == set(CHAPTERS), f"{lang}: missing {set(CHAPTERS) - have}, extra {have - set(CHAPTERS)}"
    assert {p.stem for p in (ROOT / "code").glob("*.py")} == set(CHAPTERS)


@pytest.mark.parametrize("chapter", CHAPTERS)
@pytest.mark.parametrize("lang", ["en", "ru"])
def test_chapter_structure(chapter: str, lang: str) -> None:
    md = (ROOT / lang / f"{chapter}.md").read_text(encoding="utf-8")
    missing = [s for s in SECTIONS + ([] if chapter == "intro" else ["try"]) if s not in anchors(md)]
    assert not missing, f"{lang}/{chapter}: sections without an anchor: {missing}"
    assert "{{code}}" in md
    if chapter != "intro":
        assert "{{try}}" in md
        assert "#hb/intro/" in md, f"{lang}/{chapter} should link to the intro chapter"
    assert md.count("$$") % 2 == 0, "unbalanced display math"
    assert re.search(r"^\|.+\|\s*$", md, flags=re.M), "a parameter table is expected"
    for target, anchor in re.findall(r"\]\(#hb/([\w-]+)/([\w-]*)\)", md):
        assert target in CHAPTERS, f"{lang}/{chapter}: link to unknown chapter {target}"
        if anchor:
            other = (ROOT / lang / f"{target}.md").read_text(encoding="utf-8")
            assert anchor in anchors(other), f"{lang}/{chapter}: #{anchor} not found in {target}"


@pytest.mark.parametrize("chapter", CHAPTERS)
def test_code_example_runs(chapter: str, tmp_path: Path) -> None:
    src = ROOT / "code" / f"{chapter}.py"
    text = src.read_text(encoding="utf-8")
    for lang in ("en", "ru"):  # every comment exists in both languages
        assert text.count(f"# @{lang} ") or "# @" not in text
    assert text.count("# @en") == text.count("# @ru"), f"{chapter}.py: EN and RU comment lines differ in number"
    r = subprocess.run([sys.executable, "-I", str(src)], cwd=tmp_path, capture_output=True, text=True, timeout=300,
                       env={"PYTHONIOENCODING": "utf-8", "OPENBLAS_NUM_THREADS": "2", "SYSTEMROOT": _sysroot()})
    assert r.returncode == 0, f"{chapter}.py failed:\n{r.stderr[-2000:]}"
    assert r.stdout.strip(), f"{chapter}.py printed nothing"


def _sysroot() -> str:
    import os

    return os.environ.get("SYSTEMROOT", "")
