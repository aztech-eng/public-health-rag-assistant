from __future__ import annotations

"""
Repository comparison utility.

Compares two repository trees, reports structural/content differences, and
generates semantic summaries intended for engineering review/portfolio use.
"""

import argparse
import datetime as dt
import filecmp
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence


REQUIRED_IGNORE_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".rag"}
DEFAULT_EXTRA_IGNORE_DIRS = {".rag_archive", "comparison"}
IGNORE_FILE_SUFFIXES = {".log"}
IGNORE_FILE_NAMES = {".env"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
CODE_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}


@dataclass(frozen=True)
class CompareConfig:
    repo_a: Path
    repo_b: Path
    output_dir: Path
    ignore_dirs: set[str]
    ignore_file_names: set[str]
    ignore_file_suffixes: set[str]


@dataclass(frozen=True)
class DiffResult:
    only_in_a: List[str]
    only_in_b: List[str]
    changed: List[str]
    same: List[str]


@dataclass(frozen=True)
class RepoMetrics:
    python_files: int
    total_lines: int


def _to_rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def should_ignore_file(path: Path, cfg: CompareConfig) -> bool:
    if path.name in cfg.ignore_file_names:
        return True
    if path.suffix.lower() in cfg.ignore_file_suffixes:
        return True
    # Explicitly requested wildcard ignore: *.env
    if path.name.endswith(".env"):
        return True
    return False


def collect_files(repo_root: Path, cfg: CompareConfig) -> Dict[str, Path]:
    file_map: Dict[str, Path] = {}
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in cfg.ignore_dirs]
        root_path = Path(root)
        for file_name in files:
            full_path = root_path / file_name
            if should_ignore_file(full_path, cfg):
                continue
            rel = _to_rel_posix(full_path, repo_root)
            file_map[rel] = full_path
    return file_map


def compare_file_maps(files_a: Dict[str, Path], files_b: Dict[str, Path]) -> DiffResult:
    paths_a = set(files_a)
    paths_b = set(files_b)

    only_a = sorted(paths_a - paths_b)
    only_b = sorted(paths_b - paths_a)
    common = sorted(paths_a & paths_b)

    changed: List[str] = []
    same: List[str] = []

    for rel in common:
        try:
            is_same = filecmp.cmp(files_a[rel], files_b[rel], shallow=False)
        except OSError:
            is_same = False
        if is_same:
            same.append(rel)
        else:
            changed.append(rel)

    return DiffResult(only_in_a=only_a, only_in_b=only_b, changed=changed, same=same)


def count_python_files_and_lines(file_map: Dict[str, Path]) -> RepoMetrics:
    py_paths = [path for rel, path in file_map.items() if rel.endswith(".py")]
    total_lines = 0
    for path in py_paths:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for _ in handle:
                    total_lines += 1
        except OSError:
            continue
    return RepoMetrics(python_files=len(py_paths), total_lines=total_lines)


def _filter_paths(paths: Iterable[str], predicate: Callable[[str], bool]) -> List[str]:
    return sorted([p for p in paths if predicate(p)])


def _is_code_like_path(rel_path: str) -> bool:
    p = Path(rel_path)
    if p.suffix.lower() not in CODE_SUFFIXES:
        return False
    # Exclude generated benchmark outputs from semantic/code-level analysis.
    if rel_path.startswith("evaluation/out/"):
        return False
    return True


def extract_new_modules_in_b(only_in_b: Iterable[str]) -> List[str]:
    return _filter_paths(only_in_b, lambda p: p.endswith(".py"))


def extract_modified_core_modules(changed: Iterable[str]) -> List[str]:
    return _filter_paths(changed, lambda p: p.startswith("rag_mvp/") and p.endswith(".py"))


def infer_feature_evidence(changed: Sequence[str], only_in_b: Sequence[str]) -> dict:
    semantic_scope = sorted(set([p for p in changed if _is_code_like_path(p)] + [p for p in only_in_b if _is_code_like_path(p)]))

    def pick(predicate: Callable[[str], bool]) -> List[str]:
        return sorted([p for p in semantic_scope if predicate(p.lower())])

    return {
        "evaluation": pick(lambda p: p.startswith("evaluation/") or "evaluate" in p or "benchmark" in p),
        "hybrid_retrieval": pick(
            lambda p: "hybrid" in p or "retrieval" in p or "vector" in p or "bm25" in p or "chroma" in p
        ),
        "api_changes": pick(lambda p: p.startswith("api/") or "/api/" in p),
        "llm_integration": pick(lambda p: "llm" in p or "openai" in p or "embedding" in p or "answering" in p),
    }


def build_new_capabilities(feature_evidence: dict) -> List[str]:
    capabilities: List[str] = []
    if feature_evidence["evaluation"]:
        capabilities.append("Domain-specific evaluation workflow with benchmark assets and scoring outputs.")
    if feature_evidence["hybrid_retrieval"]:
        capabilities.append("Hybrid retrieval path (BM25 + vector) with fusion/selection logic extensions.")
    if feature_evidence["api_changes"]:
        capabilities.append("Expanded API layer for ingestion/query workflows and production-style serving.")
    if feature_evidence["llm_integration"]:
        capabilities.append("Grounded LLM answer generation and embedding-driven retrieval integration.")
    if not capabilities:
        capabilities.append("No additional capabilities inferred from filename/code-diff signals.")
    return capabilities


def build_modified_components(modified_core_modules: Sequence[str], changed: Sequence[str], only_in_b: Sequence[str]) -> List[str]:
    scope = list(changed) + list(only_in_b)
    items: List[str] = []
    if modified_core_modules:
        items.append("Core `rag_mvp` modules were modified.")
    if any(p.startswith("scripts/") for p in scope):
        items.append("Scripts layer changed (evaluation/automation additions).")
    if any(p.startswith("api/") for p in scope):
        items.append("API components changed or were added.")
    if not items:
        items.append("No major modified component cluster inferred.")
    return items


def build_architectural_differences(changed: Sequence[str], only_in_b: Sequence[str]) -> List[str]:
    scope = list(changed) + list(only_in_b)
    diffs = ["Public Health RAG Assistant is structured as a production-style extension of the MVP."]

    if any(p.startswith("evaluation/") for p in scope):
        diffs.append("Adds a first-class evaluation subsystem (`evaluation/`, benchmark artifacts, score reports).")
    if any(p.startswith("api/") for p in scope):
        diffs.append("Adds/expands API-serving boundaries (`api/`) beyond CLI-only experimentation.")
    if any(p == "rag_mvp/retrieval.py" or "hybrid" in p.lower() for p in scope):
        diffs.append("Retrieval architecture evolves from baseline retrieval to hybrid/vector-aware retrieval flows.")
    if any("llm" in p.lower() or "answering" in p.lower() for p in scope):
        diffs.append("Answering architecture emphasizes grounded LLM behavior with safer abstention patterns.")
    return diffs


def _has_prefix(file_map: Dict[str, Path], prefix: str) -> bool:
    return any(rel.startswith(prefix) for rel in file_map)


def _has_path_fragment(file_map: Dict[str, Path], fragment: str) -> bool:
    frag = fragment.lower()
    return any(frag in rel.lower() for rel in file_map)


def _repo_has_keyword(file_map: Dict[str, Path], keyword: str) -> bool:
    key = keyword.lower()
    for rel, path in file_map.items():
        if key in rel.lower():
            return True
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if key in content.lower():
            return True
    return False


def build_feature_comparison(files_a: Dict[str, Path], files_b: Dict[str, Path]) -> List[dict]:
    def row(feature: str, a: bool, b: bool, notes: str) -> dict:
        return {"feature": feature, "repo_a": "Yes" if a else "No", "repo_b": "Yes" if b else "No", "notes": notes}

    return [
        row(
            "ingestion",
            _has_path_fragment(files_a, "ingest"),
            _has_path_fragment(files_b, "ingest"),
            "Both support ingestion, with domain-specific docs emphasized in Public Health RAG Assistant.",
        ),
        row(
            "retrieval",
            _has_path_fragment(files_a, "retrieval") or _has_path_fragment(files_a, "bm25"),
            _has_path_fragment(files_b, "retrieval") or _has_path_fragment(files_b, "bm25"),
            "Both implement retrieval; assistant extends retrieval behavior for practical evaluation loops.",
        ),
        row(
            "hybrid",
            _repo_has_keyword(files_a, "hybrid"),
            _repo_has_keyword(files_b, "hybrid"),
            "Hybrid retrieval behavior exists in both, with extended fusion/evaluation flow in assistant.",
        ),
        row(
            "evaluation",
            _has_prefix(files_a, "evaluation/") or _has_path_fragment(files_a, "evaluate"),
            _has_prefix(files_b, "evaluation/") or _has_path_fragment(files_b, "evaluate"),
            "Public Health RAG Assistant includes explicit benchmark/report artifacts.",
        ),
        row(
            "api",
            _has_prefix(files_a, "api/"),
            _has_prefix(files_b, "api/"),
            "API surface supports service-style usage and integration testing.",
        ),
        row(
            "llm_usage",
            _repo_has_keyword(files_a, "llm") or _repo_has_keyword(files_a, "openai"),
            _repo_has_keyword(files_b, "llm") or _repo_has_keyword(files_b, "openai"),
            "LLM grounding/integration is present and tuned for safer domain behavior in assistant.",
        ),
    ]


def render_markdown_table(rows: Sequence[dict]) -> str:
    header = "| Capability | RAG 36h MVP | Public Health RAG Assistant | Notes |"
    sep = "| --- | --- | --- | --- |"
    body = [f"| {r['feature']} | {r['repo_a']} | {r['repo_b']} | {r['notes']} |" for r in rows]
    return "\n".join([header, sep, *body])


def build_text_report(report: dict) -> str:
    summary = report["summary"]
    semantic = report["semantic"]
    rows = report["feature_comparison"]
    metrics = report["metrics"]

    lines: List[str] = [
        "Repository Comparison Report",
        "==========================",
        "",
        f"Generated: {report['generated_at']}",
        f"Repo A (MVP): {report['repo_a']}",
        f"Repo B (Assistant): {report['repo_b']}",
        "",
        "Summary",
        "-------",
        f"- Files only in repo A: {summary['only_in_a_count']}",
        f"- Files only in repo B: {summary['only_in_b_count']}",
        f"- Files changed in both: {summary['changed_count']}",
        "",
        "High-Level Differences",
        "----------------------",
        "- Repo A is an MVP-style baseline for RAG building blocks.",
        "- Repo B is a production-shaped, domain-focused extension with evaluation-driven iteration.",
        "",
        "New Capabilities",
        "----------------",
    ]

    lines.extend([f"- {item}" for item in semantic["new_capabilities"]])
    lines.extend(["", "Modified Components", "-------------------"])
    lines.extend([f"- {item}" for item in semantic["modified_components"]])
    if semantic["modified_core_modules"]:
        lines.append("- Modified core modules:")
        lines.extend([f"  - {mod}" for mod in semantic["modified_core_modules"]])

    lines.extend(["", "Architectural Differences", "-------------------------"])
    lines.extend([f"- {item}" for item in semantic["architectural_differences"]])

    lines.extend(["", "Feature Comparison", "------------------", render_markdown_table(rows), ""])
    lines.extend(
        [
            "Repository Metrics (Optional)",
            "-----------------------------",
            (
                f"- Repo A python files: {metrics['repo_a']['python_files']} | "
                f"total lines: {metrics['repo_a']['total_lines']}"
            ),
            (
                f"- Repo B python files: {metrics['repo_b']['python_files']} | "
                f"total lines: {metrics['repo_b']['total_lines']}"
            ),
            "",
            "Key Improvements In Public Health RAG Assistant",
            "------------------------------------------------",
            "- Stronger domain adaptation over real-world HPR data.",
            "- More explicit benchmarking/evaluation loop for iterative quality gains.",
            "- Expanded retrieval/LLM/API surface for production-style workflows.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    repo_b_default = script_path.parents[1]
    repo_a_default = repo_b_default.parent / "rag-36h"
    output_default = repo_b_default / "comparison"

    parser = argparse.ArgumentParser(description="Compare two repositories and generate structured reports.")
    parser.add_argument("--repo-a", type=Path, default=repo_a_default, help="Path to repo A (baseline MVP).")
    parser.add_argument("--repo-b", type=Path, default=repo_b_default, help="Path to repo B (assistant repo).")
    parser.add_argument("--output-dir", type=Path, default=output_default, help="Output directory for reports.")
    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="Include local/generated folders (comparison, .rag_archive) in diff results.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> CompareConfig:
    ignore_dirs = set(REQUIRED_IGNORE_DIRS)
    if not args.include_generated:
        ignore_dirs.update(DEFAULT_EXTRA_IGNORE_DIRS)
    return CompareConfig(
        repo_a=args.repo_a.resolve(),
        repo_b=args.repo_b.resolve(),
        output_dir=args.output_dir.resolve(),
        ignore_dirs=ignore_dirs,
        ignore_file_names=set(IGNORE_FILE_NAMES),
        ignore_file_suffixes=set(IGNORE_FILE_SUFFIXES),
    )


def ensure_repo(path: Path, label: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {path}")


def generate_report(cfg: CompareConfig) -> dict:
    ensure_repo(cfg.repo_a, "Repo A")
    ensure_repo(cfg.repo_b, "Repo B")

    files_a = collect_files(cfg.repo_a, cfg)
    files_b = collect_files(cfg.repo_b, cfg)
    diff = compare_file_maps(files_a, files_b)

    feature_evidence = infer_feature_evidence(changed=diff.changed, only_in_b=diff.only_in_b)
    modified_core = extract_modified_core_modules(diff.changed)
    semantic = {
        "new_modules_in_public_health_repo": extract_new_modules_in_b(diff.only_in_b),
        "modified_core_modules": modified_core,
        "inferred_features": feature_evidence,
        "new_capabilities": build_new_capabilities(feature_evidence),
        "modified_components": build_modified_components(modified_core, diff.changed, diff.only_in_b),
        "architectural_differences": build_architectural_differences(diff.changed, diff.only_in_b),
    }

    metrics = {
        "repo_a": asdict(count_python_files_and_lines(files_a)),
        "repo_b": asdict(count_python_files_and_lines(files_b)),
    }

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_a": str(cfg.repo_a),
        "repo_b": str(cfg.repo_b),
        "summary": {
            "only_in_a_count": len(diff.only_in_a),
            "only_in_b_count": len(diff.only_in_b),
            "changed_count": len(diff.changed),
            "same_count": len(diff.same),
        },
        "files": asdict(diff),
        "semantic": semantic,
        "feature_comparison": build_feature_comparison(files_a, files_b),
        "metrics": metrics,
        "ignore_config": {
            "ignore_dirs": sorted(cfg.ignore_dirs),
            "ignore_file_names": sorted(cfg.ignore_file_names),
            "ignore_file_suffixes": sorted(cfg.ignore_file_suffixes),
        },
    }


def write_reports(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    txt_path = output_dir / "report.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    txt_path.write_text(build_text_report(report), encoding="utf-8")
    return json_path, txt_path


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    report = generate_report(cfg)
    json_path, txt_path = write_reports(report, cfg.output_dir)
    summary = report["summary"]

    print("Comparison complete")
    print(f"Repo A: {cfg.repo_a}")
    print(f"Repo B: {cfg.repo_b}")
    print(f"Files only in repo A: {summary['only_in_a_count']}")
    print(f"Files only in repo B: {summary['only_in_b_count']}")
    print(f"Files changed in both: {summary['changed_count']}")
    print(f"Saved JSON report: {json_path}")
    print(f"Saved text report: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
