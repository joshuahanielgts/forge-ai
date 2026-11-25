"""Deterministic groundability guard. No model calls.

Each must-answer topic becomes one BM25 query: the same alphanumeric split as
retrieval, with stopwords removed. A topic is unsupported when its best score
is below the threshold in tests/fixtures/groundability_threshold.json.
"""
import argparse
import json
import sys
from pathlib import Path

from forgeai.retrieval import BM25Index, load_documents, tokenize

STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with",
    "from", "by", "at", "as", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "this", "that", "these", "those", "about", "into",
    "over", "under", "than", "then", "so", "if", "but", "not", "no", "nor",
    "only", "own", "same", "too", "very", "can", "will", "just", "should",
    "must", "how", "what", "when", "where", "which", "who", "whom", "why",
    "do", "does", "did", "doing", "have", "has", "had", "having",
})
_LOADABLE = {".txt", ".md", ".csv", ".pdf"}
_DEFAULT_THRESHOLD = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "groundability_threshold.json"
)


class GuardError(Exception):
    """The requirements, the corpus, or the threshold file cannot be scored."""


def decompose(topic: str) -> str:
    """One retrieval query. Split into alphanumeric tokens and drop stopwords."""
    if not isinstance(topic, str):
        raise GuardError("topic must be a string")
    return " ".join(token for token in tokenize(topic) if token not in STOPWORDS)


def load_threshold(path=None) -> float:
    """Read the calibrated cutoff. The number does not live in this module."""
    file_path = Path(path) if path is not None else _DEFAULT_THRESHOLD
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        threshold = data["threshold"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise GuardError(f"could not read threshold: {file_path}: {exc}") from exc
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise GuardError(f"threshold must be a number: {file_path}")
    return float(threshold)


def read_topics(path) -> list[str]:
    topics = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            topics.append(stripped)
    return topics


def calibrate(docs_dir, supported_path, unsupported_path) -> dict:
    """Pick the cutoff that mislabels the fewest fixture topics.

    When a gap exists, the cutoff is the midpoint between the highest
    unsupported score and the lowest supported score. Overlap count is how
    many labelled topics sit inside the crossing band.
    """
    supported = read_topics(supported_path)
    unsupported = read_topics(unsupported_path)
    if len(supported) < 10 or len(unsupported) < 10:
        raise GuardError("calibration lists need at least 10 topics each")
    index = _index_for(docs_dir)
    supported_rows = [(topic, _best(index, decompose(topic))[0]) for topic in supported]
    unsupported_rows = [(topic, _best(index, decompose(topic))[0]) for topic in unsupported]
    supported_scores = [score for _, score in supported_rows]
    unsupported_scores = [score for _, score in unsupported_rows]
    min_supported = min(supported_scores)
    max_unsupported = max(unsupported_scores)
    if max_unsupported < min_supported:
        unseparated = []
    else:
        unseparated = (
            [topic for topic, score in supported_rows if score <= max_unsupported]
            + [topic for topic, score in unsupported_rows if score >= min_supported]
        )
    threshold = _best_threshold(supported_scores, unsupported_scores)
    return {
        "threshold": threshold,
        "min_supported_score": min_supported,
        "max_unsupported_score": max_unsupported,
        "overlap_count": len(unseparated),
        "unseparated": unseparated,
    }


def format_separation(stats: dict) -> str:
    lines = [
        f"min_supported_score={stats['min_supported_score']}",
        f"max_unsupported_score={stats['max_unsupported_score']}",
        f"overlap_count={stats['overlap_count']}",
        f"threshold={stats['threshold']}",
    ]
    if stats.get("unseparated"):
        lines.append("unseparated=" + " | ".join(stats["unseparated"]))
    return "\n".join(lines)


def evaluate(requirements, docs_dir, threshold, accept_gaps=()) -> dict:
    """Score every must-answer topic. Acceptance is recorded, not relabelled."""
    if not isinstance(requirements, dict) or not isinstance(requirements.get("must_answer"), list):
        raise GuardError("requirements must contain a must_answer list")
    accepted = {gap.strip() for gap in accept_gaps}
    index = _index_for(docs_dir)
    topics = []
    for topic in requirements["must_answer"]:
        if not isinstance(topic, str):
            raise GuardError("each must-answer topic must be a string")
        query = decompose(topic)
        score, chunk_id = _best(index, query)
        status = "unsupported" if score < threshold else "supported"
        topics.append({
            "topic": topic,
            "query": query,
            "score": score,
            "chunk_id": chunk_id,
            "status": status,
            "gap_accepted": status == "unsupported" and topic.strip() in accepted,
        })
    blocking = [row["topic"] for row in topics if row["status"] == "unsupported" and not row["gap_accepted"]]
    return {
        "threshold": threshold,
        "accepted_gaps": [gap.strip() for gap in accept_gaps],
        "topics": topics,
        "stopped": bool(blocking),
        "blocking_topics": blocking,
    }


def run_guard(requirements_path, docs_dir, run_id, runs_dir="runs", threshold_path=None,
              accept_gaps=()) -> int:
    _check_run_id(run_id)
    requirements = _read_json(requirements_path)
    report = evaluate(requirements, docs_dir, load_threshold(threshold_path), accept_gaps)
    directory = Path(runs_dir) / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "groundability.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    return 1 if report["stopped"] else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Flag must-answer topics the corpus does not support.")
    parser.add_argument("--requirements", type=Path, help="requirements.json from intake.")
    parser.add_argument("--docs", type=Path, help="Document folder.")
    parser.add_argument("--run-id")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--threshold", type=Path, default=None,
                        help="JSON file with a threshold field. Defaults to the calibrated fixture.")
    parser.add_argument("--accept-gap", action="append", default=[],
                        help="Allow this unsupported topic to pass. Repeat for more than one.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Print separation numbers for the fixture lists and exit.")
    args = parser.parse_args(argv)
    try:
        if args.calibrate:
            root = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
            stats = calibrate(
                root / "corpus",
                root / "supported_topics.txt",
                root / "unsupported_topics.txt",
            )
            print(format_separation(stats))
            return 0
        if not args.requirements or not args.docs or not args.run_id:
            parser.error("--requirements, --docs, and --run-id are required")
        return run_guard(
            args.requirements, args.docs, args.run_id, runs_dir=args.runs_dir,
            threshold_path=args.threshold, accept_gaps=args.accept_gap,
        )
    except GuardError as exc:
        print(f"guard failed: {exc}", file=sys.stderr)
        return 2


def _index_for(docs_dir):
    folder = Path(docs_dir)
    if not folder.is_dir():
        raise GuardError(f"document folder not found: {folder}")
    paths = [
        path for path in sorted(folder.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in _LOADABLE
    ]
    chunks = load_documents(paths) if paths else []
    return BM25Index(chunks)


def _best(index: BM25Index, query: str):
    if not query:
        return 0.0, None
    result = index.search(query, top_k=1)
    if result.status != "ok" or not result.hits:
        return 0.0, None
    hit = result.hits[0]
    return float(hit.score), f"{Path(hit.source).name}#{hit.index}"


def _best_threshold(supported_scores, unsupported_scores) -> float:
    """Minimize mislabels. Break ties by the larger gap around the cutoff."""
    observed = sorted(set(supported_scores) | set(unsupported_scores))
    candidates = []
    for left, right in zip(observed, observed[1:]):
        candidates.append((left + right) / 2.0)
    if not candidates:
        candidates.append(observed[0] if observed else 0.0)
    best_threshold = candidates[0]
    best_errors = None
    best_margin = None
    for threshold in candidates:
        errors = (
            sum(score < threshold for score in supported_scores)
            + sum(score >= threshold for score in unsupported_scores)
        )
        margin = min(abs(score - threshold) for score in observed)
        if (
            best_errors is None
            or errors < best_errors
            or (errors == best_errors and margin > best_margin)
        ):
            best_threshold = threshold
            best_errors = errors
            best_margin = margin
    return best_threshold


def _read_json(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GuardError(f"could not read requirements: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise GuardError("requirements must be a JSON object")
    return data


def _check_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise GuardError("run_id must be a single path segment")


if __name__ == "__main__":
    sys.exit(main())
