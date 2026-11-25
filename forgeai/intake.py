"""Intake interviewer. At most five clarifying questions, then a requirements document.

The question cap is enforced here, not only in the prompt. Interactive mode reads
stdin. Non-interactive mode writes the draft and records unanswered questions,
so an unattended review run does not block. The approve-or-edit pause is off
unless checkpoint=True; the assignment lists that pause as a stretch goal.
"""
import argparse
import csv
import json
import logging
import re
import sys
import warnings
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from forgeai.llm import LLMClient

MAX_QUESTIONS = 5
_REQUIRED = ("scope", "audience", "tone", "must_answer", "must_refuse")
_PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "intake"


class IntakeError(Exception):
    """The brief, the folder, or the model output cannot become requirements."""


def inspect_folder(docs_dir) -> list[dict]:
    """List names, types, sizes, and page or row counts. Top level only."""
    folder = Path(docs_dir)
    if not folder.is_dir():
        raise IntakeError(f"document folder not found: {folder}")
    entries = []
    for path in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
        if path.name.startswith("."):
            continue
        if path.is_dir():
            entries.append({
                "name": path.name, "type": "directory", "bytes": None,
                "pages": None, "rows": None, "error": None,
            })
            continue
        entry = {
            "name": path.name,
            "type": path.suffix.lower().lstrip(".") or "unknown",
            "bytes": path.stat().st_size,
            "pages": None,
            "rows": None,
            "error": None,
        }
        if path.suffix.lower() == ".pdf":
            entry["pages"], entry["error"] = _pdf_pages(path)
        elif path.suffix.lower() == ".csv":
            entry["rows"], entry["error"] = _csv_rows(path)
        entries.append(entry)
    return entries


def run_intake(brief, docs_dir, run_id, runs_dir="runs", llm=None, input_fn=None,
               output_fn=None, interactive=True, checkpoint=False) -> dict:
    """Interview, then write requirements.json and requirements.md."""
    if not isinstance(brief, str) or not brief.strip():
        raise IntakeError("brief must be a non-empty string")
    _check_run_id(run_id)
    if checkpoint and not interactive:
        raise IntakeError("checkpoint requires interactive intake")
    brief = brief.strip()
    inventory = inspect_folder(docs_dir)
    client = llm if llm is not None else LLMClient(run_id=run_id, runs_dir=runs_dir)
    reader = input_fn or input
    writer = output_fn or _write_out

    draft = _parse_json(_complete(client, _prompt(
        "draft.md", BRIEF=brief, INVENTORY=json.dumps(inventory, indent=2),
    )))
    proposed = draft.get("questions", [])
    questions = _cap_questions(proposed)
    truncated = len(proposed) > len(questions) if isinstance(proposed, list) else False
    if interactive:
        asked = _ask(questions, reader, writer)
    else:
        asked = [{"question": question, "answer": None} for question in questions]
    if interactive and asked:
        merged = _parse_json(_complete(client, _prompt(
            "merge.md",
            BRIEF=brief,
            INVENTORY=json.dumps(inventory, indent=2),
            DRAFT=json.dumps({key: draft.get(key) for key in _REQUIRED}, indent=2),
            ANSWERS=json.dumps(asked, indent=2),
        )))
        fields = _validate(merged)
    else:
        fields = _validate(draft)

    requirements = {
        "brief": brief,
        **fields,
        "questions": asked,
        "questions_truncated": truncated,
        "questions_dropped": max(0, len(proposed) - len(questions)) if isinstance(proposed, list) else 0,
        "document_inventory": inventory,
        "checkpoint": "skipped",
    }
    directory = Path(runs_dir) / run_id
    directory.mkdir(parents=True, exist_ok=True)
    if checkpoint:
        requirements["checkpoint"] = _checkpoint(requirements, reader, writer)
    _write_requirements(directory, requirements)
    return requirements


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Interview a brief and write a requirements document.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--brief", help="The one-paragraph brief.")
    source.add_argument("--brief-file", type=Path, help="File containing the brief.")
    parser.add_argument("--docs", required=True, type=Path, help="Document folder.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Do not read stdin. Record unanswered questions.")
    parser.add_argument("--checkpoint", action="store_true",
                        help="After writing, ask the requester to approve or edit. Stretch goal.")
    args = parser.parse_args(argv)
    brief = args.brief if args.brief is not None else args.brief_file.read_text(encoding="utf-8")
    try:
        result = run_intake(
            brief, args.docs, args.run_id, runs_dir=args.runs_dir,
            interactive=not args.non_interactive, checkpoint=args.checkpoint,
        )
    except IntakeError as exc:
        print(f"intake failed: {exc}", file=sys.stderr)
        return 1
    print(Path(args.runs_dir) / args.run_id / "requirements.md")
    print(f"questions asked: {sum(1 for item in result['questions'] if item['answer'] is not None)}")
    return 0


def _ask(questions, input_fn, output_fn) -> list[dict]:
    asked = []
    for question in questions:
        if len(asked) >= MAX_QUESTIONS:
            raise IntakeError("refusing to ask more than 5 questions")
        output_fn(f"Q{len(asked) + 1}: {question}\n")
        answer = input_fn()
        if answer is None:
            answer = ""
        asked.append({"question": question, "answer": str(answer).strip()})
    return asked


def _cap_questions(questions) -> list[str]:
    if questions is None:
        return []
    if not isinstance(questions, list):
        raise IntakeError("questions must be a list")
    cleaned = []
    for item in questions:
        if not isinstance(item, str) or not item.strip():
            raise IntakeError("each question must be a non-empty string")
        cleaned.append(item.strip())
    return cleaned[:MAX_QUESTIONS]


def _validate(data) -> dict:
    if not isinstance(data, dict):
        raise IntakeError("requirements must be a JSON object")
    for key in _REQUIRED:
        if key not in data:
            raise IntakeError(f"missing {key}")
    fields = {}
    for key in ("scope", "audience", "tone"):
        value = data[key]
        if not isinstance(value, str) or not value.strip():
            raise IntakeError(f"{key} must be a non-empty string")
        fields[key] = value.strip()
    for key in ("must_answer", "must_refuse"):
        value = data[key]
        if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
            raise IntakeError(f"{key} must be a list of non-empty strings")
        fields[key] = [item.strip() for item in value]
    return fields


def _checkpoint(requirements, input_fn, output_fn) -> str:
    output_fn(_markdown(requirements))
    output_fn(
        "Type approve to accept, or one line like scope=... / audience=... / tone=... "
        "/ must_answer=a; b / must_refuse=a; b\n"
    )
    line = input_fn()
    if line is None or line.strip().lower() == "approve":
        return "approved"
    if "=" not in line:
        raise IntakeError("checkpoint expects 'approve' or key=value")
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if key in ("scope", "audience", "tone"):
        if not value:
            raise IntakeError(f"{key} must be a non-empty string")
        requirements[key] = value
    elif key in ("must_answer", "must_refuse"):
        items = [part.strip() for part in value.split(";") if part.strip()]
        if not items:
            raise IntakeError(f"{key} must be a list of non-empty strings")
        requirements[key] = items
    else:
        raise IntakeError(f"cannot edit {key}")
    return "edited"


def _write_requirements(directory: Path, requirements: dict) -> None:
    (directory / "requirements.json").write_text(
        json.dumps(requirements, indent=2) + "\n", encoding="utf-8",
    )
    (directory / "requirements.md").write_text(_markdown(requirements), encoding="utf-8")


def _markdown(requirements: dict) -> str:
    def bullets(items):
        return "\n".join(f"- {item}" for item in items) or "- (none)"

    rows = ["| File | Type | Bytes | Pages | Rows |", "|---|---|---|---|---|"]
    for entry in requirements["document_inventory"]:
        name = entry["name"].replace("|", "\\|")
        rows.append(
            f"| {name} | {entry['type']} | {entry['bytes']} | {entry['pages']} | {entry['rows']} |"
        )
    questions = []
    for number, item in enumerate(requirements["questions"], start=1):
        answer = item["answer"] if item["answer"] is not None else "(not asked)"
        questions.append(f"{number}. {item['question']}\n   Answer: {answer}")
    body = [
        "# Requirements",
        "",
        "## Scope",
        requirements["scope"],
        "",
        "## Audience",
        requirements["audience"],
        "",
        "## Tone",
        requirements["tone"],
        "",
        "## Must answer",
        bullets(requirements["must_answer"]),
        "",
        "## Must refuse",
        bullets(requirements["must_refuse"]),
        "",
        "## Documents",
        "\n".join(rows),
        "",
        "## Clarifying questions",
        "\n".join(questions) or "(none)",
        "",
    ]
    return "\n".join(body)


def _complete(llm, user: str) -> str:
    return llm.chat(
        [
            {"role": "system", "content": "Return only the JSON object requested. No markdown."},
            {"role": "user", "content": user},
        ],
        temperature=0,
        seed=0,
    )


def _prompt(name: str, **values) -> str:
    text = (_PROMPTS / name).read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace(f"<<{key}>>", value)
    return text


def _parse_json(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise IntakeError("model did not return a JSON object")
    try:
        data = json.loads(stripped[start:end + 1])
    except json.JSONDecodeError as exc:
        raise IntakeError(f"model returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise IntakeError("model did not return a JSON object")
    return data


def _check_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise IntakeError("run_id must be a single path segment")


def _pdf_pages(path: Path):
    logger = logging.getLogger("pypdf")
    previous = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                reader = PdfReader(str(path))
                if reader.is_encrypted:
                    return None, "encrypted"
                return len(reader.pages), None
            except PdfReadError as exc:
                return None, str(exc)
    finally:
        logger.setLevel(previous)


def _csv_rows(path: Path):
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            if next(reader, None) is None:
                return 0, None
            return sum(1 for _ in reader), None
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return None, str(exc)


def _write_out(text: str) -> None:
    sys.stdout.write(text if text.endswith("\n") else text + "\n")


if __name__ == "__main__":
    sys.exit(main())
