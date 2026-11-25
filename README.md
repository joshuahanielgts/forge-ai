# ForgeAI: status, scope, and risks

Brief in, tested agent out. Assignment: OceanAI Jr. GenAI Engineer, campus round.

## Status (2026-11-26)

| Req | Item | State |
|---|---|---|
| 1 | Intake interviewer (<=5 questions, requirements doc) | Done. Questions capped in code at 5. Non-interactive mode does not block. Approve-or-edit is opt-in (`--checkpoint`). 12/12 passing. |
| 2 | Groundability guard | Done. Deterministic BM25, threshold from the calibration file, non-zero exit on an unsupported topic unless `--accept-gap`. 8/8 passing. Overlap count on the fixture lists is 0. |
| 3 | Agent generator (prompt, retrieval, tools, output contract) | Not started |
| 4 | Eval generator (>=20 cases, generation prompts committed) | Not started |
| 5 | Judge (LLM + deterministic checks) | Not started |
| 6 | Critic-optimizer loop (max 3 iterations, change log) | Not started |
| 7 | Honest report and scorecard | Not started |
| 8 | Runnable chat runner | Not started. CLI when built. Streamlit only after all nine core requirements pass. |
| 9 | Two end-to-end demo runs, committed in full | Not started. Briefs approved. Model chosen. Documents not built yet. |

Built and tested (stdlib, plus `pypdf` for PDF text only):
- `tools/registry.py`: four tools (csv_lookup, calculator, current_date, weather) and `attach()`, which returns only the tools a brief requests. Unknown tool names raise an error. 7/7 passing, including a code-injection check on the calculator.
- `forgeai/llm.py`: OpenAI-compatible `/v1/chat/completions` client. Default model `qwen2.5:3b-instruct` (`FORGEAI_MODEL` overrides it). Timeout, two retries on connection failure and HTTP 5xx, no retry on 4xx, call log at `runs/<run-id>/calls.jsonl`. 4/4 passing against a local fake server. The live smoke skipped: nothing accepted a connection on `FORGEAI_BASE_URL` (default `http://127.0.0.1:11434/v1`).
- `forgeai/retrieval.py`: paragraph chunking and BM25 for `.txt`, `.md`, `.csv`, and `.pdf`. 10/10 passing.
- `forgeai/intake.py`: reads a brief, inventories the document folder (name, type, bytes, PDF pages, CSV rows), asks at most 5 clarifying questions, and writes `runs/<run-id>/requirements.json` and `requirements.md`. Prompts are in `prompts/intake/`. 12/12 passing with scripted stdin and a fake model. No live model call.
- `forgeai/guard.py`: scores each must-answer topic with BM25 and flags it unsupported when the best score is below `tests/fixtures/groundability_threshold.json`. No model call. On the fixture corpus, min supported score is 2.245001920848362, max unsupported score is 0, overlap count is 0, threshold is 1.122500960424181. 8/8 passing. No topic was left unseparated.

`pypdf`: the Python standard library cannot extract PDF text, and both approved demos are PDF-backed.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -I tests\test_registry.py
.\.venv\Scripts\python.exe -I tests\test_llm.py
.\.venv\Scripts\python.exe -I tests\test_retrieval.py
.\.venv\Scripts\python.exe -I tests\test_intake.py
.\.venv\Scripts\python.exe -I tests\test_guard.py
.\.venv\Scripts\python.exe -m forgeai.intake --brief "..." --docs path\to\docs --run-id run1
.\.venv\Scripts\python.exe -m forgeai.guard --requirements runs\run1\requirements.json --docs path\to\docs --run-id run1
```

## Blockers in this sandbox

These notes describe the sandbox where the scaffold was drafted. They are not a measurement of this Windows machine. What this checkout verified is the test modules above. The live model call did not run.

- huggingface.co, ollama.com, and api.open-meteo.com are unreachable. Model weights cannot be downloaded here.
- 2 CPU cores, 7.8 GB RAM, no GPU. Nothing useful runs locally here.
- PyPI works, so pure-Python code is buildable here. Model inference is not.
- The Anthropic endpoint is configured in this environment. Using Claude as the generator or judge would be a different project. It would violate the assignment's premise (small open model) and would be visible to reviewers. Not doing that.

Consequence: requirements 1-8 can be written here, but requirement 9 and any real eval numbers require running on your own machine.

## Architecture

- Model interface: OpenAI-compatible `/v1/chat/completions`. Ollama exposes this locally. Stages never import a model SDK directly.
- Runtime model: Qwen2.5-3B-Instruct, Ollama tag `qwen2.5:3b-instruct`. It fits in the RTX 3050's 6 GB of VRAM. A 7B model is an optional extra pass on the two demos only, not the factory default. Llama is not a candidate.
- Retrieval: chunk documents, BM25, no embedding model. PDF text is extracted with `pypdf`.

## Design risks (stress-tested)

1. **Circular judging.** If the same small model writes the answers, writes the eval questions, and judges the answers, the scores measure self-consistency, not quality. Mitigation: deterministic checks first. A case is grounded only if its expected key fact appears verbatim in a retrieved chunk. Refusals are checked against a labelled out-of-scope set. The LLM judge runs only on the residue, with position-swapped pairs to counter position bias.
2. **Eval questions the model can answer from priors.** Generated questions can be answered without the documents. Mitigation: every generated case must carry an expected fact that is a substring of its source document. Cases failing that check are discarded and logged.
3. **The groundability guard is the hardest requirement.** "Can the documents support this brief?" cannot be answered by asking the model. Mitigation: decompose each must-answer topic into a query, retrieve, and flag a topic whose best BM25 score is below the cutoff in `tests/fixtures/groundability_threshold.json`. That cutoff is the midpoint that separates the fixture lists. On this corpus the lists do not overlap.
4. **Critic-optimizer loop can overfit to its own evals.** Fixes tuned to a fixed eval set improve the score without improving the agent. Mitigation: split the eval set. Generate it once, freeze it, and report the score on the held-out half.
5. **"Unseen brief at review."** Any choice that works only on your two demos is a defect. Test on a third, deliberately different brief before the review, and treat any hard-coded domain term as a bug.
6. **Scope.** Nine mandatory requirements, two committed demos, and an unattended run is a multi-week build. One evening will not finish it. Cut stretch goals first. Do not cut the honest report, the change log, or the deterministic checks.

## Decisions

1. Machine: Windows, RTX 3050 with 6 GB VRAM. System RAM was not stated. The 3B model fits fully in VRAM.
2. Model: Qwen2.5-3B-Instruct (`qwen2.5:3b-instruct` in Ollama). A 7B model is an optional extra pass on the demos only.
3. Demo 1: HR policy assistant over 2–3 public HR policy PDFs. Demo 2: ship-spares procurement assistant over a catalogue CSV plus supplier PDFs. Synthesis prompts or source URLs are committed under `demos/` when those documents are built. Not built yet.
4. Chat runner: CLI. Streamlit only after all nine core requirements pass.
5. `pypdf` is approved. No other third-party package without asking.

Still open: the assignment lists "requester approves or edits the requirements before generation" as a stretch goal. Stage 2 in `IDE_PROMPTS.md` treats that pause as mandatory. This checkout follows the PDF. The pause is available with `--checkpoint` and is off by default, because review day runs unattended. Say if you want it on by default.
