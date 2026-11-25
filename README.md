# ForgeAI: status, scope, and risks

Brief in, tested agent out. Assignment: OceanAI Jr. GenAI Engineer, campus round.

## Status (2026-11-25)

| Req | Item | State |
|---|---|---|
| 1 | Intake interviewer (<=5 questions, requirements doc) | Not started |
| 2 | Groundability guard | Not started. Design below. |
| 3 | Agent generator (prompt, retrieval, tools, output contract) | Not started |
| 4 | Eval generator (>=20 cases, generation prompts committed) | Not started |
| 5 | Judge (LLM + deterministic checks) | Not started |
| 6 | Critic-optimizer loop (max 3 iterations, change log) | Not started |
| 7 | Honest report and scorecard | Not started |
| 8 | Runnable chat runner | Not started |
| 9 | Two end-to-end demo runs, committed in full | Blocked on a local model (see below) |

Built and tested (stdlib only, 7/7 passing):
- `tools/registry.py`: four tools (csv_lookup, calculator, current_date, weather) and `attach()`, which returns only the tools a brief requests. Unknown tool names raise an error.
- `tests/test_registry.py`: tests include a code-injection check on the calculator.
Run with: `python3 -I tests/test_registry.py`

## Blockers in this sandbox

- huggingface.co, ollama.com, and api.open-meteo.com are unreachable. Model weights cannot be downloaded here.
- 2 CPU cores, 7.8 GB RAM, no GPU. Nothing useful runs locally here.
- PyPI works, so pure-Python code is buildable here. Model inference is not.
- The Anthropic endpoint is configured in this environment. Using Claude as the generator or judge would be a different project. It would violate the assignment's premise (small open model) and would be visible to reviewers. Not doing that.

Consequence: requirements 1-8 can be written here, but requirement 9 and any real eval numbers require running on your own machine.

## Architecture

- Model interface: OpenAI-compatible `/v1/chat/completions`. Ollama exposes this locally. Stages never import a model SDK directly.
- Recommended model: Qwen2.5-3B-Instruct or Llama-3.2-3B-Instruct (Q4). A 7B model at Q4 needs about 6 GB free RAM, which is tight on an 8 GB laptop. Confirm your RAM before choosing.
- Retrieval: chunk documents, TF-IDF or BM25 with no embedding model, so retrieval runs without another model download. Simpler, fewer failure points, and easier to defend.

## Design risks (stress-tested)

1. **Circular judging.** If the same small model writes the answers, writes the eval questions, and judges the answers, the scores measure self-consistency, not quality. Mitigation: deterministic checks first. A case is grounded only if its expected key fact appears verbatim in a retrieved chunk. Refusals are checked against a labelled out-of-scope set. The LLM judge runs only on the residue, with position-swapped pairs to counter position bias.
2. **Eval questions the model can answer from priors.** Generated questions can be answered without the documents. Mitigation: every generated case must carry an expected fact that is a substring of its source document. Cases failing that check are discarded and logged.
3. **The groundability guard is the hardest requirement.** "Can the documents support this brief?" cannot be answered by asking the model. Mitigation: decompose the brief into must-answer topics, retrieve for each, and flag topics with no chunk above a similarity threshold. The threshold must be calibrated on known-good and known-missing topics, not guessed.
4. **Critic-optimizer loop can overfit to its own evals.** Fixes tuned to a fixed eval set improve the score without improving the agent. Mitigation: split the eval set. Generate it once, freeze it, and report the score on the held-out half.
5. **"Unseen brief at review."** Any choice that works only on your two demos is a defect. Test on a third, deliberately different brief before the review, and treat any hard-coded domain term as a bug.
6. **Scope.** Nine mandatory requirements, two committed demos, and an unattended run is a multi-week build. One evening will not finish it. Cut stretch goals first. Do not cut the honest report, the change log, or the deterministic checks.

## Decisions needed from you

1. Your machine: RAM, GPU (if any), OS.
2. Which two demo briefs and document sets. Suggested: HR policy over 2-3 public HR PDFs, plus a ship-spares procurement assistant over a catalogue CSV and supplier PDFs.
3. Whether to ship a Streamlit UI (stretch) or keep to the CLI.
