# ForgeAI: staged prompts

There is no single prompt that builds this project well. A one-shot prompt produces a pipeline that looks finished and whose evals grade themselves. Run one stage per session. Paste the stage prompt into the IDE, in the repo folder, after CLAUDE.md is in place.

## Kickoff (run once)

> Read CLAUDE.md in full, then read the assignment PDF at `/home/claude/OceanAI - Jr. GenAI Engineer Campus Recruitment Assignment.pdf`. Do Stage 0 and Stage 1 only. Before writing any code, list every file you will create and your test plan, then stop and wait for my go. Do not implement later stages. When done, report what passes, what does not, and what needs my input (hardware, model choice, demo briefs).

## Stage 0: setup

> Create the repo skeleton from the Layout section of CLAUDE.md. Create a Python 3.11+ virtualenv. Write `forgeai/llm.py`: a client for an OpenAI-compatible `/v1/chat/completions` endpoint, base URL from an env var, with timeout, retry (max 2), and call logging to `runs/<run-id>/calls.jsonl`. Add a smoke test that skips cleanly if the endpoint is absent. Do not add dependencies beyond the stdlib without asking.

## Stage 1: retrieval

> Write `forgeai/retrieval.py`: chunking (paragraph-aware, configurable size and overlap) and BM25 over chunks, stdlib only. Accept .txt, .md and .csv (one chunk per row, with the header). Add PDF text extraction only if pypdf is approved; ask first. Tests: ranking on a fixture corpus, chunk boundaries, empty corpus, zero-match queries returning an explicit "no match" rather than a random chunk.

## Stage 2: intake (human checkpoint)

> Write `forgeai/intake.py`: reads a brief from a string, inspects a document folder (file names, types, sizes, page or row counts), asks at most 5 clarifying questions on stdin, and writes `runs/<run-id>/requirements.json` and `requirements.md`: scope, audience, tone, must-answer topics, must-refuse topics. Then show the requirements and ask the requester to approve or edit before the run continues. Tests use scripted stdin. Never ask more than 5 questions; enforce it in code, not by prompt alone.

## Stage 3: groundability guard

> Write `forgeai/guard.py`. Decompose each must-answer topic into a retrievable query, retrieve against the corpus, and flag any topic whose best BM25 score falls below a threshold. Calibrate the threshold on two hand-labelled lists in `tests/fixtures/`: topics the corpus supports and topics it does not. Report the calibration numbers. Output: `runs/<run-id>/groundability.json` listing each topic, its score, and a supported / unsupported flag. Stop the run if any must-answer topic is unsupported, unless the requester explicitly accepts the gap.

## Stage 4: eval generator

> Write `forgeai/evalgen.py` and its generation prompts under `prompts/evalgen/`. Generate at least 20 cases: answerable (with source document and key facts), out-of-scope (must refuse), edge cases (ambiguous phrasing, answers spanning two documents). Every expected key fact must be a verbatim substring of its source document; discard and log any case that fails this. Split into train and holdout, freeze the holdout to `runs/<run-id>/evals/holdout.json` with a hash in the log, and never let the optimizer read it. Tests: the substring filter, the split, the freeze.

## Stage 5: agent generator

> Write `forgeai/generator.py` and `prompts/generator/`. Input: requirements plus retrieved context. Output: `runs/<run-id>/agent.json` with system prompt, chunk size, overlap, top-k, tool list, and output contract. Tools must be a subset of `tools/registry.py` and must be justified by a requirement. Validate the JSON against a schema in code. An agent that attaches an unjustified tool fails validation.

## Stage 6: judge

> Write `forgeai/judge.py`. Deterministic checks first: grounded (key fact in retrieved chunk and in the answer), refusal correctness (against the labelled out-of-scope set). LLM-as-judge only for cases the deterministic checks cannot decide, with position-swapped pairs and a length-controlled prompt. Hand-label at least 20 cases myself and report judge agreement with those labels. Output a per-case scorecard.

## Stage 7: critic-optimizer loop

> Write `forgeai/loop.py`. Per iteration: run train evals, classify each failure into exactly one of retrieval miss / prompt defect / hallucination / wrong refusal, apply one targeted change, re-run, log before and after scores and the failure IDs addressed. Max 3 iterations. Any change that drops the train score is reverted. Holdout is scored once, at the end. Tests: classification on fixture failures, the revert rule, the iteration cap.

## Stage 8: report

> Write `forgeai/report.py`. Produce `runs/<run-id>/report.md`: the agent definition, a per-iteration scorecard, holdout score, every remaining failure with its category and a plain-language cause, and the judge's agreement rate. The report must state what does not work. No "100%" wording unless the holdout proves it, and even then, say how small the set is.

## Stage 9: chat runner

> Write `chat.py`: loads any `agent.json` and its run's index, answers in a CLI loop, shows which chunks it used. Test it on a produced agent. Streamlit or Gradio only after approval.

## Stage 10: demos

> Build `demos/hr-policy/` and `demos/ship-spares/`: brief, documents, and the synthesis prompts (commit them). Run the factory end to end on each. Commit every artifact under `runs/`. Do not hand-edit any generated artifact; if a result is wrong, fix the factory, then re-run.

## Stage 11: review rehearsal

> Write a third brief and a document folder of a kind neither demo covers. Run the factory on it unattended, with no edits between. Report the outcome, including any hard-coded assumption that broke. Fix factory defects only; do not tune the demos.
