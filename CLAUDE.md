# ForgeAI: project rules (read first, every session)

## What we are building
ForgeAI takes a one-paragraph brief plus a folder of documents and produces a tested agent: system prompt, retrieval config, tool selection, output contract, and an auto-generated eval set. It judges the agent, applies targeted fixes for at most 3 iterations, and reports honestly what still fails.

Source of truth: the assignment PDF at `/home/claude/OceanAI - Jr. GenAI Engineer Campus Recruitment Assignment.pdf`. If this file and the PDF disagree, the PDF wins. Tell me about the conflict; do not silently pick one.

## The nine mandatory requirements
1. **Intake interviewer.** Reads the brief, inspects the document folder, asks at most 5 clarifying questions (CLI is fine), writes a structured requirements document: scope, audience, tone, must-answer topics, must-refuse topics.
2. **Groundability guard.** If the brief asks for things the documents cannot support, flag it at the requirements stage. Never generate an agent that hallucinates the gap away.
3. **Agent generator.** From requirements plus documents, produce a complete declarative agent definition: system prompt, chunking and retrieval parameters, tool selection (only the tools the brief needs), output contract.
4. **Eval generator.** At least 20 cases, generated from the documents, not hand-written. Cover answerable questions (with source document plus key facts), out-of-scope questions that must be refused, and edge cases (ambiguous phrasing, answers spanning two documents). Commit the generation prompts.
5. **Judge.** Per case: grounded? correct against expected facts? refused when it should be? LLM-as-judge is allowed, but defend it against position bias, self-preference and verbosity bias. Deterministic checks first.
6. **Critic-optimizer loop.** Run evals, classify every failure (retrieval miss / prompt defect / hallucination / wrong refusal), apply a targeted fix (not a full regeneration), re-run. Max 3 iterations. Log every change: what, why, which failures it addressed, score before and after.
7. **Honest reporting.** Output: agent definition, per-iteration scorecard, and a report stating what still fails and why. Never claim 100%.
8. **Runnable product.** A minimal chat runner (CLI or simple UI) that loads any produced definition.
9. **Two full demonstrations.** End-to-end runs on two very different briefs and document sets. Commit the complete artifacts of both: requirements, definitions, eval sets, iteration logs, scorecards, plus the synthesis prompts used to make the documents.

Review day: an unseen brief and document folder, run unattended. Build for that, not for the two demos. Any domain term hard-coded in factory code is a defect.

## Hard constraints
- **Runtime model:** a small open model behind an OpenAI-compatible endpoint (Ollama locally). ForgeAI's runtime must never call a hosted frontier model, including as generator or judge. This is the assignment's premise.
- Must run offline, except the open-meteo weather tool.
- Python 3.11+. Stdlib first. Every new dependency needs a one-line justification in README.md. Ask before adding one.
- Retrieval: BM25 in stdlib. No embedding model unless justified in an ADR under `docs/adr/`.
- Tools come only from `tools/registry.py`. `attach()` must receive only what requirements need.
- Every LLM call logs model name, prompt hash, token counts and latency to `runs/<run-id>/calls.jsonl`.
- Generation and judging use temperature 0 with a fixed seed where the backend supports it.

## Honesty rules (these are graded)
- Deterministic checks decide wherever they can: a grounded answer must contain the expected key fact verbatim in a retrieved chunk; refusals are checked against the labelled out-of-scope set.
- The LLM judge runs only on the residue, with position-swapped pairs. Report its agreement with a hand-labelled sample of at least 20 cases.
- Eval holdout: generate the set once, freeze it, split it. The optimizer sees only the train split. Report scores on the holdout.
- Every failure gets exactly one of four categories. An unclassified failure is a bug.
- Eval cases whose expected fact is not a substring of the source document are discarded and logged.
- Never claim the pipeline works without the run artifacts to prove it.

## Working rules
- One stage per session. A stage is done when its tests pass, its artifacts are committed, and README.md status is updated.
- Minimal diffs. Change the smallest thing that fixes the problem. No full rewrites without asking.
- Commit messages end with:
  `Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01QgcsbVXBxMgH3Wsymx89zc`
- Never commit secrets, model weights, `.venv/`, or `__pycache__/`.
- Command-line instructions to the user go in one code block per task.

## Layout
```
forgeai/
  CLAUDE.md            this file
  README.md            status table, decisions, design risks
  IDE_PROMPTS.md       staged prompts used to build the project
  forgeai/             factory package
    llm.py             OpenAI-compatible client, call logging
    retrieval.py       chunking + BM25
    intake.py          stage 2
    guard.py           stage 3
    evalgen.py         stage 4
    generator.py       stage 5
    judge.py           stage 6
    loop.py            stage 7
    report.py          stage 8
  tools/registry.py    CSV lookup, calculator, current date, weather
  chat.py              stage 9 runner
  prompts/             every generation and judge prompt, committed
  demos/
    hr-policy/         documents, synthesis prompts, brief
    ship-spares/       CSV, supplier docs, synthesis prompts, brief
  runs/                run artifacts, committed for the two demos
  tests/
```

## Definition of done (whole project)
- All nine requirements run from one documented command each.
- Two demo runs committed with full artifacts.
- An unseen brief completes unattended with an honest report.
