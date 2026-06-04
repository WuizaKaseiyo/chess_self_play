---
name: task-to-instructions
description: Create an execution-ready Codex instruction document from a casual task description. Use when the user asks for an instruction prompt, a handoff prompt, or durable task instructions for another Codex instance.
---

# Task-to-Instructions Skill

## Purpose

Write `inst.md`: a concise Markdown instruction document another Codex instance can execute. The document should capture the objective, known constraints, relevant repo policies, unknowns, concrete tasks, and validation expectations.

The output is an instruction document, not a plan, analysis log, or transcript.

## Use When

- The user wants a task turned into an instruction document or Codex prompt.
- You need to hand work off to another Codex instance and want execution guidance rather than a plan.

## Conflict Handling

If the active prompt context, applicable `AGENTS.md` files, repo documentation, or user-provided instructions conflict, stop and explain:

- Which instructions conflict.
- Where each instruction came from.
- Why they cannot both be followed as written.
- What decision is needed.

Do not write `inst.md` or proceed with the handoff until the user resolves any blocking conflict.

## Output Rules

- Write the instruction document to `inst.md` in the current working directory unless a blocking ambiguity prevents a useful document.
- If a blocking ambiguity exists, ask one concise clarifying question and do not write `inst.md`.
- `inst.md` must contain only the instruction document.
- In chat, confirm that you wrote `inst.md`; do not paste the full document unless the user asks.
- Do not include analysis, planning notes, or meta commentary in `inst.md`.
- Do not include instructions about using this skill in `inst.md`.
- Do not include generator-facing guidance such as optional-section notes in `inst.md`.
- Replace all template placeholders with task-specific content before finishing.
- Replace generic checklist items with concrete checks for the task.
- Before finishing, verify that no unreplaced template placeholders remain in `inst.md`.

## Workflow

1. Read the full prompt context and the user's request carefully.
2. Extract the objective, constraints, success criteria, relevant repo policies, known facts, and open questions.
3. Separate confirmed facts from unknowns. Do not invent file paths, commands, configs, dataset names, or current behavior.
4. Use the template below. Omit a section only when it would add no value to the final instruction document.
5. Keep the instructions high-level but executable. Prefer discovery tasks over brittle guesses.
6. Classify uncertainty clearly.
   - `Blocking`: The executor cannot safely or usefully proceed without user input.
   - `Non-blocking`: The executor can investigate, proceed with a stated assumption, or report the uncertainty in the final response.
7. Include repo-specific guidance only when it is confirmed by the active prompt context, `AGENTS.md` files, repo documentation, scripts, configs, or command output.
   - In this repository, relevant examples may include reading `restricted_moves.md` and `iterative.md` when the task touches prompt templates, reward parsing, evaluators, or training launchers; using `conda activate verl` for simple local work; treating local runs as smoke tests only; using non-interactive SSH to `a5l.aip2.isambard`; submitting Slurm jobs with `sbatch --wait`; and using Git for sync.
   - If the task touches selection/restricted-moves prompts, reward parsing, evaluators, or training launchers, preserve the selection contract and terminology from `AGENTS.md` and `restricted_moves.md` instead of inventing a new framing.
   - Do not pad the instruction document with irrelevant policy.
8. Do not carry over environment names, benchmark names, cluster hosts, scheduler policies, sync workflows, model choices, or path assumptions from other repositories unless they are explicitly confirmed for this repository.
9. Keep the writing direct, concise, and easy to scan.

## Writing Guidance

- Prefer durable instructions such as "inspect the training launcher for the relevant flag" over guessed file paths or exact commands.
- Preserve concrete facts from the user. Do not dilute hard constraints into vague suggestions.
- Make discovery tasks targeted: what should be inspected, why it matters, and what evidence should be captured.
- Make implementation tasks outcome-oriented. The executor should know what change to make, not just that they should "work on it."
- Include validation that matches the task: tests, a focused repro, an eval, or another direct check.
- Surface assumptions and uncertainty plainly. Use explicit language such as "unknown from the prompt," "must be verified during execution," or "ask the user before proceeding."
- Ask for user input when needed, but keep the questions minimal and explain why the answer matters.
- Replace template placeholders with task-specific content. Omit sections that would otherwise be empty instead of leaving placeholder text behind.
- Replace generic checklist items with concrete checks that match the task.
- Adapt section names or task wording when the request is not a code-change task. Do not force every handoff into awkward terminology.

## Instruction Template

```md
**Context / Read First**
Read the full prompt context first. Then read the repo's `AGENTS.md` and any nested `AGENTS.md` files that apply to the files or workflows you touch. Read the most relevant project docs, scripts, configs, and notes before making changes.

**Background**
{Summarize only the known context and constraints from the user and prompt. If something is unknown, say that it is unknown instead of filling the gap.}

**Objective**
{State the end goal in concrete terms.}

**Known Constraints** (optional)
- {Hard requirements from the user, repo, environment, or tooling.}

**Open Questions / Unknowns**
- `Blocking`: {Write `None` or list only unknowns where the executor cannot safely or usefully proceed without user input.}
- `Non-blocking`: {Write `None` or list unknowns the executor should try to resolve through discovery. If discovery cannot resolve them, the executor may proceed with a clearly stated assumption and call it out in the final response.}

## Tasks
1. Discovery: {Inspect the relevant files, docs, configs, scripts, and current behavior. Capture evidence from code references or command output.}
2. Implementation: {Describe the changes or analysis to perform. Prefer the simplest approach that satisfies the objective.}
3. Validation: {Run the most relevant tests, smoke checks, or reproductions to verify the result.}

**Requirements**
- Read the relevant local context before making changes.
- Do not guess repo details that have not been confirmed from the prompt, docs, code, or command output.
- Ground claims in inspected code, documentation, or command output.
- If the task depends on missing user input, ask the user instead of guessing.
- Run commands in the foreground and inspect their outputs before moving on.
- Keep the approach simple and directly tied to the objective.

**Repo-Specific Requirements** (optional)
- {Include this section only when the task is for work in this repo and the items are relevant to the task.}
- Preserve the repo's actual policies and constraints when they are relevant to the task instead of restating generic defaults.
- Do not carry over environment names, benchmark names, cluster hosts, scheduler policies, sync workflows, model choices, or path assumptions from other repositories unless they are explicitly confirmed for this repository.
- If the task involves local execution, discover and use this repo's documented setup, environment, and validation commands. Keep exploratory runs lightweight unless the task or docs require broader validation.
- If the task involves remote or cluster execution, include only hosts, schedulers, and workflow policies confirmed by this repo or the active user request. If the task uses Slurm in this repository, use `sbatch --wait` unless the prompt says otherwise.
- If the task touches prompt templates, reward parsing, evaluators, or training launchers, read `restricted_moves.md` and `iterative.md` first, and preserve the restricted-moves selection contract and naming from `AGENTS.md`.
- If the task involves local execution in this repository, use `conda activate verl`; keep local runs lightweight; follow the GPU guidance in `AGENTS.md`.
- If the task involves cluster execution in this repository, use non-interactive SSH to `a5l.aip2.isambard`; for long `sbatch --wait` calls, set local `timeout_ms=86400000`.
- If the task involves syncing code to the cluster in this repository, use Git (`commit` / `push` locally, `git pull` remotely) unless the repo documents an exception.
- If the task is experimental or research-oriented, favor getting the workflow working correctly over production hardening.
- If the task touches datasets or downloaded artifacts, inspect a small sample instead of assuming schema or content.
- If the task uses subagents in this repository, reference and follow the subagent policy in `AGENTS.md`.
- Treat subagents as delegated work: use them for well-scoped delegated tasks, tell them to think extremely hard and deeply before doing anything, choose the right type per `AGENTS.md`, provide enough context, and do not redo delegated work except for limited preliminary work needed to understand the task and make sense of the subagent's output.
- Subagents must never run persistent monitoring or relaunch loops. This includes infinite `for`/`while` loops, recurring polling, background watch processes, any autonomous workflow that continues running after the subagent's main task should have ended, or any logic that automatically resubmits or relaunches jobs. Automatic job relaunch or resubmission is not allowed.
- If monitoring is needed, it must be done as a bounded, one-shot status check that exits immediately after reporting the current state. Monitoring must never continue in the background.
- Before moving on to the next step, fully close out all delegated subagent work: wait for all subagents to finish, collect each final result, review their outputs, ensure they are no longer running, and ensure no background monitoring, polling, or follow-up submissions remain active. If an agent was created for a single, self-contained task, verify that the task was completed correctly and then close the agent. Only keep the agent open if it contains especially valuable context that would be difficult to recover. Use `timeout_ms=3600000` when waiting on subagents.
- Do not use subagents to launch remote `sbatch` jobs in this repository, except for a very quick smoke run with an expected runtime of at most 30 minutes; any longer or non-smoke remote `sbatch` job must be launched by the main agent.
- Some sections can be omitted if they are genuinely unnecessary.

**Checklist**
- [ ] {Task-specific completion check with concrete evidence or outputs.}
- [ ] {Task-specific repo constraints, if relevant, were preserved.}
- [ ] {Task-specific unknowns were resolved, classified as `Blocking`, or handled as `Non-blocking` assumptions to verify.}
- [ ] {Task-specific validation, smoke checks, or completion checks were included.}
```

## Self-Check

- `inst.md` is an instruction document, not a plan or analysis.
- If no blocking ambiguity prevents a useful instruction document, `inst.md` exists and contains only the instruction document.
- If a blocking ambiguity prevents a useful instruction document, no `inst.md` was written and one concise clarifying question was asked.
- Unknowns are explicit; blocking ambiguities trigger a user question instead of a guess.
- Repo-specific policies are accurate, confirmed, and relevant to the task.
- The executor has enough context to act without re-reading the original conversation.
- The tasks cover discovery, implementation, and validation.
- No unreplaced template placeholders remain in `inst.md`.
- The checklist is specific to the task, not generic.
