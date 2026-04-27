---
name: daily-research-journal
description: Create or update a concise daily research journal entry for this research repo. Use when the user asks to write, update, summarize, or generate today's journal from outputs/experiment folders. Only consider experiment folders modified today unless the user explicitly asks otherwise.
---

You are working inside my research repository.

Your task is to update today’s research journal entry in the `journal/` folder.

Context:
- I am a PhD student doing computer vision research.
- The architecture we are working on is called LatentFormer and is described in `LATENT_ARCH.md`
- This repository uses Detectron2.
- Experiment results are stored in the `outputs/` folder.
- A typical experiment folder looks like:

  outputs/h200/latentformer_R50_no_seed_hungarian_pattern_loss_hungarian/

  and may contain:
  - checkpoints
  - log.txt
  - config.yaml
  - logs/
    - branch / commit information
    - uncommitted changes
    - reproducibility metadata
    - other run information

Important constraint:
Only consider experiment folders inside `outputs/` that were modified today.

Do not summarize old experiments unless they are needed as comparison baselines and are explicitly referenced by today’s runs.

Goal:
Create a concise, useful daily journal entry that summarizes what happened today without cluttering the journal with too much detail. I may run 5–10 experiments per day, many of them quick tests that are aborted early. The journal should stay readable.

The journal is not supposed to duplicate all reproducibility details. Those already live inside each experiment folder. The journal should instead explain:
- what I was trying to understand today
- which experiments mattered
- which experiments were quick/aborted tests
- what conclusions or partial conclusions can be drawn
- what I should do next

Process:

1. Determine today’s date.
   - Use the machine’s local date unless the repository has another clear convention.
   - Use ISO format: YYYY-MM-DD.

2. Find experiment folders modified today under `outputs/`.

   Treat an experiment folder as the nearest folder under `outputs/<group>/<experiment_name>/` that contains at least one of:
   - `config.yaml`
   - `log.txt`
   - `logs/`
   - checkpoint files
   - Detectron2 output artifacts

   Example valid experiment folder:
   - `outputs/h200/latentformer_R50_no_seed_hungarian_pattern_loss_hungarian`

   Do not treat nested folders like `logs/`, `inference/`, `events/`, or checkpoint subfolders as separate experiments.

3. For each experiment folder modified today, inspect available files such as:
   - `config.yaml`
   - `log.txt`
   - `metrics.json`
   - files inside `logs/`
   - checkpoint names
   - evaluation outputs, if present
   - tensorboard/event/log summaries, if easily readable
   - git metadata stored in `logs/`

   When `metrics.json` is present, explicitly check for seed-selection summary metrics such as:
   - `GTOracleSeedSelection/panoptic_seg/PQ`
   - `ClusteringSeedSelection/panoptic_seg/PQ`
   - `GoldenSeedSelection/panoptic_seg/PQ`

   Use them when they are available and relevant, especially for runs comparing seed-selection strategies.

4. Infer the purpose of each run from:
   - the experiment folder name
   - the config
   - the logs
   - branch/commit information
   - any existing intent/notes file, if present

5. Classify every modified experiment into one of these levels:

   - `transient`
     Debug run, smoke test, broken config, very short run, or run with no research value.
     These should usually be mentioned only as a grouped sentence, not individually.

   - `rapid`
     A quick experiment that tested a real idea but was aborted, failed, or was not promising.
     These should be summarized in a compact table.

   - `notable`
     A run that completed, produced meaningful results, changed my interpretation, revealed an informative failure mode, or deserves follow-up.
     These should get a short subsection.

   - `milestone`
     A run that is potentially thesis/paper-relevant, becomes a baseline, sets a new best result, or is important enough to return to later.
     These should get a slightly more detailed subsection.

6. Keep the journal concise.
   The final entry should be readable in under five minutes.

7. Do not create long experiment cards for every run.
   Only `notable` and `milestone` experiments should receive paragraph-level discussion.
   `rapid` experiments should go in a table.
   `transient` experiments should be grouped together or omitted unless they explain something important.

8. Create the journal file at:

   journal/YYYY/MM/YYYY-MM-DD.md

   If the repository already uses a different journal structure, follow the existing convention.

9. If today’s journal file already exists:
   - update it carefully
   - preserve any manually written notes
   - add new information without deleting useful existing content
   - avoid duplicating sections

10. Use this structure for the journal entry:

   # Journal — YYYY-MM-DD

   ## Daily summary

   Write one concise paragraph summarizing the day.
   Focus on the main research direction and the most important outcome.

   ## Research focus

   State the main question or theme investigated today.
   Examples:
   - testing no-seed variants
   - debugging Hungarian assignment
   - comparing LatentFormer R50 variants
   - checking training stability
   - validating a new loss/configuration

   ## Important repository changes

   Summarize only high-level code or config changes from today that matter scientifically.
   Do not list every changed file unless necessary.
   Mention branch and relevant commits if available from the run logs.

   ## Experiments modified today

   Start with a compact overview table of all non-transient modified experiment folders.

   Columns:
   - Level
   - Run
   - Purpose
   - Status
   - Outcome
   - Decision

   The `Run` column should include the relative path to the output folder.

   ## Notable experiments

   Include this section only if there are notable or milestone runs.

   For each notable or milestone run, include:

   ### `run_name`

   Path:
   `outputs/.../run_name`

   Level:
   `notable` or `milestone`

   Purpose:
   Briefly explain what the experiment tested.

   Setup:
   Mention only the most important factors, such as architecture, backbone, seed/no-seed, loss, matching strategy, dataset, or config family.

   Result:
   Summarize the observed result. Include metrics only if they are clearly available.
   If the run was aborted or incomplete, say so.

   Interpretation:
   Explain what this suggests scientifically.
   Be explicit about uncertainty.

   Decision:
   Use one of:
   - continue
   - compare
   - rerun
   - abandon
   - keep as baseline
   - needs qualitative inspection
   - needs evaluation
   - inconclusive

   Next action:
   One concrete next step.

   ## Rapid or aborted runs

   Include this section for quick tests that do not deserve detailed discussion.

   Use a compact table:

   | Run | Purpose | Outcome | Decision |
   |---|---|---|---|

   Keep each row short.

   ## Transient/debug activity

   Mention grouped debug/smoke-test activity only if useful.
   Example:
   “Several short debug runs were used to validate config changes and were not kept as research evidence.”

   ## Key conclusions

   Bullet list of what was actually learned today.
   Focus on conclusions that affect future work.

   ## Decisions made

   Bullet list of decisions.
   Examples:
   - continue a given line of experiments
   - abandon a variant
   - compare against a baseline
   - inspect qualitative outputs
   - rerun with different seed/loss weight

   ## Open questions

   Bullet list of unresolved questions.

   ## Next actions

   Concrete checklist for the next work session.

11. Style requirements:
   - Be concise.
   - Be factual.
   - Clearly separate facts from interpretation.
   - Do not overstate results.
   - If a metric/result is unavailable, write “not available” or “needs evaluation”.
   - If the purpose of a run is inferred from the name/config, say “appears to test” rather than pretending certainty.
   - Prefer relative paths, not absolute paths.
   - Do not include huge logs or long diffs.
   - Do not paste full configs.
   - Do not duplicate reproducibility metadata that already lives in the output folder.

12. Important interpretation rule:
   A run deserves detailed journal space only if it changed what I believe or what I should do next.

13. At the end, print a short summary of what you created or updated:
   - journal file path
   - number of modified experiment folders found today
   - number classified as transient, rapid, notable, and milestone
   - any uncertainty or missing information
