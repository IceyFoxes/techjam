# Repository Instructions

## Start Here

- Read [`TASK.md`](TASK.md) before researching or implementing anything. It is the authority for scope and deliverables.
- My personal role on this team is recorded in [`MY_ROLE.md`](MY_ROLE.md) (git-ignored) and follows the split in [`research/team-coordination/four-way-team-split.md`](research/team-coordination/four-way-team-split.md). `MY_ROLE.md` may also include a very basic statement on current research progress and aim, and should be updated regularly as research and implementation progresses. If the file 
does not exist, check with the user and add accordingly
- Use [`torch_transformer_benchmark.py`](torch_transformer_benchmark.py) as the authoritative benchmark and reference implementation, including executable numerical-correctness behavior.
- This is a shared research and development repository for the Transformer GPU-kernel task.

## Research

- Commit research directly to `master`; reserve implementation changes for branches and merge requests.
- Store essential findings under `research/`. Use `research/README.md` as the high-level index, a `README.md` in each topic directory as its local index, and focused leaf documents for full detail. Each deeper level should be more specific than its parent.
- Research is non-destructive. Do not delete or overwrite a research file because its conclusions may be outdated. Mark it `superseded`, `stale`, or `disputed` in the nearest index, with the date, reason, and a link to the replacement or contrary evidence.
- When an official clarification changes the task, preserve the previous wording in `TASK.md`, label its status, and add the new dated wording and source.
- For every public source, include its URL, access date, and a concise summary of the details relevant to this task. For source code, also identify the repository revision and relevant file or symbol when available.
- Before research work or rebasing, run `git switch master` and `git fetch origin master`, then read incoming commit messages with `git log --oneline HEAD..origin/master`. Decide whether they affect the current task; inspect relevant commits and investigate their impact before continuing, otherwise proceed with `git rebase origin/master`. Commit focused research checkpoints and `git push origin master` frequently. If a push is rejected, repeat this review-and-rebase workflow and preserve both contributors' work when resolving conflicts.
- Never force-push `master` or discard another contributor's changes.

## Code Changes

- Create a separate branch from the latest `master` for implementation, benchmark, test, build, or runtime configuration changes. Do not commit these changes directly to `master`.
- Place all working code under `src/`.
- Do not edit, move, rename, format, or otherwise modify the root `torch_transformer_benchmark.py`. If an editable harness is needed, copy it under `src/` and modify only the copy.
- Push branch checkpoints frequently and synchronize with `master` without rewriting a shared branch.
- Submit code through an MR/PR that explains the problem, decision and alternatives, affected behavior, expected performance impact, risks, numerical-correctness evidence, benchmark environment and results, and verification commands.

## Testing

- Preserve baseline, optimization-checkpoint, regression, and final benchmark runs under `research/benchmarks/<date>-<gpu>-<commit>/`; do not commit every exploratory run.
- For each preserved run, record the exact command, Git commit, timestamp, input shapes, dtype, correctness result, latency, and speedup, plus CPU, GPU, OS, GPU driver, CUDA, and PyTorch versions.
- Keep small Markdown or JSON results in Git. Store large profiler traces outside Git and link to them from the run document.
- Never overwrite a recorded run. Mark invalid, stale, or superseded runs in `research/benchmarks/README.md`, with the date, reason, and replacement link when applicable.
