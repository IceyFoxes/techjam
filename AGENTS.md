# Repository Instructions

## Start Here

- Read [`TASK.md`](TASK.md) before researching or implementing anything. It is the authority for scope, numerical tolerances, and deliverables.
- This is a shared research and development repository for the Transformer GPU-kernel task.

## Research

- Commit research directly to `master`; reserve implementation changes for branches and merge requests.
- Store essential findings under `research/`. Use `research/README.md` as the high-level index, a `README.md` in each topic directory as its local index, and focused leaf documents for full detail. Each deeper level should be more specific than its parent.
- Research is non-destructive. Do not delete or overwrite a research file because its conclusions may be outdated. Mark it `superseded`, `stale`, or `disputed` in the nearest index, with the date, reason, and a link to the replacement or contrary evidence.
- When an official clarification changes the task, preserve the previous wording in `TASK.md`, label its status, and add the new dated wording and source.
- For every public source, include its URL, access date, and a concise summary of the details relevant to this task. For source code, also identify the repository revision and relevant file or symbol when available.
- Before research work, run `git switch master` and `git pull --rebase origin master`. Commit focused research checkpoints and `git push origin master` frequently. If a push is rejected, pull with rebase and preserve both contributors' work when resolving conflicts.
- Never force-push `master` or discard another contributor's changes.

## Code Changes

- Create a separate branch from the latest `master` for implementation, benchmark, test, build, or runtime configuration changes. Do not commit these changes directly to `master`.
- Push branch checkpoints frequently and synchronize with `master` without rewriting a shared branch.
- Submit code through an MR/PR that explains the problem, decision and alternatives, affected behavior, expected performance impact, risks, numerical-correctness evidence, benchmark environment and results, and verification commands.
