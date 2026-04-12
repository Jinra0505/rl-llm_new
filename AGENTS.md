# AGENTS.md

## Purpose

This repository uses `requirements.txt` as the authoritative dependency file for Python packages.
When working in this repo, you must actively maintain `requirements.txt` so that dependencies
discovered during the task are recorded and reusable in future runs.

## Python dependency policy

1. Treat `requirements.txt` as the single source of truth for Python dependencies.
2. Before running any Python code, read `requirements.txt` first and prefer packages that are already listed there.
3. If `requirements.txt` does not exist, create it.
4. Whenever you encounter a missing Python package during coding, testing, linting, execution, or debugging:
   - first check whether it is already listed in `requirements.txt`;
   - if it is not listed, add it to `requirements.txt`;
   - use a reasonably pinned version when possible;
   - then install it and continue the task.
5. Do not repeatedly perform ad hoc temporary installs without recording them in `requirements.txt`.
6. Reuse packages from `requirements.txt` in later steps instead of re-deciding dependencies from scratch each time.
7. Keep dependencies minimal and directly relevant to the task. Do not add unnecessary packages.
8. Do not remove existing dependencies from `requirements.txt` unless they are clearly incorrect, broken, obsolete for this repo, or being replaced for a specific reason.
9. Avoid duplicate entries and keep `requirements.txt` clean, normalized, and easy to read.
10. When choosing between multiple possible packages, prefer the most standard, reliable, and widely used package that best matches the actual import or error.
11. If a package is only needed to complete the current task, it should still be recorded in `requirements.txt` rather than installed silently.
12. If an import name differs from the package name, choose the correct installable package and record that package in `requirements.txt`.
13. After updating dependencies, continue the work instead of stopping at the dependency issue.

## Execution behavior

- Always inspect `requirements.txt` before installing anything.
- When a missing import or module error appears, update `requirements.txt` first, then install the required package, then proceed.
- Prefer commands that keep the environment aligned with the dependency file.
- If a single urgent package is needed, still record it in `requirements.txt` before or at the time of installation.
- Do not assume a package is available just because it was installed in a previous run; check the repository files and current environment.

## Change discipline

- Make the smallest dependency change that solves the actual problem.
- Do not add broad toolchains or optional extras unless they are clearly required.
- Keep version pinning reasonable and consistent with the rest of the file.
- If a version must be changed, preserve compatibility with the codebase as much as possible.

## Final response requirements

At the end of the task, include a short dependency summary that states:
- which packages were added or changed in `requirements.txt`;
- why each package was needed;
- whether any versions were pinned, upgraded, downgraded, or adjusted.

## Priority

These instructions apply for the entire task and should be followed unless the user explicitly gives a higher-priority instruction that overrides them.
