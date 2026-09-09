# Stable 0.4.2 validation — 2026-09-09

Historical evidence below belongs to the original 0.4.2 lane, which has since
been published. It does not certify the production-based 0.4.3 consolidation
candidate. That candidate must pass its own full suite, CI and package checks.

Candidate worktree: `wt-containers-v010-stable`, branch
`codex/v010-stable-packages`, based on `0465124`.

- Full suite: **910 passed, 10 skipped**, 111.71 seconds.
- Source Ruff and `git diff --check`: passed.
- Stable wheel and source distribution build; Twine checks passed.
- No provider-backed or paid evaluation was run.

The eight historical failures were obsolete expectations, not waived behavior:
the metadata checks now include environment version/live annotation operations;
C2 still requires an absent/null reward before completion, then verifies scoring;
task-catalog tests cover both scored and omitted rewards, including restart.
The workflow now runs the entire suite with no deselections.

The first full run additionally failed the annotation test's timing-floor
measurement under concurrent builds. The complete rerun passed, including all
load tests, without changing thresholds or scheduler behavior.

At the time of this original receipt, publication was incomplete. Review/merge,
protected CI, immutable tag, trusted publishing, and public-index installation
remain gates for every new candidate. TBLite is not
a production dependency or publication prerequisite.
