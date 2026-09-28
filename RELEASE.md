# Release: synth-containers

Release this package from the root of this standalone repository. It is
versioned and published independently of other Synth packages.

## Build

Run from the repository root:

```bash
uv run --group dev pytest tests
uv run --group dev ruff check src
python3 scripts/check-type-debt.py
uv build
uv run --group dev twine check dist/*
```

The eight historical metadata/reward failures have been reconciled with the
current contracts. Missing rewards are still asserted before completion;
terminal and recovered catalogs now test both scored and omitted rewards.
The publish workflow runs the entire suite without deselections. Full-suite
verification remains mandatory, including timing-sensitive annotation tests.
The type-debt gate retains an explicit 173-diagnostic historical baseline;
the integrated candidate currently reports 172 diagnostics,
mostly `invalid-argument-type` and `unresolved-attribute`; annotating that
surface is separate work. Neither is a licence to add more.

For cookbook-facing releases, also compile the touched cookbook entrypoints.
Cookbooks live in a separate checkout (for example
[`synth-cookbooks-public`](https://github.com/synth-laboratories/synth-cookbooks-public));
from this repository root, point the command at that checkout:

```bash
COOKBOOKS_DIR=../synth-cookbooks-public
uv run python -m py_compile $(rg --files "$COOKBOOKS_DIR" -g '*.py')
```

Internal developer-machine registration of a local build is optional and not
part of the release flow; see `docs/internal/local-dev-build.md`.

## Changelog

- Update `changelog.log` in the same change that updates package version or release docs.
- Organize entries by day: `## YYYY-MM-DD`.
- Keep the file terse: about 20 total lines for the current daily-dev window.
- Use bullets only; no paragraphs, migration snippets, or install code blocks.
- Prefer shipped user-facing changes over implementation narration.
- Link merged PRs where available, for example `[PR #2](https://github.com/synth-laboratories/containers/pull/2)`.
- Include the PyPI version in one bullet when a package was published.
- Keep unreleased or blocked work explicit and short.

## Publish

After confirming the version and inspecting the generated artifacts:

```bash
uv publish dist/*
```

The `publish-pypi.yml` workflow validates official version tags, tests and builds
the package, then publishes through the protected `pypi-release` environment
using PyPI trusted publishing. Never create a public version tag before all
release gates pass. Published `0.4.2` is immutable. The production-based
consolidation candidate is `0.4.3`; its gates and publication remain pending.
