# Release: synth-containers

Release this package independently from the repository root and from other
packages in this monorepo.

## Build

Run from `packages/synth-containers/`:

```bash
uv run --group dev ruff check src
uv run --group dev ty check src
uv build
uv run --group dev twine check dist/*
```

For cookbook-facing releases, also compile the touched cookbook entrypoints
from the repository root:

```bash
PYTHONPATH=packages/synth-containers/src python -m py_compile $(rg --files cookbooks -g '*.py')
```

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

Publish automation is intentionally TBD until the repository-level packaging
pipeline is chosen.
