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

## Publish

After confirming the version and inspecting the generated artifacts:

```bash
uv publish dist/*
```

Publish automation is intentionally TBD until the repository-level packaging
pipeline is chosen.
