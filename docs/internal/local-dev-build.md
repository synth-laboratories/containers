# Internal: register a local development build

Internal developer-machine workflow only. It is not part of the public install
path or the release procedure in `RELEASE.md`.

After changing Containers or its package version, register the current checkout:

```bash
./scripts/register-local-dev-build.sh
```

The script builds a wheel, installs it into an immutable, versioned directory
under `~/.synth-desktop/dev-builds/synth-containers/`, verifies the installed
version, and atomically selects it for local desktop (Workshop) builds, which
resolve their checked-in version from that registry. No flags or environment
variables are required. Re-running it reuses an identical registered wheel.
