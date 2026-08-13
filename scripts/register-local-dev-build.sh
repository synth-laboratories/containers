#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
registry_root="${HOME}/.synth-desktop/dev-builds/synth-containers"
package_version=$(uv run --directory "$repo_root" python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
source_commit=$(git -C "$repo_root" rev-parse HEAD)
temporary_root=$(mktemp -d "${TMPDIR:-/tmp}/synth-containers-register.XXXXXX")
trap 'rm -rf "$temporary_root"' EXIT HUP INT TERM

source_fingerprint=$(
    {
        printf '%s\n' "$source_commit"
        git -C "$repo_root" diff --binary HEAD -- pyproject.toml src
        git -C "$repo_root" -c core.quotePath=false ls-files --others --exclude-standard -- pyproject.toml src |
            while IFS= read -r file_path; do
                shasum -a 256 "$repo_root/$file_path"
            done
    } | shasum -a 256 | awk '{print $1}'
)
build_root="$registry_root/$package_version/by-source/$source_fingerprint"

if test ! -x "$build_root/.venv/bin/synth-trace"; then
    uv build --directory "$repo_root" --wheel --out-dir "$temporary_root/dist"
    wheel_path=$(find "$temporary_root/dist" -type f -name '*.whl' -print -quit)
    test -n "$wheel_path"
    wheel_sha256=$(shasum -a 256 "$wheel_path" | awk '{print $1}')
    mkdir -p "$build_root"
    uv venv "$build_root/.venv"
    uv pip install --python "$build_root/.venv/bin/python" "$wheel_path"
    manifest_path="$build_root/build.toml"
    {
        printf 'schema = "synth.local-dev-build.v1"\n'
        printf 'package = "synth-containers"\n'
        printf 'version = "%s"\n' "$package_version"
        printf 'source_commit = "%s"\n' "$source_commit"
        printf 'source_fingerprint = "%s"\n' "$source_fingerprint"
        printf 'wheel_sha256 = "%s"\n' "$wheel_sha256"
        printf 'cli = "%s"\n' "$build_root/.venv/bin/synth-trace"
    } > "$manifest_path"
fi

installed_version=$($build_root/.venv/bin/synth-trace version)
if test "$installed_version" != "$package_version"; then
    echo "registered synth-containers version mismatch: expected $package_version, got $installed_version" >&2
    exit 1
fi

mkdir -p "$registry_root/$package_version"
link_path="$registry_root/$package_version/current"
ln -sfn "$build_root" "$link_path"

printf 'Registered synth-containers %s\n%s\n' "$package_version" "$link_path/.venv/bin/synth-trace"
