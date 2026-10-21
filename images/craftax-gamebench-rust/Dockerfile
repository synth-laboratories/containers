# Runtime assembly only. The explicit mbx preparation duty produces the frozen
# craftax_engine context; containers_wheel contains one qualified SDK wheel.
FROM python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70 AS runtime

COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /tmp/requirements.lock \
    && rm /tmp/requirements.lock
COPY --from=containers_wheel *.whl /tmp/containers-wheel/
RUN set -eu; set -- /tmp/containers-wheel/*.whl; test "$#" -eq 1; \
    python -m pip install --no-cache-dir --no-deps "$1"; rm -rf /tmp/containers-wheel

# CARGO_MANIFEST_DIR was fixed by the compiler duty to this task root.
COPY --from=craftax_engine . /opt/gamebench/tasks/craftax-singleplayer/
COPY verify_engine_artifact.py /tmp/verify_engine_artifact.py
RUN python /tmp/verify_engine_artifact.py /opt/gamebench/tasks/craftax-singleplayer \
    && rm /tmp/verify_engine_artifact.py

WORKDIR /app
COPY craftax_gold /app/craftax_gold
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    SYNTH_CRAFTAX_GOLD_BIN=/opt/gamebench/tasks/craftax-singleplayer/craftax_gold \
    GAMEBENCH_CRAFTAX_ASSETS_DIR=/opt/gamebench/tasks/craftax-singleplayer/shared/assets/craftax \
    SYNTH_CONTAINER_PORT=8080 \
    SYNTH_CONTAINER_BIND_ANY=1 \
    SYNTH_CONTAINER_STORAGE=/var/lib/synth/storage
RUN mkdir -p /var/lib/synth/storage \
    && test -x "${SYNTH_CRAFTAX_GOLD_BIN}" \
    && test -f /opt/gamebench/tasks/craftax-singleplayer/defaults/worlds.json \
    && test -f "${GAMEBENCH_CRAFTAX_ASSETS_DIR}/grass.png" \
    && test -f "${GAMEBENCH_CRAFTAX_ASSETS_DIR}/player-down.png"
EXPOSE 8080
VOLUME ["/var/lib/synth/storage"]
CMD ["python", "-m", "craftax_gold", "--host", "0.0.0.0", "--port", "8080"]
