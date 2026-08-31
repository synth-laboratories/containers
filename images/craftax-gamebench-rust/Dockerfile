# Craftax GameBench gold — one container, engine baked in.
#
#   stage rust-builder   cargo build --release --bin craftax_gold
#   stage runtime        python + synth-containers + craftax_gold/ + the binary
#                        and its defaults/ at the layout task_dir() expects
#
# PID 1 is `python -m craftax_gold`; the rust engine is a process CHILD of PID 1
# inside this container. `docker stop` kills the pid namespace, so nothing leaks.
#
# Named build contexts:
#   containers = the synth-containers source checkout
#   gamebench  = gamebench/tasks/craftax-singleplayer (crate + defaults/)

# --------------------------------------------------------------- rust-builder
FROM rust:1.90-slim-bookworm AS rust-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# task_dir() is CARGO_MANIFEST_DIR/.. baked at COMPILE time. Build at the exact
# path the runtime stage will hold, or defaults/ resolves to nothing.
WORKDIR /opt/gamebench/tasks/craftax-singleplayer
COPY --from=gamebench gold_rust/Cargo.toml gold_rust/Cargo.lock ./gold_rust/
COPY --from=gamebench gold_rust/src ./gold_rust/src
COPY --from=gamebench defaults ./defaults

RUN cargo build --release --locked --manifest-path gold_rust/Cargo.toml --bin craftax_gold

# -------------------------------------------------------------------- runtime
FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=containers . /opt/synth-containers
RUN pip install --no-cache-dir /opt/synth-containers && rm -rf /opt/synth-containers

# Engine + the data it reads relative to its own compiled task_dir().
COPY --from=rust-builder /opt/gamebench/tasks/craftax-singleplayer/gold_rust/target/release/craftax_gold \
     /opt/gamebench/tasks/craftax-singleplayer/gold_rust/target/release/craftax_gold
COPY --from=gamebench defaults /opt/gamebench/tasks/craftax-singleplayer/defaults
COPY --from=gamebench shared/assets/craftax /opt/gamebench/tasks/craftax-singleplayer/shared/assets/craftax

WORKDIR /app
COPY . /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    SYNTH_CRAFTAX_GOLD_BIN=/opt/gamebench/tasks/craftax-singleplayer/gold_rust/target/release/craftax_gold \
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

