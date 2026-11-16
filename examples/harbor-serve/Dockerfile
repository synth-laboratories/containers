# Build from the repository root; both input images must be resolved by the package builder.
ARG PYTHON_IMAGE
FROM ${PYTHON_IMAGE}
ARG SYNTH_WHEEL
COPY ${SYNTH_WHEEL} /opt/wheels/
RUN python -c 'from pathlib import Path; import subprocess, sys; wheels=list(Path("/opt/wheels").glob("*.whl")); assert len(wheels)==1; subprocess.run([sys.executable,"-m","pip","install","--no-cache-dir",str(wheels[0])+"[harbor-serve]"],check=True)' \
    && rm -r /opt/wheels
COPY examples/harbor-serve/tasks/ /tasks/
RUN mkdir -p /var/lib/harbor
EXPOSE 8390
CMD ["synth-harbor-serve", "--task-root", "/tasks", "--max-concurrent", "4"]
