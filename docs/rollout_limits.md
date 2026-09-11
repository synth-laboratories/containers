# Observed rollout limits

`synth_containers.rollout_limits` contains the first shared limit-decision
component used by public optimizer evaluation. `RolloutLimits` currently
declares positive finite work duration and a positive output-byte threshold.
`RolloutLimitSupervisor` starts its monotonic clock before execution and accepts
output samples from the execution owner. It returns one immutable, sticky
`RolloutLimitDecision` when the deadline is reached or output exceeds its limit.

The decision is a request to stop, not proof of termination. Its payload fits
inside the existing worker/lifecycle event family; it introduces no replacement
stream schema. Adapters own their stop actuators and report failures explicitly.
CPU/memory controls, concurrency admission and provider spend accounting retain
their existing owners.

## Initial optimizer integration

The optimizer OCI executor maps its existing `TrialLimits` fields into these
shared limits. It samples output before and after process waits, uses monotonic
elapsed time, and invokes the stop actuator even when a limit-event observer
fails. Heartbeat failures also clean up the owned process. An already cancelled
trial does not launch. Output-limit stops are failed/incomplete evidence rather
than scores read from a target's otherwise successful result.

The OCI kill command has an explicit timeout. Failure to confirm its stop is an
infrastructure error requiring reconciliation; killing the local CLI does not
prove that a remote Docker daemon stopped the container.

## Current guarantees and limits

- The output check is sampled retained footprint, not a filesystem quota. It
  can overshoot between polls and cannot observe temporary files removed between
  samples. Live inspection also depends on the filesystem responding.
- The stop decision remains sticky when later output samples become smaller.
- The supervisor is single-owner and in-process. It is not a durable restart
  checkpoint or an independent watchdog surviving worker failure.
- No provider spend, token, step or parent/child reservation enforcement is
  implemented by this module yet.
- Docker/Daytona parity, hosted recovery, persistent enforcement receipts and
  the complete container deployment lifecycle remain unqualified.

Deterministic component and fake-OCI boundary checks cover this initial slice.
They are not real-provider or Cloud release acceptance evidence.
