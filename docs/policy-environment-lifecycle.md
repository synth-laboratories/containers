# Policy and environment lifecycle

Images can run an engine child behind a long-lived compatibility façade.
Gold-family runtimes call that engine over HTTP. Policy execution is a separate
boundary: code policies run in an isolated policy process, while NanoHorizon
constructs a planner from the rollout's pinned revision and named configuration.

For targets advertising the corresponding affordances:

| Operation | Meaning |
| --- | --- |
| `PUT /policy` | Install/select content-addressed source, harness, configuration/model identity and source revision. Returns `policy_revision_id`. |
| `GET /policy` | Read the selected identity and digests without source or credentials. |
| `POST /policy-configs` or `/policy-configs/{id}` | Register a named configuration. Identical retries succeed; a changed payload at the same ID returns `409 policy_config_conflict`. Use a new ID for a new sampler/checkpoint configuration. |
| `POST /policy/restart` | Restart the façade's selected policy child, where that harness uses one. Returns its revision and generation without restarting the engine. |
| `POST /rollouts` | Pin `policy_revision_id` and `policy_ref` to the desired source and sampler configuration. |

Updating the selected policy does not replace revisions already pinned by
rollouts. Gold code-policy rollouts own their own policy processes. Re-selecting
an older installed revision restores its source before restarting the façade's
policy child; repeating the current selection does not restart it again.

Policy mutations use a dedicated lock and run outside the HTTP event loop. A
child waiting during startup does not hold the platform state lock or block
health/policy reads. Startup timeout and readiness remain separate follow-ups.

These revision/config registries are currently process-local. A policy install
receipt establishes the stored definition; process startup errors remain in its
isolation receipt. Durable revision storage and explicit readiness/failed-start
states still need implementation before claiming recovery across façade loss.

PID 1 can restart a crashed engine child on its pinned internal port, bounded by
`SYNTH_CHILD_RESTART_LIMIT`. That does not prove environment state was restored.
`POST /world/restart` currently returns `409 environment_restart_unsupported` for
unsupported targets and `501 environment_restart_not_implemented` otherwise.
The world-stop façade records a stopped world; it is not a general provider
deletion actuator. Restore-capable environment restart must use the target's
checkpoint contract and retained state before this API can advertise success.
