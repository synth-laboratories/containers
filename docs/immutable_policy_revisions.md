# Immutable policy revisions

`PUT /policy` installs an immutable policy revision. Its `harness`, source,
configuration identity, model identity, and source revision participate in the
revision digest. Repeating an identical PUT is idempotent; changing any of that
material produces a new `policy_revision_id`.

The most recently installed revision is the default pointer returned by
`GET /policy`. It is not the only runnable revision and installing another
policy does not deactivate older revisions.

NanoHorizon rollouts select an installed revision explicitly:

```json
{
  "policy_ref": {
    "harness": "nanohorizon",
    "config": "goal-policy-sampler"
  },
  "policy_revision_id": "polrev_0123456789abcdef",
  "task_instance_id": "seed:780005"
}
```

The selected revision's harness must match `policy_ref.harness`. The sampler
configuration is registered independently with `POST /policy-configs`; it may
vary by rollout without changing the installed source revision. This permits
multiple prompt, tool-protocol, and harness-source variants to run concurrently
in one container, with every rollout pinned to its own immutable revision.

`trace.opened`, rollout responses, and task-catalog entries carry the selected
`policy_revision_id`, so consumers can group and compare results without
inferring policy identity from rollout names or from the latest `GET /policy`
response.
