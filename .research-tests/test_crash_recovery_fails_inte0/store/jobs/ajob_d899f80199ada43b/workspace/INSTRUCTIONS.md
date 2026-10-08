# Trace V5 annotation task — Environment step status



## Non-negotiable rules

- You are inspecting one immutable, sealed Trace V5. You cannot change it, and nothing you say changes the recorded run, its reward, or its engine state.
- Use only the trace inspection tools you were given. There is no shell, no network, and no file access. Do not attempt them.
- Every applied finding must cite evidence selectors you verified with `trace_resolve_selector`. Uncited or unresolvable findings are discarded.
- Cite at least 1 distinct evidence selector(s) per finding. Prefer `part` or `message` selectors with a `quote` for text, `span` selectors for environment steps, and `event` selectors for engine facts.
- A `quote` must be a verbatim substring of the entity text exactly as the tools render it (`trace_get_message` / `trace_get_event` / `trace_resolve_selector`): copy it character for character, including spacing and punctuation. Quote short scalar phrases — a sentence fragment, one `key: value` pair as rendered — and never re-type, reflow, or summarise JSON (`{"actions": …}` retyped from memory is a `quote_mismatch`). If you are not sure of the exact text, fetch the entity with `trace_get_message` / `trace_get_event` and copy from it; a selector without a `quote` is always valid when the whole part or event is the evidence.
- If the evidence you need is absent, emit an abstention with a typed reason. Never guess, never complete a finding from memory or from summaries.
- `rationale` is a concise, retained explanation (under 2000 characters). Do not include hidden or step-by-step reasoning; it is neither requested nor stored.
- Engine facts (reward, achievements, actions, inventory, termination) are authoritative; annotations never override them.
- Labels must come from the taxonomy below; unknown labels invalidate the finding.

## Taxonomy

Annotation types: environment_step

- `environment_step.ok`: span status ok
- `environment_step.error`: span status error
- `environment_step.no_effect`: the engine reported a no-op transition
- `environment_step.blocked`: the engine reported a blocked transition

Payload fields:
- `action` (string, optional): 
- `transition` (string, optional): 
- `reason` (string, optional): 
- `step_index` (integer, optional): 
## Output

Reply with a single JSON object conforming to `synth.annotation-proposal.v1` (see output_schema.json). Set `source_trace_id` and `source_trace_digest` to:

- source_trace_id: `trace_db61f67b4e043a03`
- source_trace_digest: `sha256:0200fb51b48c94283b663531ff2fab26126237a4ed81a4eb2c7663be95fccdbf`

Strict output format: every field is present (use null for unknown optionals); `payload_json` is the payload object JSON-encoded as a string; `rationale` stays under 2000 characters.

## Limits

- tool calls: 200
- bytes per tool response: 32000
- cumulative tool bytes: 2000000
- total tokens: 400000

