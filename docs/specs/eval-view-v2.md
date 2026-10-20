# Eval view and index projections

`eval_trace_view_v2` and `eval_index_v1` are typed read models projected from
Trace V5. The source trace, its content identity and retained evidence remain
authoritative. The view keeps planned, executed and rejected actions separate,
and media records declare their provenance.

Detailed views omit unavailable optional facts. Index V1 requires an observed
seed and call count: `project_eval_index_row` raises `code=missing_seed` or
`code=missing_calls` when either is unknown. It never substitutes zero. A
measured seed of zero or a measured zero-call run remains valid. A trace can
still be projected to its detailed view when it cannot produce an index row.

The `synth-trace project eval-view` command loads a bundle and validates its
projected view before writing it. `synth-trace validate-eval-view` validates
view or index documents. ATIF export is an interoperability projection and
does not replace the native trace or its evidence custody.
