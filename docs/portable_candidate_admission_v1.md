# Portable candidate admission v1

`PortableCandidateRegistry` is the credential-free, local register-then-run
boundary for `synth.candidate.v1` and `synth.task-contract.v1`.

1. Registration parses and strictly validates both schemas and self-digests,
   checks the candidate's declared task digest, and records both canonical
   identities plus hashes of the exact input bytes.
2. Run admission requires the immutable registration ID and the exact candidate
   and task bytes. It revalidates both documents immediately before execution.
3. The declared evaluator and seed must be admitted by the task contract.
4. The returned admission carries exact execution, evaluation, and trajectory
   references. Callers attach content-specific receipts when those bodies exist;
   absent values remain absent and are never represented as zero.

No folder, filename, tag, port, current process, or timestamp participates in
candidate or task identity. The registry does not open Workshop SQLite and has
no backend or cloud dependency.

The fixtures in `tests/fixtures/portable-contracts-v1` are release-stamped copies
of the authority in Optimizers `contracts/synth-spine-v1`. Their candidate and
task digests are regression assertions; update them only as a new contract
version, never by reinterpreting v1.
