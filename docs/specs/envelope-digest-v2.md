# Envelope digest v2

`synth.envelope-digest.v2` is the byte contract for the 16-hex `digest` on a
rollout journal envelope. Producers include `"digest_schema":
"synth.envelope-digest.v2"` on every envelope that uses this contract.

The digest is the first 16 lowercase hex characters of SHA-256 over:

```text
utf8("synth.envelope-digest.v2") || 0x00 || encode({
  "kind": kind,
  "sequence": sequence,
  "payload": payload
})
```

`encode` is recursive and every value is tagged:

| Value | Bytes |
| --- | --- |
| null | `n` |
| false / true | `f` / `t` |
| integer | `i` + base-10 integer + `;` |
| finite binary64 float | `d` + 16 lowercase hex IEEE-754 bits + `;` |
| string | `s` + UTF-8 byte length + `:` + raw UTF-8 bytes |
| array | `a` + item count + `[` + encoded items + `]` |
| object | `o` + member count + `{` + encoded key/value pairs + `}` |

Object keys are strings sorted lexicographically by their UTF-8 bytes. Integer
values are limited to the interoperable serde JSON domain (`i64` through
`u64`). Non-finite floats are invalid. The binary64 bit encoding intentionally
preserves negative zero and makes the digest independent of JSON exponent
formatting. Raw UTF-8 makes it independent of JSON Unicode escaping.

The normative vectors live at
`contracts/fixtures/journal/envelope-digest-v2.json`. Consumer repositories
vendor that file and exercise it in their own implementation tests. A new
encoding requires a new `digest_schema`; never reinterpret an existing marker.

Unmarked envelopes use the legacy implementation-specific digest. They may be
recovered and verified under that legacy rule, but must never be silently
upgraded because their event-chain head already commits to the old digest.
