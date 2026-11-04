# Candidate22: interrupted discovery and known failures

This checkpoint is not green. Full Python discovery was interrupted after a
float64 JVP output validation error and before a new fixture that attempted to
construct 2^45 empty Python lists from an empty tensor. The separate metadata
gate aborted in that fixture. No full Python pass count is claimed.

The cumulative focused gate passed 12 of 13 methods. Correcting the huge-empty
fixture also exposed a Torch tape reuse fixture missing retain_graph=True.
Both original failures and the interrupted discovery log remain hash-recorded.
All eight standalones, native workspace and native DLPack checks passed.

[Candidate22b](../candidate22b/README.md) fixes the actual forward wrapper/Cast gap and both
fixtures, then passes the full matching-build validation. [manifest.json](manifest.json)
retains candidate22 source/wheel hashes and raw evidence.
