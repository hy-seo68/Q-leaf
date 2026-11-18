# Upstream provenance

- Upstream: https://github.com/ZJULearning/nsg
- Commit: `eceb09e2754d978b4c8b07e3248018987feb597a`
- Local changes: Public compact-graph accessors and entry-point overloads; distance counters; entry initialization and prefetch changes; array-delete corrections; const parameter lookup; build compatibility and optional test allocator linkage.
- Patch against that commit: [`../../patches/nsg.patch`](../../patches/nsg.patch). Apply from the upstream root with `patch -p1`.
- The root Q-LEAF CMake builds the required library sources directly.
