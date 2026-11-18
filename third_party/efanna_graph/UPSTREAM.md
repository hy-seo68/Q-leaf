# Upstream provenance

- Upstream: https://github.com/ZJULearning/efanna_graph
- Commit: `50c4445a9e654f845213f9454106bc038a278764`
- Local changes: Move constructor and assignment for nhood; CMake compatibility; optional PQ source removal and disabled FAISS example target.
- Patch against that commit: [`../../patches/efanna_graph.patch`](../../patches/efanna_graph.patch). Apply from the upstream root with `patch -p1`.
- The root Q-LEAF CMake builds the required library sources directly.
