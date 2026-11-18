// Check Q-LEAF data structures.
#include <cassert>
#include <cmath>

#include "qleaf_gvp.h"

int main() {
  const float values[] = {1.f, 2.f, 3.f, 4.f};
  qleaf::VectorPool pool;
  pool.base = values;
  pool.stride = 2;
  pool.dim = 2;
  pool.n = 2;

  qleaf::LabelIndex label;
  label.name = "test";
  label.l2g = {0, 1};
  label.g.n = 2;
  label.g.width = 1;
  label.g.off = {0, 1, 2};
  label.g.adj = {1, 0};

  qleaf::PackedLabel packed;
  packed.build(label, pool, true);
  assert(packed.node_size() == (2 + 1) * 4 + (1 + 1) * 4);
  assert(packed.bytes() == packed.node_size() * 2);
  assert(std::fabs(packed.norm_optimized(0) - 5.f) < 1e-6f);
  assert(packed.vec_optimized(0)[0] == 1.f && packed.vec_optimized(0)[1] == 2.f);
  uint32_t degree = 0;
  const uint32_t* neighbors = packed.nbrs_optimized(0, degree);
  assert(degree == 1 && neighbors[0] == 1);

  qleaf::PackedOptimizedAccess access{label, packed, 2};
  const float query[] = {1.f, 1.f};
  assert(std::fabs(access.dist(query, 0) - (-1.f)) < 1e-6f);
  return 0;
}
