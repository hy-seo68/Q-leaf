// Check Q-LEAF data structures.
#include <cassert>
#include <vector>

#include "qleaf_gvp.h"

int main() {
  std::vector<qleaf::Result> candidates{{1.0f, 7}, {2.0f, 8}, {3.0f, 7}, {4.0f, 9}};
  qleaf::Result output[4]{};

  const size_t count = qleaf::merge_topk(candidates, 4, output);

  assert(count == 3);
  assert(output[0].gid == 7);
  assert(output[1].gid == 8);
  assert(output[2].gid == 9);
  return 0;
}
