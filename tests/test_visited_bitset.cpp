// Check Q-LEAF data structures.
#include <cassert>

#include "qleaf_gvp.h"

int main() {
  qleaf::Scratch scratch;
  scratch.begin<true>(130, 10);
  assert(scratch.visited_bits.size() == 3);
  assert(!scratch.seen<true>(1));
  assert(scratch.seen<true>(1));
  assert(!scratch.seen<true>(65));
  assert(scratch.touched_words.size() == 2);
  assert(scratch.visited_bits[0] != 0);
  assert(scratch.visited_bits[1] != 0);
  assert(scratch.visited_bits[2] == 0);

  scratch.end<true>();
  assert(scratch.touched_words.empty());
  assert(scratch.visited_bits[0] == 0);
  assert(scratch.visited_bits[1] == 0);
  assert(scratch.visited_bits[2] == 0);

  scratch.begin<true>(130, 10);
  assert(!scratch.seen<true>(1));
  scratch.end<true>();
  assert(scratch.visited_bits[0] == 0);
  return 0;
}
