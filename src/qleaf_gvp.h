// Graph search with local graph IDs and global vector IDs.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace qleaf {

class MappedFile {
 public:
  MappedFile() = default;
  explicit MappedFile(const std::string& path) { open(path); }
  ~MappedFile() { close(); }
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  void open(const std::string& path) {
    close();
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open " + path);
    struct stat st {};
    if (fstat(fd_, &st) != 0) throw std::runtime_error("fstat failed: " + path);
    size_ = static_cast<size_t>(st.st_size);
    ptr_ = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (ptr_ == MAP_FAILED) {
      ptr_ = nullptr;
      throw std::runtime_error("mmap failed: " + path);
    }
  }
  void advise(int advice) const {
    if (ptr_) madvise(ptr_, size_, advice);
  }
  void close() {
    if (ptr_) munmap(ptr_, size_);
    if (fd_ >= 0) ::close(fd_);
    ptr_ = nullptr;
    fd_ = -1;
    size_ = 0;
  }
  const char* data() const { return static_cast<const char*>(ptr_); }
  size_t size() const { return size_; }

 private:
  int fd_ = -1;
  void* ptr_ = nullptr;
  size_t size_ = 0;
};

struct VectorPool {
  const float* base = nullptr;
  size_t stride = 0;
  size_t offset = 0;
  uint32_t dim = 0;
  size_t n = 0;
  inline const float* get(uint32_t gid) const { return base + static_cast<size_t>(gid) * stride + offset; }
};

inline VectorPool pool_from_fvecs(const MappedFile& f) {
  int32_t d = 0;
  std::memcpy(&d, f.data(), sizeof(int32_t));
  if (d <= 0) throw std::runtime_error("invalid fvecs header");
  VectorPool p;
  p.base = reinterpret_cast<const float*>(f.data());
  p.dim = static_cast<uint32_t>(d);
  p.stride = static_cast<size_t>(d) + 1;
  p.offset = 1;
  p.n = f.size() / (sizeof(float) * p.stride);
  return p;
}

class OwnedPool {
 public:
  OwnedPool() = default;
  OwnedPool(const OwnedPool&) = delete;
  OwnedPool& operator=(const OwnedPool&) = delete;
  ~OwnedPool() { std::free(mem_); }
  VectorPool load(const VectorPool& src) {
    std::free(mem_);
    const size_t huge = size_t(2) << 20;
    bytes_ = (src.n * src.dim * sizeof(float) + huge - 1) / huge * huge;
    mem_ = static_cast<float*>(std::aligned_alloc(huge, bytes_));
    if (!mem_) throw std::runtime_error("cannot allocate pool copy");
#ifdef MADV_HUGEPAGE
    madvise(mem_, bytes_, MADV_HUGEPAGE);
#endif
    for (size_t i = 0; i < src.n; ++i) std::memcpy(mem_ + i * src.dim, src.get(static_cast<uint32_t>(i)), src.dim * 4);
    VectorPool p = src;
    p.base = mem_;
    p.stride = src.dim;
    p.offset = 0;
    return p;
  }
  size_t bytes() const { return bytes_; }

 private:
  float* mem_ = nullptr;
  size_t bytes_ = 0;
};

inline float l2sq(const float* a, const float* b, uint32_t d) {
#if defined(__AVX2__)
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + 16 <= d; i += 16) {
    __m256 x0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    __m256 x1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(x0, x0));
    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(x1, x1));
  }
  for (; i + 8 <= d; i += 8) {
    __m256 x0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(x0, x0));
  }
  acc0 = _mm256_add_ps(acc0, acc1);
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  float r = _mm_cvtss_f32(s);
  for (; i < d; ++i) {
    float t = a[i] - b[i];
    r += t * t;
  }
  return r;
#else
  float r = 0.f;
  for (uint32_t i = 0; i < d; ++i) {
    float t = a[i] - b[i];
    r += t * t;
  }
  return r;
#endif
}

inline float dot(const float* a, const float* b, uint32_t d);

inline float fast_l2_rank(const float* x, const float* q, uint32_t d) {
  return dot(x, x, d) - 2.f * dot(x, q, d);
}

inline void prefetch(const void* p) { __builtin_prefetch(p, 0, 3); }
inline void prefetch_t1(const void* p) { __builtin_prefetch(p, 0, 2); }

struct Graph {
  uint32_t n = 0, width = 0, nsg_ep = 0;
  std::vector<uint32_t> off;
  std::vector<uint32_t> adj;
  inline const uint32_t* nbrs(uint32_t u, uint32_t& deg) const {
    deg = off[u + 1] - off[u];
    return adj.data() + off[u];
  }
  size_t bytes() const { return off.size() * sizeof(uint32_t) + adj.size() * sizeof(uint32_t); }
};

inline Graph load_nsg_graph(const std::string& path, uint32_t expected_n) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open graph " + path);
  const size_t bytes = static_cast<size_t>(in.tellg());
  if (bytes < 8 || bytes % 4) throw std::runtime_error("malformed graph file " + path);
  in.seekg(0);
  std::vector<uint32_t> buf(bytes / 4);
  in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(bytes));
  Graph g;
  g.width = buf[0];
  g.nsg_ep = buf[1];
  g.off.reserve(static_cast<size_t>(expected_n) + 1);
  g.off.push_back(0);
  g.adj.reserve(buf.size());
  size_t pos = 2;
  while (pos < buf.size()) {
    const uint32_t k = buf[pos++];
    if (pos + k > buf.size()) throw std::runtime_error("truncated graph file " + path);
    g.adj.insert(g.adj.end(), buf.begin() + static_cast<std::ptrdiff_t>(pos),
                 buf.begin() + static_cast<std::ptrdiff_t>(pos + k));
    pos += k;
    if (g.adj.size() > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("graph too large: " + path);
    g.off.push_back(static_cast<uint32_t>(g.adj.size()));
  }
  g.adj.shrink_to_fit();
  g.n = static_cast<uint32_t>(g.off.size() - 1);
  if (expected_n && g.n != expected_n)
    throw std::runtime_error(path + ": graph has " + std::to_string(g.n) + " nodes, idx has " +
                             std::to_string(expected_n));
  for (uint32_t v : g.adj)
    if (v >= g.n) throw std::runtime_error(path + ": neighbour id out of range");
  if (g.nsg_ep >= g.n) g.nsg_ep = 0;
  return g;
}

inline std::vector<uint32_t> load_idx(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open idx " + path);
  uint32_t c = 0;
  in.read(reinterpret_cast<char*>(&c), 4);
  std::vector<uint32_t> v(c);
  in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(c) * 4);
  if (!in) throw std::runtime_error("truncated idx " + path);
  return v;
}

inline std::vector<uint32_t> load_entry_points_txt(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open entry points " + path);
  std::vector<uint32_t> eps;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    long long local = -1;
    if (ss >> local && local >= 0) eps.push_back(static_cast<uint32_t>(local));
  }
  return eps;
}

template <typename T>
inline void deduplicate_preserve_order(std::vector<T>& values) {
  std::unordered_set<T> seen;
  seen.reserve(values.size());
  std::vector<T> unique;
  unique.reserve(values.size());
  for (const T& value : values)
    if (seen.insert(value).second) unique.push_back(value);
  values.swap(unique);
}

// Graph neighbors and entry IDs are local; l2g maps to the shared base pool.
struct LabelIndex {
  std::string name;
  Graph g;
  std::vector<uint32_t> l2g;
  std::vector<uint32_t> eps;
  std::vector<float> ep_vecs;

  void load(const std::string& label, const std::string& graph_file, const std::string& idx_file,
            const std::string& ep_file, const VectorPool& pool) {
    name = label;
    l2g = load_idx(idx_file);
    g = load_nsg_graph(graph_file, static_cast<uint32_t>(l2g.size()));
    for (uint32_t gid : l2g)
      if (gid >= pool.n) throw std::runtime_error(idx_file + ": global id out of range");
    eps.clear();
    if (!ep_file.empty() && ep_file != "-") {
      eps = load_entry_points_txt(ep_file);
      const size_t original_count = eps.size();
      deduplicate_preserve_order(eps);
      std::cout << "entry points " << ep_file << ": original=" << original_count
                << ", unique=" << eps.size() << '\n';
    }
    eps.erase(std::remove_if(eps.begin(), eps.end(), [&](uint32_t e) { return e >= g.n; }), eps.end());
    if (eps.empty()) eps.push_back(g.nsg_ep);
    ep_vecs.resize(eps.size() * pool.dim);
    for (size_t i = 0; i < eps.size(); ++i)
      std::memcpy(&ep_vecs[i * pool.dim], pool.get(l2g[eps[i]]), pool.dim * sizeof(float));
  }
  size_t bytes() const { return g.bytes() + l2g.size() * 4 + eps.size() * 4 + ep_vecs.size() * 4; }
};

class Postings {
 public:
  void open(const std::string& path) {
    f_.open(path);
    if (f_.size() < 16) throw std::runtime_error("malformed postings " + path);
    std::memcpy(&ncol_, f_.data(), 8);
    std::memcpy(&nnz_, f_.data() + 8, 8);
    const size_t expected = 16 + 8 * (ncol_ + 1) + 4 * nnz_;
    if (f_.size() != expected) throw std::runtime_error("postings size mismatch " + path);
    ptr_ = reinterpret_cast<const uint64_t*>(f_.data() + 16);
    ids_ = reinterpret_cast<const uint32_t*>(f_.data() + 16 + 8 * (ncol_ + 1));
  }
  bool is_open() const { return ptr_ != nullptr; }
  uint64_t num_tags() const { return ncol_; }

  void validate_offsets() const {
    if (!ptr_ || ptr_[0] != 0 || ptr_[ncol_] != nnz_)
      throw std::runtime_error("invalid packed postings endpoints");
    for (uint64_t tag = 0; tag < ncol_; ++tag)
      if (ptr_[tag] > ptr_[tag + 1] || ptr_[tag + 1] > nnz_)
        throw std::runtime_error("invalid packed postings offsets");
  }
  inline const uint32_t* list(uint64_t tag, size_t& count) const {
    count = static_cast<size_t>(ptr_[tag + 1] - ptr_[tag]);
    return ids_ + ptr_[tag];
  }

 private:
  MappedFile f_;
  uint64_t ncol_ = 0, nnz_ = 0;
  const uint64_t* ptr_ = nullptr;
  const uint32_t* ids_ = nullptr;
};

struct Result {
  float dist;
  uint32_t gid;
  bool operator<(const Result& o) const { return dist < o.dist || (dist == o.dist && gid < o.gid); }
};

struct Neighbor {
  uint32_t id;
  float dist;
  bool flag;
  bool operator<(const Neighbor& o) const { return dist < o.dist; }
};

// Keep entry-selection and graph/scan distance counts separate.
struct SearchStats {
  uint64_t ndc_entry = 0;
  uint64_t ndc_search = 0;
  uint32_t initial_candidates = 0;
};

enum class InitMode { NsgCompat, EntryOnly, AllEntries };

struct Scratch {
  std::vector<uint16_t> mark;
  uint16_t epoch = 0;
  std::vector<uint64_t> visited_bits;
  std::vector<uint32_t> touched_words;
  std::vector<Neighbor> pool;
  std::vector<uint32_t> init;
  std::vector<Neighbor> entry_candidates;
  std::vector<Result> buf;
  explicit Scratch(uint32_t = 0) {}

  template <bool Bitset>
  void begin(uint32_t n, uint32_t cap) {
    if constexpr (Bitset) {
      const size_t words = (static_cast<size_t>(n) + 63) / 64;
      if (visited_bits.size() < words) visited_bits.resize(words, 0);
      touched_words.clear();
    } else {
      if (mark.size() < n) {
        mark.assign(n, 0);
        epoch = 0;
      }
      if (++epoch == 0) {
        std::fill(mark.begin(), mark.end(), 0);
        epoch = 1;
      }
    }
    if (pool.size() < static_cast<size_t>(cap) + 1) pool.resize(static_cast<size_t>(cap) + 1);
    init.clear();
  }
  template <bool Bitset>
  inline bool seen(uint32_t i) {
    if constexpr (Bitset) {
      const uint32_t word = i >> 6;
      const uint64_t mask = uint64_t{1} << (i & 63);
      if (visited_bits[word] & mask) return true;
      if (visited_bits[word] == 0) touched_words.push_back(word);
      visited_bits[word] |= mask;
      return false;
    } else {
      if (mark[i] == epoch) return true;
      mark[i] = epoch;
      return false;
    }
  }
  template <bool Bitset>
  void end() {
    if constexpr (Bitset) {
      for (uint32_t word : touched_words) visited_bits[word] = 0;
      touched_words.clear();
    }
  }
  void begin(uint32_t n, uint32_t cap) { begin<false>(n, cap); }
  inline bool seen(uint32_t i) { return seen<false>(i); }
};

inline uint32_t insert_pool(Neighbor* pool, uint32_t& size, uint32_t cap, const Neighbor& nn) {
  uint32_t lo = 0, hi = size;
  while (lo < hi) {
    const uint32_t mid = (lo + hi) / 2;
    if (pool[mid].dist > nn.dist)
      hi = mid;
    else
      lo = mid + 1;
  }
  const uint32_t pos = lo;
  const uint32_t moved = (size < cap) ? size - pos : size - 1 - pos;
  std::memmove(pool + pos + 1, pool + pos, static_cast<size_t>(moved) * sizeof(Neighbor));
  pool[pos] = nn;
  if (size < cap) ++size;
  return pos;
}

inline uint32_t select_entry(const LabelIndex& li, const VectorPool& vp, const float* q, float& best_dist,
                             SearchStats& st) {
  uint32_t best = li.eps[0];
  best_dist = std::numeric_limits<float>::max();
  for (size_t i = 0; i < li.eps.size(); ++i) {
    const float d = l2sq(q, &li.ep_vecs[i * vp.dim], vp.dim);
    if (d < best_dist) {
      best_dist = d;
      best = li.eps[i];
    }
  }
  st.ndc_entry += li.eps.size();
  return best;
}

inline void select_all_entries(const LabelIndex& li, const VectorPool& vp, const float* q, float qnorm,
                               Scratch& s, SearchStats& st) {
  s.entry_candidates.resize(li.eps.size());
  for (size_t i = 0; i < li.eps.size(); ++i) {
    const float d = l2sq(q, &li.ep_vecs[i * vp.dim], vp.dim);
    s.entry_candidates[i] = Neighbor{li.eps[i], d - qnorm, true};
  }
  st.ndc_entry += li.eps.size();
}

inline size_t search_graph(const LabelIndex& li, const VectorPool& vp, const float* q, uint32_t L, uint32_t k,
                           uint32_t ep, float ep_dist, InitMode mode, Scratch& s, Result* out, SearchStats& st) {
  const Graph& g = li.g;
  const uint32_t n = g.n;
  if (n == 0 || k == 0) return 0;
  const uint32_t* l2g = li.l2g.data();
  const uint32_t dim = vp.dim;
  L = std::min(std::max(L, k), n);
  s.begin(n, L);
  Neighbor* pool = s.pool.data();
  uint32_t size = 0;

  uint32_t deg = 0;
  const uint32_t* nb = g.nbrs(ep, deg);
  const float qnorm = dot(q, q, dim);
  if (mode == InitMode::NsgCompat) {
    s.seen(ep);
    s.init.push_back(ep);
    for (uint32_t i = 0; i < deg && s.init.size() < L; ++i)
      if (nb[i] < n && !s.seen(nb[i])) s.init.push_back(nb[i]);
    while (s.init.size() < L) {
      const uint32_t id = static_cast<uint32_t>(std::rand()) % n;
      if (s.seen(id)) continue;
      s.init.push_back(id);
    }
    for (uint32_t id : s.init) prefetch(vp.get(l2g[id]));
    for (uint32_t id : s.init) pool[size++] = Neighbor{id, fast_l2_rank(vp.get(l2g[id]), q, dim), true};
    st.ndc_search += s.init.size();
    std::sort(pool, pool + size);
  } else {
    s.seen(ep);
    float d0;
    if (ep_dist >= 0.f) {
      d0 = ep_dist - qnorm;
    } else {
      d0 = fast_l2_rank(vp.get(l2g[ep]), q, dim);
      st.ndc_search++;
    }
    pool[size++] = Neighbor{ep, d0, true};
  }

  uint32_t kk = 0;
  while (kk < size) {
    uint32_t nk = std::numeric_limits<uint32_t>::max();
    if (pool[kk].flag) {
      pool[kk].flag = false;
      uint32_t du = 0;
      const uint32_t* nu = g.nbrs(pool[kk].id, du);
      for (uint32_t m = 0; m < du; ++m) prefetch(vp.get(l2g[nu[m]]));
      for (uint32_t m = 0; m < du; ++m) {
        const uint32_t v = nu[m];
        if (s.seen(v)) continue;
        const float d = fast_l2_rank(vp.get(l2g[v]), q, dim);
        st.ndc_search++;
        if (size == L && d >= pool[size - 1].dist) continue;
        const uint32_t r = insert_pool(pool, size, L, Neighbor{v, d, true});
        if (r < nk) nk = r;
      }
    }
    if (nk <= kk)
      kk = nk;
    else
      ++kk;
  }
  const size_t m = std::min<size_t>(k, size);
  for (size_t i = 0; i < m; ++i) out[i] = Result{pool[i].dist + qnorm, l2g[pool[i].id]};
  return m;
}

inline size_t scan_list(const uint32_t* ids, size_t count, const VectorPool& vp, const float* q, uint32_t k,
                        Scratch& s, Result* out, SearchStats& st) {
  s.buf.resize(count);
  for (size_t i = 0; i + 8 < count; i += 8) prefetch(vp.get(ids[i + 8]));
  for (size_t i = 0; i < count; ++i) s.buf[i] = Result{l2sq(q, vp.get(ids[i]), vp.dim), ids[i]};
  st.ndc_search += count;
  const size_t m = std::min<size_t>(k, count);
  std::partial_sort(s.buf.begin(), s.buf.begin() + static_cast<std::ptrdiff_t>(m), s.buf.end());
  std::copy(s.buf.begin(), s.buf.begin() + static_cast<std::ptrdiff_t>(m), out);
  return m;
}

struct PackedScanList {
  const uint32_t* ids = nullptr;
  size_t count = 0;
  uint32_t dim = 0;
  std::vector<float> vectors;

  void build(const uint32_t* posting_ids, size_t posting_count, const VectorPool& vp) {
    ids = posting_ids;
    count = posting_count;
    dim = vp.dim;
    if (count > vectors.max_size() / dim) throw std::runtime_error("packed scan block too large");
    vectors.resize(count * dim);
    for (size_t i = 0; i < count; ++i)
      std::memcpy(vectors.data() + i * dim, vp.get(ids[i]), dim * sizeof(float));
  }
  const float* get(size_t i) const { return vectors.data() + i * dim; }
  size_t bytes() const { return vectors.size() * sizeof(float); }
};

inline size_t scan_packed_list(const PackedScanList& block, const float* q, uint32_t k,
                               Scratch& s, Result* out, SearchStats& st) {
  s.buf.resize(block.count);
  for (size_t i = 0; i + 8 < block.count; i += 8) prefetch(block.get(i + 8));
  for (size_t i = 0; i < block.count; ++i)
    s.buf[i] = Result{l2sq(q, block.get(i), block.dim), block.ids[i]};
  st.ndc_search += block.count;
  const size_t m = std::min<size_t>(k, block.count);
  std::partial_sort(s.buf.begin(), s.buf.begin() + static_cast<std::ptrdiff_t>(m), s.buf.end());
  std::copy(s.buf.begin(), s.buf.begin() + static_cast<std::ptrdiff_t>(m), out);
  return m;
}

inline size_t merge_topk(std::vector<Result>& cand, uint32_t k, Result* out) {
  std::sort(cand.begin(), cand.end());
  size_t m = 0;
  for (size_t i = 0; i < cand.size() && m < k; ++i) {
    bool duplicate = false;
    for (size_t j = 0; j < m; ++j) {
      if (out[j].gid == cand[i].gid) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) continue;
    out[m++] = cand[i];
  }
  return m;
}

inline float dot(const float* a, const float* b, uint32_t d) {
#if defined(__AVX2__)
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + 16 <= d; i += 16) {
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8)));
  }
  for (; i + 8 <= d; i += 8)
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
  acc0 = _mm256_add_ps(acc0, acc1);
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  float r = _mm_cvtss_f32(s);
  for (; i < d; ++i) r += a[i] * b[i];
  return r;
#else
  float r = 0.f;
  for (uint32_t i = 0; i < d; ++i) r += a[i] * b[i];
  return r;
#endif
}

inline std::vector<float> compute_norms(const VectorPool& vp) {
  std::vector<float> nr(vp.n);
  for (size_t i = 0; i < vp.n; ++i) {
    const float* x = vp.get(static_cast<uint32_t>(i));
    nr[i] = dot(x, x, vp.dim);
  }
  return nr;
}

class PackedLabel {
 public:
  PackedLabel() = default;
  PackedLabel(const PackedLabel&) = delete;
  PackedLabel& operator=(const PackedLabel&) = delete;
  ~PackedLabel() { std::free(mem_); }
  void build(const LabelIndex& li, const VectorPool& vp, bool optimize_graph_layout = false) {
    std::free(mem_);
    n_ = li.g.n;
    dim_ = vp.dim;
    width_ = 0;
    for (uint32_t u = 0; u < n_; ++u) width_ = std::max(width_, li.g.off[u + 1] - li.g.off[u]);
    optimized_ = optimize_graph_layout;
    if (optimized_) width_ = li.g.width;
    const size_t data_len = (static_cast<size_t>(dim_) + 1) * 4;
    const size_t neighbor_len = (static_cast<size_t>(width_) + 1) * 4;
    if (optimized_) {
      node_ = data_len + neighbor_len;
      bytes_ = std::max<size_t>(node_ * n_, 1);
      mem_ = static_cast<char*>(std::malloc(bytes_));
    } else {
      const size_t raw = static_cast<size_t>(dim_) * 4 + 4 + neighbor_len;
      node_ = (raw + 31) / 32 * 32;
      bytes_ = std::max<size_t>(node_ * n_, 32);
      mem_ = static_cast<char*>(std::aligned_alloc(32, (bytes_ + 31) / 32 * 32));
    }
    if (!mem_) throw std::runtime_error("cannot allocate packed label " + li.name);
    for (uint32_t u = 0; u < n_; ++u) {
      char* r = mem_ + static_cast<size_t>(u) * node_;
      const float* x = vp.get(li.l2g[u]);
      const float nrm = dot(x, x, dim_);
      uint32_t deg = 0;
      const uint32_t* nb = li.g.nbrs(u, deg);
      if (optimized_) {
        if (deg > width_) throw std::runtime_error("graph degree exceeds header width for " + li.name);
        std::memcpy(r, &nrm, 4);
        std::memcpy(r + 4, x, dim_ * 4);
        std::memcpy(r + data_len, &deg, 4);
        std::memcpy(r + data_len + 4, nb, static_cast<size_t>(deg) * 4);
      } else {
        std::memcpy(r, x, dim_ * 4);
        std::memcpy(r + dim_ * 4, &nrm, 4);
        std::memcpy(r + dim_ * 4 + 4, &deg, 4);
        std::memcpy(r + dim_ * 4 + 8, nb, static_cast<size_t>(deg) * 4);
      }
    }
  }
  inline const float* vec_current(uint32_t u) const {
    return reinterpret_cast<const float*>(mem_ + static_cast<size_t>(u) * node_);
  }
  inline float norm_current(uint32_t u) const {
    return *reinterpret_cast<const float*>(mem_ + static_cast<size_t>(u) * node_ + dim_ * 4);
  }
  inline const uint32_t* nbrs_current(uint32_t u, uint32_t& deg) const {
    const char* r = mem_ + static_cast<size_t>(u) * node_ + dim_ * 4 + 4;
    std::memcpy(&deg, r, 4);
    return reinterpret_cast<const uint32_t*>(r + 4);
  }
  inline const float* vec_optimized(uint32_t u) const {
    return reinterpret_cast<const float*>(mem_ + static_cast<size_t>(u) * node_ + 4);
  }
  inline float norm_optimized(uint32_t u) const {
    return *reinterpret_cast<const float*>(mem_ + static_cast<size_t>(u) * node_);
  }
  inline const uint32_t* nbrs_optimized(uint32_t u, uint32_t& deg) const {
    const char* r = mem_ + static_cast<size_t>(u) * node_ + data_len();
    std::memcpy(&deg, r, 4);
    return reinterpret_cast<const uint32_t*>(r + 4);
  }
  inline const void* record(uint32_t u) const { return mem_ + static_cast<size_t>(u) * node_; }
  uint32_t n() const { return n_; }
  size_t node_size() const { return node_; }
  size_t data_len() const { return (static_cast<size_t>(dim_) + 1) * 4; }
  size_t bytes() const { return bytes_; }

 private:
  char* mem_ = nullptr;
  size_t node_ = 0, bytes_ = 0;
  uint32_t n_ = 0, dim_ = 0, width_ = 0;
  bool optimized_ = false;
};

struct PoolAccess {
  const LabelIndex& li;
  const VectorPool& vp;
  const float* norms;
  uint32_t n() const { return li.g.n; }
  inline const uint32_t* nbrs(uint32_t u, uint32_t& deg) const { return li.g.nbrs(u, deg); }
  inline void prefetch_vertex(uint32_t u) const { prefetch(vp.get(li.l2g[u])); }
  inline float dist(const float* q, uint32_t u) const {
    const uint32_t g = li.l2g[u];
    return norms[g] - 2.f * dot(q, vp.get(g), vp.dim);
  }
  inline uint32_t gid(uint32_t u) const { return li.l2g[u]; }
};
struct PackedAccess {
  const LabelIndex& li;
  const PackedLabel& pk;
  uint32_t dim;
  uint32_t n() const { return pk.n(); }
  inline const uint32_t* nbrs(uint32_t u, uint32_t& deg) const { return pk.nbrs_current(u, deg); }
  inline void prefetch_vertex(uint32_t u) const {
    const char* r = static_cast<const char*>(pk.record(u));
    prefetch(r);
    prefetch(r + 64);
  }
  inline float dist(const float* q, uint32_t u) const {
    return pk.norm_current(u) - 2.f * dot(q, pk.vec_current(u), dim);
  }
  inline uint32_t gid(uint32_t u) const { return li.l2g[u]; }
};
struct PackedOptimizedAccess {
  const LabelIndex& li;
  const PackedLabel& pk;
  uint32_t dim;
  uint32_t n() const { return pk.n(); }
  inline const uint32_t* nbrs(uint32_t u, uint32_t& deg) const { return pk.nbrs_optimized(u, deg); }
  inline void prefetch_vertex(uint32_t u) const { prefetch(pk.record(u)); }
  inline void prefetch_next(uint32_t u) const {
    const char* r = static_cast<const char*>(pk.record(u));
    prefetch(r);
    prefetch(r + pk.data_len());
  }
  inline void prefetch_neighbor(uint32_t u) const { prefetch_t1(pk.record(u)); }
  inline float dist(const float* q, uint32_t u) const {
    return pk.norm_optimized(u) - 2.f * dot(q, pk.vec_optimized(u), dim);
  }
  inline uint32_t gid(uint32_t u) const { return li.l2g[u]; }
};

template <bool Bitset = false, bool OptimizePrefetch = false, class Acc>
inline size_t search_label(const Acc& a, const float* q, float qnorm, uint32_t L, uint32_t k, uint32_t ep,
                           float ep_l2, InitMode mode, Scratch& s, Result* out, SearchStats& st) {
  const uint32_t n = a.n();
  if (n == 0 || k == 0) return 0;
  L = std::min(std::max(L, k), n);
  s.begin<Bitset>(n, L);
  Neighbor* pool = s.pool.data();
  uint32_t size = 0, deg = 0;
  const uint32_t* nb = a.nbrs(ep, deg);
  if (mode == InitMode::AllEntries) {

    for (const Neighbor& entry : s.entry_candidates) s.seen<Bitset>(entry.id);
    std::stable_sort(s.entry_candidates.begin(), s.entry_candidates.end());
    size = std::min<uint32_t>(L, static_cast<uint32_t>(s.entry_candidates.size()));
    std::copy_n(s.entry_candidates.begin(), size, pool);
    st.initial_candidates = size;
  } else if (mode == InitMode::NsgCompat) {
    s.seen<Bitset>(ep);
    s.init.push_back(ep);
    for (uint32_t i = 0; i < deg && s.init.size() < L; ++i)
      if (nb[i] < n && !s.seen<Bitset>(nb[i])) s.init.push_back(nb[i]);
    while (s.init.size() < L) {
      const uint32_t id = static_cast<uint32_t>(std::rand()) % n;
      if (s.seen<Bitset>(id)) continue;
      s.init.push_back(id);
    }
    for (uint32_t id : s.init) a.prefetch_vertex(id);
    for (uint32_t id : s.init) pool[size++] = Neighbor{id, a.dist(q, id), true};
    st.ndc_search += s.init.size();
    std::sort(pool, pool + size);
  } else {
    s.seen<Bitset>(ep);
    float d0;
    if (ep_l2 >= 0.f) {
      d0 = ep_l2 - qnorm;
    } else {
      d0 = a.dist(q, ep);
      st.ndc_search++;
    }
    pool[size++] = Neighbor{ep, d0, true};
  }
  uint32_t kk = 0;
  while (kk < size) {
    uint32_t nk = std::numeric_limits<uint32_t>::max();
    if (pool[kk].flag) {
      pool[kk].flag = false;
      if constexpr (OptimizePrefetch) {
        if (kk + 1 < size && pool[kk + 1].flag) a.prefetch_next(pool[kk + 1].id);
      }
      uint32_t du = 0;
      const uint32_t* nu = a.nbrs(pool[kk].id, du);
      if constexpr (!OptimizePrefetch)
        for (uint32_t m = 0; m < du; ++m) a.prefetch_vertex(nu[m]);
      for (uint32_t m = 0; m < du; ++m) {
        if constexpr (OptimizePrefetch) {
          constexpr uint32_t kPrefetchDistance = 3;
          if (m + kPrefetchDistance < du) a.prefetch_neighbor(nu[m + kPrefetchDistance]);
        }
        const uint32_t v = nu[m];
        if (s.seen<Bitset>(v)) continue;
        const float d = a.dist(q, v);
        st.ndc_search++;
        if (size == L && d >= pool[size - 1].dist) continue;
        const uint32_t r = insert_pool(pool, size, L, Neighbor{v, d, true});
        if (r < nk) nk = r;
      }
    }
    if (nk <= kk)
      kk = nk;
    else
      ++kk;
  }
  const size_t m = std::min<size_t>(k, size);
  for (size_t i = 0; i < m; ++i) out[i] = Result{pool[i].dist + qnorm, a.gid(pool[i].id)};
  s.end<Bitset>();
  return m;
}

}
