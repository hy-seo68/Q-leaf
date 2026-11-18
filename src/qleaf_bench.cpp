// Benchmark filtered search and report recall, latency, distances, and memory.
#include <omp.h>

#include <chrono>
#include <cerrno>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>

#include "qleaf_gvp.h"

using namespace qleaf;
using Clock = std::chrono::steady_clock;

// Search inputs, output destinations, and CLI defaults.
struct Args {
  std::string base, config, postings, queries, labels, gt, meta, out, per_query_out, result_out;
  std::string packed_scan_dump;
  std::string prefix = "t", init = "nsg", visited = "bitset", packed_record = "current";
  std::string label_format = "legacy", pool_mode = "hugepage", layout = "auto";
  std::vector<uint32_t> Ls{10, 20, 40, 80, 160, 320};
  uint32_t k = 10, warmup = 200, debug_init = 0;
  int threads = 1, runs = 1;
  bool overwrite = false;
};

static std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  std::istringstream ss(s);
  while (std::getline(ss, cur, sep)) {
    const size_t a = cur.find_first_not_of(" \t\r\n"), b = cur.find_last_not_of(" \t\r\n");
    out.push_back(a == std::string::npos ? "" : cur.substr(a, b - a + 1));
  }
  return out;
}

static Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if (k == "--base") a.base = next();
    else if (k == "--config") a.config = next();
    else if (k == "--postings") a.postings = next();
    else if (k == "--queries") a.queries = next();
    else if (k == "--labels") a.labels = next();
    else if (k == "--label-format") a.label_format = next();
    else if (k == "--gt") a.gt = next();
    else if (k == "--meta") a.meta = next();
    else if (k == "--out") a.out = next();
    else if (k == "--per-query-out") a.per_query_out = next();
    else if (k == "--result-out") a.result_out = next();
    else if (k == "--packed-scan-dump") a.packed_scan_dump = next();
    else if (k == "--prefix") a.prefix = next();
    else if (k == "--init") a.init = next();
    else if (k == "--pool") a.pool_mode = next();
    else if (k == "--layout") a.layout = next();
    else if (k == "--visited") a.visited = next();
    else if (k == "--packed-record") a.packed_record = next();
    else if (k == "--overwrite") a.overwrite = true;
    else if (k == "--k") a.k = std::stoul(next());
    else if (k == "--threads") a.threads = std::stoi(next());
    else if (k == "--runs") a.runs = std::stoi(next());
    else if (k == "--warmup") a.warmup = std::stoul(next());
    else if (k == "--debug-init") a.debug_init = std::stoul(next());
    else if (k == "--L") {
      a.Ls.clear();
      for (auto& t : split(next(), ',')) a.Ls.push_back(std::stoul(t));
    } else throw std::runtime_error("unknown argument " + k);
  }
  if (a.base.empty() || a.queries.empty() || a.labels.empty() || a.gt.empty())
    throw std::runtime_error("required: --base --queries --labels --gt (and --config and/or --postings)");
  if (a.init != "nsg" && a.init != "entry" && a.init != "all")
    throw std::runtime_error("--init must be nsg, entry or all");
  if (a.label_format != "legacy" && a.label_format != "plain") throw std::runtime_error("--label-format legacy|plain");
  if (a.pool_mode != "hugepage" && a.pool_mode != "mmap") throw std::runtime_error("--pool hugepage|mmap");
  if (a.layout != "auto" && a.layout != "packed" && a.layout != "pool") throw std::runtime_error("--layout auto|packed|pool");
  if (a.visited != "epoch" && a.visited != "bitset") throw std::runtime_error("--visited epoch|bitset");
  if (a.packed_record != "current" && a.packed_record != "optimized")
    throw std::runtime_error("--packed-record current|optimized");
  const std::vector<std::pair<const char*, const std::string*>> outputs = {
      {"--out", &a.out}, {"--per-query-out", &a.per_query_out}, {"--result-out", &a.result_out},
      {"--packed-scan-dump", &a.packed_scan_dump}};
  for (size_t i = 0; i < outputs.size(); ++i)
    for (size_t j = i + 1; j < outputs.size(); ++j)
      if (!outputs[i].second->empty() && *outputs[i].second == *outputs[j].second)
        throw std::runtime_error(std::string(outputs[i].first) + " and " + outputs[j].first +
                                 " must be different files");
  if (a.runs < 1) a.runs = 1;
  return a;
}

static void require_output_absent(const std::string& path, bool overwrite) {
  if (path.empty() || overwrite) return;
  struct stat st {};
  if (::stat(path.c_str(), &st) == 0)
    throw std::runtime_error("output already exists (use --overwrite to replace): " + path);
  if (errno != ENOENT) throw std::runtime_error("cannot check output path " + path + ": " + std::strerror(errno));
}

static FILE* open_output(const std::string& path, bool overwrite) {
  if (path.empty()) return nullptr;
  const int flags = O_WRONLY | O_CREAT | (overwrite ? O_TRUNC : O_EXCL);
  const int fd = ::open(path.c_str(), flags, 0644);
  if (fd < 0) throw std::runtime_error("cannot create output " + path + ": " + std::strerror(errno));
  FILE* f = ::fdopen(fd, "w");
  if (!f) {
    const std::string message = "fdopen failed for " + path + ": " + std::strerror(errno);
    ::close(fd);
    throw std::runtime_error(message);
  }
  return f;
}

template <typename T>
static std::vector<T> read_vecs(const std::string& path, uint32_t& dim) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  const size_t bytes = static_cast<size_t>(in.tellg());
  in.seekg(0);
  int32_t d = 0;
  in.read(reinterpret_cast<char*>(&d), 4);
  in.seekg(0);
  const size_t row = 4 * (static_cast<size_t>(d) + 1);
  if (d <= 0 || bytes % row) throw std::runtime_error("malformed vecs file " + path);
  const size_t n = bytes / row;
  std::vector<T> raw(n * (d + 1));
  in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(bytes));
  std::vector<T> out(n * d);
  for (size_t i = 0; i < n; ++i) std::memcpy(&out[i * d], &raw[i * (d + 1) + 1], d * sizeof(T));
  dim = static_cast<uint32_t>(d);
  return out;
}

static long proc_status_kb(const char* key) {
  std::ifstream f("/proc/self/status");
  std::string line;
  const size_t n = std::strlen(key);
  while (std::getline(f, line))
    if (line.compare(0, n, key) == 0) return std::atol(line.c_str() + n);
  return -1;
}

static inline uint32_t li_dim(const LabelIndex&, const VectorPool& vp) { return vp.dim; }

static double percentile_sorted(const std::vector<double>& sorted, double p) {
  if (sorted.empty()) return 0.0;
  const double rank = p * static_cast<double>(sorted.size() - 1);
  const size_t lo = static_cast<size_t>(rank);
  const size_t hi = std::min(lo + 1, sorted.size() - 1);
  const double frac = rank - static_cast<double>(lo);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
}

struct Term {
  int graph = -1;
  uint64_t tag = 0;
};

int main(int argc, char** argv) {
  try {
    Args a = parse(argc, argv);

    require_output_absent(a.out, a.overwrite);
    require_output_absent(a.per_query_out, a.overwrite);
    require_output_absent(a.result_out, a.overwrite);
    require_output_absent(a.packed_scan_dump, a.overwrite);
    omp_set_num_threads(a.threads);

    MappedFile base_file(a.base);
    base_file.advise(MADV_WILLNEED);
    VectorPool file_pool = pool_from_fvecs(base_file);
    VectorPool pool = file_pool;
    std::printf("global pool file: n=%zu dim=%u, %.2f GB\n", pool.n, pool.dim, base_file.size() / 1e9);

    std::vector<LabelIndex> graphs;
    std::unordered_map<std::string, int> by_name;
    size_t graph_bytes = 0, graph_nodes = 0;
    if (!a.config.empty()) {
      std::ifstream cf(a.config);
      if (!cf) throw std::runtime_error("cannot open config " + a.config);
      std::string line;
      auto t0 = Clock::now();
      while (std::getline(cf, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto p = split(line, ',');
        if (p.size() != 4) throw std::runtime_error("config line needs 4 fields: " + line);
        graphs.emplace_back();
        graphs.back().load(p[0], p[1], p[2], p[3], pool);
        by_name[p[0]] = static_cast<int>(graphs.size()) - 1;
        graph_bytes += graphs.back().bytes();
        graph_nodes += graphs.back().g.n;
      }
      std::printf("label subgraphs: %zu, total vertices %zu (%.2fx n), metadata in memory %.3f GB, loaded in %.1fs\n",
                  graphs.size(), graph_nodes, double(graph_nodes) / pool.n, graph_bytes / 1e9,
                  std::chrono::duration<double>(Clock::now() - t0).count());
    }
    Postings post;
    if (!a.postings.empty()) post.open(a.postings);

    bool disjoint = true;
    {
      std::vector<uint8_t> seen(pool.n, 0);
      for (const auto& li : graphs)
        for (uint32_t gid : li.l2g) {
          if (seen[gid]) { disjoint = false; break; }
          seen[gid] = 1;
        }
    }
    std::string layout = a.layout;
    if (layout == "auto") layout = (disjoint && a.postings.empty()) ? "packed" : "pool";
    if (layout == "packed" && !disjoint)
      throw std::runtime_error("--layout packed needs disjoint labels");
    if (!a.packed_scan_dump.empty() && layout != "packed")
      throw std::runtime_error("--packed-scan-dump requires --layout packed");
    if (layout != "packed" && a.packed_record != "current")
      throw std::runtime_error("--packed-record optimized requires --layout packed");
    std::vector<std::unique_ptr<PackedLabel>> packed;
    std::vector<PackedScanList> packed_scans;
    OwnedPool owned;
    std::vector<float> norms;
    size_t layout_bytes = 0;
    if (layout == "packed") {
      const uint32_t packed_dim = pool.dim;
      packed.resize(graphs.size());
      for (size_t i = 0; i < graphs.size(); ++i) {
        packed[i].reset(new PackedLabel());
        packed[i]->build(graphs[i], pool, a.packed_record == "optimized");
        layout_bytes += packed[i]->bytes();
      }
      if (post.is_open()) {
        post.validate_offsets();
        std::vector<uint8_t> seen(pool.n, 0);
        for (const auto& li : graphs)
          for (uint32_t gid : li.l2g) seen[gid] = 1;
        packed_scans.resize(post.num_tags());
        size_t scan_labels = 0, scan_vectors = 0, scan_bytes = 0;
        for (uint64_t tag = 0; tag < post.num_tags(); ++tag) {
          if (by_name.count(a.prefix + std::to_string(tag))) continue;
          size_t count;
          const uint32_t* ids = post.list(tag, count);
          for (size_t i = 0; i < count; ++i) {
            const uint32_t gid = ids[i];
            if (gid >= pool.n) throw std::runtime_error("packed scan ID out of range: " + std::to_string(gid));
            if (seen[gid])
              throw std::runtime_error("packed scan ID overlaps graph or scan label: " + std::to_string(gid));
            seen[gid] = 1;
          }
          packed_scans[tag].build(ids, count, pool);
          ++scan_labels;
          scan_vectors += count;
          scan_bytes += packed_scans[tag].bytes();
        }
        layout_bytes += scan_bytes;
        std::printf("packed scan blocks: labels=%zu vectors=%zu vector_bytes=%zu (global IDs in postings mapping)\n",
                    scan_labels, scan_vectors, scan_bytes);
      }
      base_file.close();
      file_pool = VectorPool{};
      pool = VectorPool{};
      pool.dim = packed_dim;
      std::printf("layout: packed (%s records, labels disjoint), records/scans %.3f GB, pool released\n",
                  a.packed_record.c_str(), layout_bytes / 1e9);
      std::printf("memory immediately after packed load/release: VmRSS %.2f GB, VmHWM %.2f GB\n",
                  proc_status_kb("VmRSS:") / 1048576.0, proc_status_kb("VmHWM:") / 1048576.0);
    } else {
      if (a.pool_mode == "hugepage") {
        pool = owned.load(file_pool);
        base_file.close();
        layout_bytes = owned.bytes();
      } else {
        layout_bytes = base_file.size();
      }
      norms = compute_norms(pool);
      std::printf("layout: pool (%s, labels %s), pool %.3f GB + norms %.3f GB\n", a.pool_mode.c_str(),
                  disjoint ? "disjoint" : "overlapping", layout_bytes / 1e9, norms.size() * 4 / 1e9);
    }

    if (!a.packed_scan_dump.empty()) {
      std::unique_ptr<FILE, decltype(&std::fclose)> dump(
          open_output(a.packed_scan_dump, a.overwrite), &std::fclose);
      auto write = [&](const void* data, size_t bytes) {
        if (bytes && std::fwrite(data, 1, bytes, dump.get()) != bytes)
          throw std::runtime_error("cannot write packed scan dump");
      };
      uint64_t blocks = 0;
      for (const auto& block : packed_scans) if (block.ids) ++blocks;
      write(&blocks, sizeof(blocks));
      for (uint64_t tag = 0; tag < packed_scans.size(); ++tag) {
        const auto& block = packed_scans[tag];
        if (!block.ids) continue;
        const uint64_t count = block.count;
        write(&tag, sizeof(tag));
        write(&count, sizeof(count));
        write(&block.dim, sizeof(block.dim));
        write(block.ids, count * sizeof(uint32_t));
        write(block.vectors.data(), block.bytes());
      }
      if (std::fflush(dump.get()) != 0) throw std::runtime_error("cannot flush packed scan dump");
    }

    uint32_t qdim = 0, kgt = 0;
    std::vector<float> Q = read_vecs<float>(a.queries, qdim);
    std::vector<int32_t> GT = read_vecs<int32_t>(a.gt, kgt);
    if (qdim != pool.dim) throw std::runtime_error("query dim != base dim");
    const size_t nq = Q.size() / qdim;
    if (GT.size() / kgt != nq) throw std::runtime_error("GT rows != queries");
    if (kgt < a.k) throw std::runtime_error("GT has fewer than k neighbours");

    std::vector<std::vector<Term>> terms(nq);
    {
      std::ifstream lf(a.labels);
      if (!lf) throw std::runtime_error("cannot open labels " + a.labels);
      std::string line;
      if (a.label_format == "legacy") std::getline(lf, line);
      size_t i = 0;
      while (i < nq && std::getline(lf, line)) {
        if (a.label_format == "legacy") {
          if (line.empty()) continue;
          if (!line.empty() && line.back() == '\r') line.pop_back();
        }
        for (auto& raw : split(line, '|')) {
          std::string name = raw;
          if (a.label_format == "legacy") std::replace(name.begin(), name.end(), ' ', '_');
          Term t;
          auto it = by_name.find(name);
          if (it != by_name.end()) {
            t.graph = it->second;
          } else {
            if (!post.is_open()) throw std::runtime_error("label " + name + " has no subgraph and no --postings");
            if (name.compare(0, a.prefix.size(), a.prefix) != 0)
              throw std::runtime_error("label " + name + " lacks prefix " + a.prefix);
            t.tag = std::stoull(name.substr(a.prefix.size()));
            if (t.tag >= post.num_tags()) throw std::runtime_error("tag out of range: " + name);
          }
          terms[i].push_back(t);
        }
        if (terms[i].empty()) throw std::runtime_error("empty label line " + std::to_string(i));
        ++i;
      }
      if (i != nq) throw std::runtime_error("label file has fewer label lines than queries");
    }

    size_t graph_queries = 0, scan_queries = 0, mixed_queries = 0;
    for (const auto& query_terms : terms) {
      bool graph = false, scan = false;
      for (const auto& term : query_terms) {
        graph |= term.graph >= 0;
        scan |= term.graph < 0;
      }
      if (graph && scan) ++mixed_queries;
      else if (graph) ++graph_queries;
      else ++scan_queries;
    }
    std::printf("query routes: graph=%zu scan=%zu mixed=%zu total=%zu (per pass; excludes warmup)\n",
                graph_queries, scan_queries, mixed_queries, nq);

    std::vector<std::pair<std::string, std::vector<std::string>>> groups;
    if (!a.meta.empty()) {
      std::ifstream mf(a.meta);
      std::string header;
      std::getline(mf, header);
      auto cols = split(header, '\t');
      std::vector<int> want;
      for (size_t c = 0; c < cols.size(); ++c)
        if (cols[c] == "band" || cols[c] == "path" || cols[c] == "stratum") {
          want.push_back(static_cast<int>(c));
          groups.push_back({cols[c], std::vector<std::string>(nq)});
        }
      std::string line;
      for (size_t i = 0; i < nq && std::getline(mf, line); ++i) {
        auto f = split(line, '\t');
        for (size_t g = 0; g < want.size(); ++g) groups[g].second[i] = f.at(want[g]);
      }
    }

    std::printf("memory after load: VmRSS %.2f GB, VmHWM %.2f GB\n", proc_status_kb("VmRSS:") / 1048576.0,
                proc_status_kb("VmHWM:") / 1048576.0);
    std::printf("memory after load (KiB): VmRSS=%ld VmHWM=%ld\n",
                proc_status_kb("VmRSS:"), proc_status_kb("VmHWM:"));
    std::printf("visited: %s\n", a.visited.c_str());
    std::printf("packed record: %s\n", a.packed_record.c_str());

    const InitMode mode = (a.init == "all") ? InitMode::AllEntries :
                          (a.init == "nsg") ? InitMode::NsgCompat : InitMode::EntryOnly;
    std::vector<Scratch> scratch;
    for (int t = 0; t < a.threads; ++t) scratch.emplace_back(1234 + t);

    std::vector<Result> res(nq * a.k);
    std::vector<uint32_t> nres(nq);
    std::vector<SearchStats> qst(nq);
    std::vector<double> lat(nq, 0.0);
    std::vector<double> entry_lat(nq, 0.0);

    auto search_term = [&](const Term& t, const float* q, float qn, uint32_t L, Scratch& s, Result* dst,
                           SearchStats& st, Clock::time_point* entry_end) -> size_t {
      if (t.graph >= 0) {
        const LabelIndex& li = graphs[t.graph];
        float ed;
        uint32_t ep;
        if (mode == InitMode::AllEntries) {
          ep = li.eps[0];
          ed = -1.f;
          select_all_entries(li, pool, q, qn, s, st);
        } else {
          ep = select_entry(li, pool, q, ed, st);
        }
        if (entry_end) *entry_end = Clock::now();
        if (!packed.empty()) {
          if (a.packed_record == "optimized") {
            PackedOptimizedAccess acc{li, *packed[t.graph], li_dim(li, pool)};
            if (a.visited == "bitset")
              return search_label<true, true>(acc, q, qn, L, a.k, ep, ed, mode, s, dst, st);
            return search_label<false, true>(acc, q, qn, L, a.k, ep, ed, mode, s, dst, st);
          } else {
            PackedAccess acc{li, *packed[t.graph], li_dim(li, pool)};
            if (a.visited == "bitset")
              return search_label<true, false>(acc, q, qn, L, a.k, ep, ed, mode, s, dst, st);
            return search_label<false, false>(acc, q, qn, L, a.k, ep, ed, mode, s, dst, st);
          }
        }
        PoolAccess acc{li, pool, norms.data()};
        if (a.visited == "bitset")
          return search_label<true>(acc, q, qn, L, a.k, ep, ed, mode, s, dst, st);
        return search_label<false>(acc, q, qn, L, a.k, ep, ed, mode, s, dst, st);
      }
      size_t cnt;
      const uint32_t* ids = post.list(t.tag, cnt);
      if (entry_end) *entry_end = Clock::now();
      if (layout == "packed") return scan_packed_list(packed_scans[t.tag], q, a.k, s, dst, st);
      return scan_list(ids, cnt, pool, q, a.k, s, dst, st);
    };

    // Per-query time includes setup and entry selection; output I/O is excluded.
    auto run_one = [&](size_t i, uint32_t L, Scratch& s) {
      const auto t0 = Clock::now();
      const float* q = &Q[i * qdim];
      const float qn = dot(q, q, qdim);
      SearchStats st;
      Result* out = &res[i * a.k];
      const auto& ts = terms[i];
      Clock::time_point entry_end;
      if (ts.size() == 1) {
        nres[i] = static_cast<uint32_t>(search_term(ts[0], q, qn, L, s, out, st, &entry_end));
      } else {
        thread_local std::vector<Result> cand, tmp;
        cand.clear();
        tmp.resize(a.k);
        bool first_term = true;
        for (const Term& t : ts) {
          const size_t got = search_term(t, q, qn, L, s, tmp.data(), st,
                                        first_term ? &entry_end : nullptr);
          first_term = false;
          cand.insert(cand.end(), tmp.begin(), tmp.begin() + static_cast<std::ptrdiff_t>(got));
        }
        nres[i] = static_cast<uint32_t>(merge_topk(cand, a.k, out));
      }
      qst[i] = st;
      lat[i] = std::chrono::duration<double>(Clock::now() - t0).count();

      entry_lat[i] = ts.size() != 1 ? std::numeric_limits<double>::quiet_NaN() :
                     st.ndc_entry == 0 ? 0.0 : std::chrono::duration<double>(entry_end - t0).count();
    };

    // Warm up once with the first search width, before all measured runs.
    for (size_t i = 0; i < std::min<size_t>(a.warmup, nq); ++i) run_one(i, a.Ls.front(), scratch[0]);

    FILE* fo = open_output(a.out, a.overwrite);
    FILE* fpq = open_output(a.per_query_out, a.overwrite);
    FILE* fr = open_output(a.result_out, a.overwrite);
    if (fo)
      std::fprintf(fo, "init\tlayout\tvisited\tpacked_record\trun\tL\tgroup\tvalue\tqueries\trecall\tqps\tndc_entry\tndc_search\t"
                       "ndc_search_p50\tndc_search_p99\tlatency_us_p50\tlatency_us_p99\t"
                       "mean_t_entry_us\tmean_t_query_us\tep_time_share\tns_per_dist_entry\tns_per_dist_search\n");
    if (fpq)
      std::fprintf(fpq, "run\tL\tqid\trecall_i\tndc_entry\tndc_search\tlatency_us\tt_entry_us\n");
    if (fr) std::fprintf(fr, "run\tL\tqid\trank\tglobal_id\n");
    std::printf("\n%-4s %-6s %-8s %-10s %8s %10s %12s %10s %10s %10s %10s\n", "run", "L", "group",
                "value", "queries", "recall", "QPS", "ndc_ep", "ndc_avg", "ndc_p50", "ndc_p99");

    for (uint32_t L : a.Ls) {
      for (int run = 0; run < a.runs; ++run) {
        // All-query QPS uses this parallel loop wall time.
        auto t0 = Clock::now();
#pragma omp parallel for schedule(dynamic, 32)
        for (size_t i = 0; i < nq; ++i) run_one(i, L, scratch[omp_get_thread_num()]);
        const double secs = std::chrono::duration<double>(Clock::now() - t0).count();

        std::vector<double> rec(nq);
        for (size_t i = 0; i < nq; ++i) {
          std::set<int32_t> g(&GT[i * kgt], &GT[i * kgt] + a.k);
          uint32_t hit = 0;
          for (uint32_t r = 0; r < nres[i]; ++r) hit += g.count(static_cast<int32_t>(res[i * a.k + r].gid));
          rec[i] = double(hit) / a.k;
        }
        if (fpq) {
          for (size_t i = 0; i < nq; ++i)
            std::fprintf(fpq, "%d\t%u\t%zu\t%.6f\t%llu\t%llu\t%.3f\t%.6f\n", run, L, i, rec[i],
                         static_cast<unsigned long long>(qst[i].ndc_entry),
                         static_cast<unsigned long long>(qst[i].ndc_search), lat[i] * 1e6, entry_lat[i] * 1e6);
        }
        if (fr) {
          for (size_t i = 0; i < nq; ++i)
            for (uint32_t rank = 0; rank < nres[i]; ++rank)
              std::fprintf(fr, "%d\t%u\t%zu\t%u\t%u\n", run, L, i, rank,
                           res[i * a.k + rank].gid);
        }
        auto report = [&](const std::string& gname, const std::string& val, const std::vector<size_t>& ids) {
          double r = 0, ne = 0, ns = 0, lt = 0, et = 0;
          std::vector<double> ndc_values, latency_values;
          ndc_values.reserve(ids.size());
          latency_values.reserve(ids.size());
          for (size_t i : ids) {
            r += rec[i];
            ne += qst[i].ndc_entry;
            ns += qst[i].ndc_search;
            lt += lat[i];
            et += entry_lat[i];
            ndc_values.push_back(static_cast<double>(qst[i].ndc_search));
            latency_values.push_back(lat[i] * 1e6);
          }
          std::sort(ndc_values.begin(), ndc_values.end());
          std::sort(latency_values.begin(), latency_values.end());
          const double ndc_p50 = percentile_sorted(ndc_values, 0.50);
          const double ndc_p99 = percentile_sorted(ndc_values, 0.99);
          const double latency_p50 = percentile_sorted(latency_values, 0.50);
          const double latency_p99 = percentile_sorted(latency_values, 0.99);
          const double m = ids.empty() ? 1 : ids.size();

          const double qps = (gname == "all") ? nq / secs : ids.size() / lt;
          const double ep_share = lt > 0 ? et / lt : 0;
          const double ns_entry = ne > 0 ? et * 1e9 / ne : 0;
          const double ns_search = ns > 0 ? (lt - et) * 1e9 / ns : 0;
          std::printf("%-4d %-6u %-8s %-10s %8zu %10.4f %12.1f %10.1f %10.1f %10.1f %10.1f\n", run, L,
                      gname.c_str(), val.c_str(), ids.size(), r / m, qps, ne / m, ns / m, ndc_p50, ndc_p99);
          std::printf("entry_timing run=%d L=%u group=%s value=%s mean_t_entry_us=%.6f "
                      "mean_t_query_us=%.6f ep_time_share=%.9f ns_per_dist_entry=%.6f ns_per_dist_search=%.6f\n",
                      run, L, gname.c_str(), val.c_str(), et * 1e6 / m, lt * 1e6 / m,
                      ep_share, ns_entry, ns_search);
          if (fo)
            std::fprintf(fo, "%s\t%s\t%s\t%s\t%d\t%u\t%s\t%s\t%zu\t%.6f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t"
                             "%.6f\t%.6f\t%.9f\t%.6f\t%.6f\n",
                         a.init.c_str(), layout.c_str(), a.visited.c_str(), a.packed_record.c_str(), run, L,
                         gname.c_str(), val.c_str(), ids.size(), r / m,
                         qps, ne / m, ns / m, ndc_p50, ndc_p99, latency_p50, latency_p99,
                         et * 1e6 / m, lt * 1e6 / m, ep_share, ns_entry, ns_search);
        };
        std::vector<size_t> all(nq);
        for (size_t i = 0; i < nq; ++i) all[i] = i;
        report("all", "-", all);
        for (auto& [gname, vals] : groups) {
          std::map<std::string, std::vector<size_t>> by;
          for (size_t i = 0; i < nq; ++i) by[vals[i]].push_back(i);
          for (auto& [v, ids] : by) report(gname, v, ids);
        }
        if (mode == InitMode::AllEntries && a.debug_init) {
          const size_t count = std::min<size_t>(a.debug_init, nq);
          for (size_t j = 0; j < count; ++j) {
            const size_t i = j * nq / count;
            std::printf("debug_init run=%d L=%u qid=%zu m_eff=%llu initial_candidates=%u\n", run, L, i,
                         static_cast<unsigned long long>(qst[i].ndc_entry), qst[i].initial_candidates);
          }
        }
      }
    }
    if (fo) std::fclose(fo);
    if (fpq) std::fclose(fpq);
    if (fr) std::fclose(fr);
    std::printf("\nmemory at end: VmRSS %.2f GB, VmHWM %.2f GB\n", proc_status_kb("VmRSS:") / 1048576.0,
                proc_status_kb("VmHWM:") / 1048576.0);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
