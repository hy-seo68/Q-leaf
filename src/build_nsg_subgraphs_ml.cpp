// Build real-tag graphs using size-dependent parameters.
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <efanna2e/index.h>
#include <efanna2e/parameters.h>
#include <efanna2e/util.h>
#include <efanna2e/neighbor.h>
#include <efanna2e/index_random.h>
#include <efanna2e/index_graph.h>

namespace efanna2e {
struct SimpleNeighbor {
  unsigned id;
  float distance;
  SimpleNeighbor() = default;
  SimpleNeighbor(unsigned id, float distance) : id{id}, distance{distance} {}
  inline bool operator<(const SimpleNeighbor& other) const { return distance < other.distance; }
};
struct SimpleNeighbors {
  std::vector<SimpleNeighbor> pool;
};
}

#include <efanna2e/index_nsg.h>

#include "nsg_qleaf_data_loader.h"

struct BuildParams {
  unsigned K, L, iter, S, R;
  unsigned L_nsg, R_nsg, C_nsg;
  const char* tier;
};

static unsigned medium_iter() {
  const char* v = std::getenv("QLEAF_MEDIUM_ITER");
  return v ? static_cast<unsigned>(std::atoi(v)) : 8u;
}
// Size thresholds select the paper construction parameters.
static BuildParams params_for_size(size_t n) {
  if (n < 50000) return {20, 30, 5, 10, 50, 20, 16, 100, "small"};
  if (n < 500000) return {40, 50, medium_iter(), 10, 100, 30, 32, 100, "medium"};
  return {150, 150, 10, 10, 100, 40, 50, 300, "large"};
}

static int env_int(const char* name, int def) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : def;
}

static size_t huge_threshold(size_t fallback, bool& explicit_minimum) {
  const char* v = std::getenv("QLEAF_HUGE_MIN_SIZE");
  explicit_minimum = v != nullptr;
  if (!v) return fallback;
  const std::string text(v);
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error("QLEAF_HUGE_MIN_SIZE must be a nonnegative integer");
  const unsigned long long value = std::stoull(text);
  if (value > std::numeric_limits<size_t>::max())
    throw std::runtime_error("QLEAF_HUGE_MIN_SIZE is too large");
  return static_cast<size_t>(value);
}

static int actual_omp_team_size() {
  int team = 1;
#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp single
    team = omp_get_num_threads();
  }
#endif
  return team;
}

static bool mkdir_p(const std::string& path) {
  std::string cur;
  std::stringstream ss(path);
  std::string part;
  if (!path.empty() && path[0] == '/') cur = "/";
  while (std::getline(ss, part, '/')) {
    if (part.empty()) continue;
    cur += part + "/";
    if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
  }
  return true;
}

static std::vector<std::string> existing_nsg_graphs(const std::string& path) {
  std::vector<std::string> found;
  DIR* dir = opendir(path.c_str());
  if (!dir) {
    if (errno == ENOENT) return found;
    throw std::runtime_error("cannot inspect output directory " + path + ": " + std::strerror(errno));
  }
  while (dirent* entry = readdir(dir)) {
    const std::string name(entry->d_name);
    const std::string suffix = "_nsg.graph";
    if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
      found.push_back(name);
  }
  closedir(dir);
  std::sort(found.begin(), found.end());
  return found;
}

struct Task {
  std::string name;
  size_t size = 0;
  BuildParams p{};
  bool large = false;
  int omp_threads = 1;
  int knn_team = 1, nsg_team = 1;
  bool success = false;
  double extract_s = 0, knn_s = 0, nsg_s = 0;
  std::string error;
};

static void build_one(Task& t, const std::string& idx_dir, const std::string& out_dir,
                      const std::shared_ptr<MmapVectorReader>& base, bool keep_knng) {
  try {
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(t.omp_threads);
#endif
    std::vector<uint32_t> ids = load_label_indices((idx_dir + "/" + t.name + ".idx").c_str());
    if (ids.size() != t.size) throw std::runtime_error("idx size differs from graph_tags.tsv");
    if (ids.size() <= t.p.K) throw std::runtime_error("subgraph smaller than K; raise the export threshold T");

    save_label_indices((out_dir + "/" + t.name + ".idx").c_str(), ids);

    const size_t n = ids.size(), dim = base->dimension();
    auto t0 = std::chrono::steady_clock::now();
    AlignedVector32<float> data(n * dim);
    for (size_t i = 0; i < n; ++i) std::memcpy(data.data() + i * dim, base->get_vector(ids[i]), dim * sizeof(float));
    auto t1 = std::chrono::steady_clock::now();

    const std::string prefix = out_dir + "/" + t.name;
    const std::string knng = prefix + "_nsg.knng", graph = prefix + "_nsg.graph";
    {
      t.knn_team = actual_omp_team_size();
      std::cout << "[" << t.name << "] KNN start: requested_threads=" << t.omp_threads
                << ", omp_team=" << t.knn_team << std::endl;
      efanna2e::IndexRandom init_index(dim, n);
      init_index.Build(n, data.data(), efanna2e::Parameters());
      efanna2e::IndexGraph index(dim, n, efanna2e::L2, (efanna2e::Index*)(&init_index));
      efanna2e::Parameters paras;
      paras.Set<unsigned>("K", t.p.K);
      paras.Set<unsigned>("L", t.p.L);
      paras.Set<unsigned>("iter", t.p.iter);
      paras.Set<unsigned>("S", t.p.S);
      paras.Set<unsigned>("R", t.p.R);
      index.Build(n, data.data(), paras);
      index.Save(knng.c_str());
    }
    auto t2 = std::chrono::steady_clock::now();
    {
      t.nsg_team = actual_omp_team_size();
      std::cout << "[" << t.name << "] NSG start: requested_threads=" << t.omp_threads
                << ", omp_team=" << t.nsg_team << std::endl;
      efanna2e::IndexNSG index(dim, n, efanna2e::L2, nullptr);
      efanna2e::Parameters paras;
      paras.Set<unsigned>("L", t.p.L_nsg);
      paras.Set<unsigned>("R", t.p.R_nsg);
      paras.Set<unsigned>("C", t.p.C_nsg);
      paras.Set<std::string>("nn_graph_path", knng);
      index.Build(n, data.data(), paras);
      index.Save(graph.c_str());
    }
    auto t3 = std::chrono::steady_clock::now();
    if (!keep_knng) std::remove(knng.c_str());

    t.extract_s = std::chrono::duration<double>(t1 - t0).count();
    t.knn_s = std::chrono::duration<double>(t2 - t1).count();
    t.nsg_s = std::chrono::duration<double>(t3 - t2).count();
    std::ofstream info(prefix + "_nsg_build.txt");
    info << "label\t" << t.name << "\nvectors\t" << n << "\ntier\t" << t.p.tier << "\nK,L,iter,S,R\t" << t.p.K << ","
         << t.p.L << "," << t.p.iter << "," << t.p.S << "," << t.p.R << "\nNSG L,R,C\t" << t.p.L_nsg << ","
         << t.p.R_nsg << "," << t.p.C_nsg << "\nextract_s\t" << t.extract_s << "\nknn_s\t" << t.knn_s
         << "\nnsg_s\t" << t.nsg_s << "\nomp_threads\t" << t.omp_threads << "\nknn_team\t" << t.knn_team
         << "\nnsg_team\t" << t.nsg_team << "\n";
    t.success = true;
  } catch (const std::exception& e) {
    t.success = false;
    t.error = e.what();
  }
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0] << " <base_fvecs> <graph_tags.tsv> <idx_dir> <output_dir> <dataset_name>\n"
              << "Env: QLEAF_BUILD_THREADS (8), QLEAF_LARGE_THREADS (8), QLEAF_MAX_LARGE (2), "
                 "QLEAF_KEEP_KNNG (0), QLEAF_OVERWRITE (0), QLEAF_HUGE_MIN_SIZE (unset: total/threads)\n";
    return 1;
  }
  const std::string base_file = argv[1], tags_file = argv[2], idx_dir = argv[3];
  const std::string out_dir = std::string(argv[4]) + "/" + argv[5] + "/auto_tiers";
  const int threads = std::max(1, env_int("QLEAF_BUILD_THREADS", 8));
  const int max_large = std::max(1, env_int("QLEAF_MAX_LARGE", 2));
  const int large_threads = std::max(1, env_int("QLEAF_LARGE_THREADS", 8));
  const bool keep_knng = env_int("QLEAF_KEEP_KNNG", 0) != 0;
  const bool overwrite = env_int("QLEAF_OVERWRITE", 0) == 1;
  try {
    const std::vector<std::string> existing = existing_nsg_graphs(out_dir);
    if (!existing.empty() && !overwrite) {
      std::cerr << "refusing to overwrite " << existing.size() << " existing *_nsg.graph files in " << out_dir
                << " (e.g. " << existing.front() << "); set QLEAF_OVERWRITE=1 to allow replacement\n";
      return 1;
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  if (!mkdir_p(out_dir)) {
    std::cerr << "cannot create " << out_dir << "\n";
    return 1;
  }

  std::vector<Task> tasks;
  {
    std::ifstream in(tags_file);
    if (!in) {
      std::cerr << "cannot open " << tags_file << "\n";
      return 1;
    }
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);
      Task t;
      long long tag = 0;
      if (!(ss >> t.name >> tag >> t.size)) continue;
      t.p = params_for_size(t.size);
      t.large = std::strcmp(t.p.tier, "large") == 0;
      tasks.push_back(t);
    }
  }
  std::stable_sort(tasks.begin(), tasks.end(), [](const Task& a, const Task& b) { return a.size > b.size; });
  size_t total = 0, n_large = 0;
  for (auto& t : tasks) {
    total += t.size;
    n_large += t.large;
  }
  bool explicit_huge_minimum = false;
  size_t huge_cut = 0;
  try {
    huge_cut = huge_threshold(total / static_cast<size_t>(threads), explicit_huge_minimum);
  } catch (const std::exception& e) {
    std::cerr << "invalid QLEAF_HUGE_MIN_SIZE: " << e.what() << "\n";
    return 1;
  }
  std::cout << "subgraphs: " << tasks.size() << " (large tier: " << n_large << "), total memberships: " << total
            << "\nthreads: " << threads << ", max concurrent large builds: " << max_large << "\nhuge threshold: "
            << huge_cut << (explicit_huge_minimum ? " (size >= explicit minimum)" : " (size > total/threads)")
            << ", huge task threads: " << large_threads << "\n";

  auto base = std::make_shared<MmapVectorReader>(base_file.c_str());
  std::cout << "global pool: " << base->num_vectors() << " vectors, dim " << base->dimension() << "\n";

  std::mutex mu;
  std::condition_variable cv;
  std::vector<char> taken(tasks.size(), 0);
  size_t remaining = tasks.size(), finished = 0;
  int large_running = 0;
  auto pick = [&]() -> long {
    std::unique_lock<std::mutex> lk(mu);
    for (;;) {
      if (remaining == 0) return -1;
      for (size_t i = 0; i < tasks.size(); ++i) {
        if (taken[i] || (tasks[i].large && large_running >= max_large)) continue;
        taken[i] = 1;
        --remaining;
        if (tasks[i].large) ++large_running;
        return static_cast<long>(i);
      }
      cv.wait(lk);
    }
  };
  auto done = [&](size_t i) {
    std::lock_guard<std::mutex> lk(mu);
    if (tasks[i].large) --large_running;
    ++finished;
    if (finished % 100 == 0 || finished == tasks.size())
      std::cout << "  finished " << finished << " / " << tasks.size() << std::endl;
    cv.notify_all();
  };

  auto t0 = std::chrono::steady_clock::now();
  if (large_threads > 1 || explicit_huge_minimum) {

    for (size_t i = 0; i < tasks.size(); ++i) {
      const bool huge = explicit_huge_minimum ? tasks[i].size >= huge_cut : tasks[i].size > huge_cut;
      if (!huge) continue;
      {
        std::lock_guard<std::mutex> lk(mu);
        taken[i] = 1;
        --remaining;
      }
      tasks[i].omp_threads = large_threads;
      build_one(tasks[i], idx_dir, out_dir, base, keep_knng);
      std::cout << "  huge " << tasks[i].name << " (" << tasks[i].size << "): "
                << (tasks[i].success ? "ok" : tasks[i].error) << ", "
                << tasks[i].extract_s + tasks[i].knn_s + tasks[i].nsg_s << " s" << std::endl;
      {
        std::lock_guard<std::mutex> lk(mu);
        ++finished;
      }
    }
  }

  std::vector<std::thread> pool;
  for (int w = 0; w < threads; ++w)
    pool.emplace_back([&]() {
      for (long i; (i = pick()) >= 0;) {
        build_one(tasks[i], idx_dir, out_dir, base, keep_knng);
        done(static_cast<size_t>(i));
      }
    });
  for (auto& th : pool) th.join();
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
  const double cpu = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1e6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1e6;

  std::ofstream rep(out_dir + "/build_tasks.tsv");
  rep << "name\tsize\ttier\tok\textract_s\tknn_s\tnsg_s\terror\tomp_threads\tknn_team\tnsg_team\n";
  size_t ok = 0;
  double task_sum = 0;
  for (auto& t : tasks) {
    rep << t.name << "\t" << t.size << "\t" << t.p.tier << "\t" << t.success << "\t" << t.extract_s << "\t" << t.knn_s
        << "\t" << t.nsg_s << "\t" << t.error << "\t" << t.omp_threads << "\t" << t.knn_team << "\t"
        << t.nsg_team << "\n";
    ok += t.success;
    task_sum += t.extract_s + t.knn_s + t.nsg_s;
  }
  rep << "#wall_s\t" << wall << "\n#process_cpu_s\t" << cpu << "\n#sum_task_s\t" << task_sum
      << "\n#peak_rss_kb\t" << ru.ru_maxrss << "\n";
  std::cout << "built " << ok << " / " << tasks.size() << " subgraphs; wall " << wall << " s, process CPU " << cpu
            << " s, sum of task times " << task_sum << " s, peak RSS " << ru.ru_maxrss / 1024.0 << " MB\n";
  return ok == tasks.size() ? 0 : 2;
}
