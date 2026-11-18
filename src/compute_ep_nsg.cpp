// Generate entry points for NSG subgraphs.
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <memory>
#include <map>
#include <set>
#include <chrono>
#include <dirent.h>
#include <sys/stat.h>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include "nsg_qleaf_data_loader.h"

#include <functional>
#include <random>

enum class EpPolicy { QLeaf, KMeans, OutDeg, RandomM, Random1, NsgEp };
static EpPolicy g_policy = EpPolicy::QLeaf;
static uint64_t g_seed = 42;

static const char* policy_name(EpPolicy p) {
    switch (p) {
        case EpPolicy::QLeaf: return "qleaf";
        case EpPolicy::KMeans: return "kmeans";
        case EpPolicy::OutDeg: return "outdeg";
        case EpPolicy::RandomM: return "random";
        case EpPolicy::Random1: return "random1";
        case EpPolicy::NsgEp: return "nsgep";
    }
    return "?";
}
static const char* policy_description(EpPolicy p) {
    switch (p) {
        case EpPolicy::QLeaf: return "qleaf: K-Means seeds + 1-hop max out-degree";
        case EpPolicy::KMeans: return "kmeans: K-Means seeds (vector nearest to each centroid)";
        case EpPolicy::OutDeg: return "outdeg: top-m out-degree vertices";
        case EpPolicy::RandomM: return "random: m random vertices";
        case EpPolicy::Random1: return "random1: one random vertex";
        case EpPolicy::NsgEp: return "nsgep: NSG navigating node (approximate medoid)";
    }
    return "?";
}
static bool parse_policy(const std::string& s, EpPolicy& p) {
    if (s == "qleaf") p = EpPolicy::QLeaf;
    else if (s == "kmeans") p = EpPolicy::KMeans;
    else if (s == "outdeg") p = EpPolicy::OutDeg;
    else if (s == "random") p = EpPolicy::RandomM;
    else if (s == "random1") p = EpPolicy::Random1;
    else if (s == "nsgep") p = EpPolicy::NsgEp;
    else return false;
    return true;
}

static int entry_count_for_size(size_t n) {
    if (n < 50000) return 16;
    if (n < 500000) return 32;
    return 64;
}

template <typename T>
static void deduplicate_preserve_order(std::vector<T>& values) {
    std::set<T> seen;
    std::vector<T> unique;
    unique.reserve(values.size());
    for (const T& value : values) {
        if (seen.insert(value).second) {
            unique.push_back(value);
        }
    }
    values.swap(unique);
}

static int64_t read_nsg_navigating_node(const std::string& graph_file, size_t num_vectors) {
    std::ifstream in(graph_file, std::ios::binary);
    uint32_t header[2] = {0, 0};
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!in) throw std::runtime_error("cannot read NSG header: " + graph_file);
    if (header[1] >= num_vectors) throw std::runtime_error("NSG navigating node out of range: " + graph_file);
    return static_cast<int64_t>(header[1]);
}

long get_vmhwm_kb() {
    std::ifstream status_file("/proc/self/status");
    if (!status_file.is_open()) {
        return -1;
    }

    std::string line;
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmHWM:") {
            std::istringstream iss(line.substr(6));
            long vmhwm_kb;
            iss >> vmhwm_kb;
            return vmhwm_kb;
        }
    }
    return -1;
}

struct EntryPointTask {
    std::string label_name;
    std::string graph_file;
    std::string idx_file;
    std::string output_file;

    bool success = false;
    size_t num_vectors = 0;
    size_t num_entry_points = 0;
    double load_time = 0.0;
    double kmeans_time = 0.0;
    double degree_time = 0.0;
    double total_time = 0.0;
    std::string error_message;
};

float l2_squared_distance(const float* v1, const float* v2, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sum;
}

// Pick full-data representatives nearest to trained FAISS centroids.
std::vector<int64_t> find_cluster_medoids(const AlignedVector32<float>& data,
                                           size_t num_vectors,
                                           size_t dim,
                                           int num_clusters,
                                           double& kmeans_time) {
    auto start = std::chrono::high_resolution_clock::now();

    faiss::Clustering clustering(dim, num_clusters);
    clustering.verbose = false;
    clustering.niter = 25;
    clustering.seed = 42;

    faiss::IndexFlatL2 index(dim);
    clustering.train(num_vectors, data.data(), index);

    faiss::IndexFlatL2 centroid_index(dim);
    centroid_index.add(num_clusters, clustering.centroids.data());

    std::vector<faiss::idx_t> assignments(num_vectors);
    std::vector<float> distances(num_vectors);
    centroid_index.search(num_vectors, data.data(), 1, distances.data(), assignments.data());

    std::vector<int64_t> medoids(num_clusters, -1);
    std::vector<float> min_dist_in_cluster(num_clusters, std::numeric_limits<float>::max());

    for (size_t i = 0; i < num_vectors; i++) {
        int cluster_id = static_cast<int>(assignments[i]);
        float dist = distances[i];

        if (dist < min_dist_in_cluster[cluster_id]) {
            min_dist_in_cluster[cluster_id] = dist;
            medoids[cluster_id] = i;
        }
    }

    std::vector<int64_t> valid_medoids;
    for (int64_t medoid : medoids) {
        if (medoid != -1) {
            valid_medoids.push_back(medoid);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    kmeans_time = std::chrono::duration<double>(end - start).count();

    return valid_medoids;
}

std::vector<int> compute_degrees(efanna2e::IndexNSG* nsg_index, size_t num_vectors) {
    std::vector<int> degrees(num_vectors, 0);
    const auto& final_graph = nsg_index->GetGraph();

    for (size_t i = 0; i < num_vectors && i < final_graph.size(); i++) {
        degrees[i] = final_graph[i].size();
    }

    return degrees;
}

std::vector<int64_t> select_entry_points_by_degree(
    const std::vector<int64_t>& medoids,
    efanna2e::IndexNSG* nsg_index,
    size_t num_vectors,
    double& degree_time) {

    auto start = std::chrono::high_resolution_clock::now();

    auto degrees = compute_degrees(nsg_index, num_vectors);
    const auto& final_graph = nsg_index->GetGraph();

    std::vector<int64_t> entry_points;

    for (int64_t medoid : medoids) {
        if (medoid < 0 || medoid >= static_cast<int64_t>(num_vectors)) {
            continue;
        }

        int64_t best_node = medoid;
        int max_degree = degrees[medoid];

        if (medoid < static_cast<int64_t>(final_graph.size())) {
            for (unsigned neighbor : final_graph[medoid]) {
                if (neighbor < num_vectors) {
                    int neighbor_degree = degrees[neighbor];
                    if (neighbor_degree > max_degree) {
                        max_degree = neighbor_degree;
                        best_node = neighbor;
                    }
                }
            }
        }

        entry_points.push_back(best_node);
    }

    auto end = std::chrono::high_resolution_clock::now();
    degree_time = std::chrono::duration<double>(end - start).count();

    return entry_points;
}

void save_entry_points(const std::string& output_file,
                      const std::vector<int64_t>& entry_points,
                      int num_clusters,
                      const std::string& graph_file,
                      const std::string& base_fvecs_file,
                      const std::string& idx_file,
                      const std::vector<uint32_t>& local_to_global_mapping) {
    std::ofstream out(output_file);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open output file: " + output_file);
    }

    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);

    out << "# NSG Entry Points (Q-leaf Algorithm)" << std::endl;
    out << "# Generated: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << std::endl;
    out << "# Architecture: Global Pool + Mapping" << std::endl;
    out << "# Base file: " << base_fvecs_file << std::endl;
    out << "# Index file: " << idx_file << std::endl;
    out << "# Graph file: " << graph_file << std::endl;
    out << "# Num clusters (target): " << num_clusters << std::endl;
    out << "# Num entry points (actual): " << entry_points.size() << std::endl;
    out << "# Algorithm: " << policy_description(g_policy) << std::endl;
    out << "#" << std::endl;
    out << "# Entry Points (Local ID | Global ID):" << std::endl;

    for (int64_t local_ep : entry_points) {
        uint32_t global_ep = local_to_global_mapping[local_ep];
        out << local_ep << " | " << global_ep << std::endl;
    }

    out.close();
}

void process_single_subgraph(EntryPointTask& task,
                              const std::string& base_fvecs_file,
                              int num_clusters,
                              std::shared_ptr<MmapVectorReader>& global_base_data) {
    try {
        auto task_start = std::chrono::high_resolution_clock::now();

        std::cout << "\n[Processing] " << task.label_name << std::endl;
        std::cout << "  Graph: " << task.graph_file << std::endl;
        std::cout << "  Index: " << task.idx_file << std::endl;

        auto load_start = std::chrono::high_resolution_clock::now();

        std::vector<uint32_t> indices = load_label_indices(task.idx_file.c_str());
        size_t num_vectors = indices.size();
        size_t dim = global_base_data->dimension();

        const int target = (num_clusters > 0) ? num_clusters : entry_count_for_size(num_vectors);
        const int m = std::min(target, static_cast<int>(num_vectors));
        const bool need_vectors = (g_policy == EpPolicy::QLeaf || g_policy == EpPolicy::KMeans);

        AlignedVector32<float> data(need_vectors ? num_vectors * dim : 0);
        if (need_vectors) {
            for (size_t i = 0; i < num_vectors; i++) {
                const float* vec = global_base_data->get_vector(indices[i]);
                std::memcpy(&data[i * dim], vec, dim * sizeof(float));
            }
        }

        auto load_end = std::chrono::high_resolution_clock::now();
        task.load_time = std::chrono::duration<double>(load_end - load_start).count();
        task.num_vectors = num_vectors;

        std::cout << "  Loaded " << num_vectors << " ids (dim=" << dim << "), policy=" << policy_name(g_policy)
                  << ", m=" << m << std::endl;

        std::vector<int64_t> entry_points;

        std::mt19937_64 rng(g_seed ^ (std::hash<std::string>{}(task.label_name) * 0x9E3779B97F4A7C15ULL));

        if (g_policy == EpPolicy::NsgEp) {

            entry_points.push_back(read_nsg_navigating_node(task.graph_file, num_vectors));
        } else if (g_policy == EpPolicy::Random1 || g_policy == EpPolicy::RandomM) {
            const size_t want = (g_policy == EpPolicy::Random1) ? 1 : static_cast<size_t>(m);
            std::vector<int64_t> pool_ids(num_vectors);
            for (size_t i = 0; i < num_vectors; i++) pool_ids[i] = static_cast<int64_t>(i);
            for (size_t i = 0; i < want && i < num_vectors; i++) {
                std::uniform_int_distribution<size_t> pick(i, num_vectors - 1);
                std::swap(pool_ids[i], pool_ids[pick(rng)]);
                entry_points.push_back(pool_ids[i]);
            }
        } else if (g_policy == EpPolicy::OutDeg) {
            auto nsg_index = std::make_shared<efanna2e::IndexNSG>(dim, num_vectors, efanna2e::L2, nullptr);
            nsg_index->Load(task.graph_file.c_str());
            auto degrees = compute_degrees(nsg_index.get(), num_vectors);
            std::vector<int64_t> order(num_vectors);
            for (size_t i = 0; i < num_vectors; i++) order[i] = static_cast<int64_t>(i);
            const size_t want = std::min(static_cast<size_t>(m), num_vectors);
            std::partial_sort(order.begin(), order.begin() + want, order.end(), [&](int64_t a, int64_t b) {
                return degrees[a] != degrees[b] ? degrees[a] > degrees[b] : a < b;
            });
            entry_points.assign(order.begin(), order.begin() + want);
        } else {

            auto medoids = find_cluster_medoids(data, num_vectors, dim, m, task.kmeans_time);
            std::cout << "  K-Means completed: " << medoids.size() << " medoids found ("
                      << task.kmeans_time << "s)" << std::endl;
            if (g_policy == EpPolicy::KMeans) {
                entry_points = medoids;
            } else {

                auto nsg_index = std::make_shared<efanna2e::IndexNSG>(dim, num_vectors, efanna2e::L2, nullptr);
                nsg_index->Load(task.graph_file.c_str());
                entry_points = select_entry_points_by_degree(medoids, nsg_index.get(), num_vectors, task.degree_time);
                std::cout << "  Degree selection completed: " << entry_points.size() << " entry points ("
                          << task.degree_time << "s)" << std::endl;
            }
        }
        const size_t original_entry_points = entry_points.size();
        deduplicate_preserve_order(entry_points);
        std::cout << "  Entry points " << task.output_file
                  << ": original=" << original_entry_points
                  << ", unique=" << entry_points.size() << std::endl;
        task.num_entry_points = entry_points.size();

        save_entry_points(task.output_file, entry_points, m,
                         task.graph_file, base_fvecs_file, task.idx_file, indices);

        auto task_end = std::chrono::high_resolution_clock::now();
        task.total_time = std::chrono::duration<double>(task_end - task_start).count();
        task.success = true;

        std::cout << "  ✓ Completed in " << task.total_time << "s" << std::endl;

    } catch (const std::exception& e) {
        task.success = false;
        task.error_message = e.what();
        std::cerr << "  ✗ Failed: " << e.what() << std::endl;
    }
}

std::vector<EntryPointTask> scan_graph_files(const std::string& graph_dir,
                                              const std::string& output_dir) {
    std::vector<EntryPointTask> tasks;

    DIR* dir = opendir(graph_dir.c_str());
    if (!dir) {
        std::cerr << "Error: Cannot open graph directory: " << graph_dir << std::endl;
        return tasks;
    }

    std::cout << "Scanning for .graph files in " << graph_dir << std::endl;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename(entry->d_name);

        if (filename.size() > 10 && filename.substr(filename.size() - 10) == "_nsg.graph") {
            std::string label = filename.substr(0, filename.size() - 10);

            EntryPointTask task;
            task.label_name = label;
            task.graph_file = graph_dir + "/" + filename;
            task.idx_file = graph_dir + "/" + label + ".idx";
            task.output_file = output_dir + "/" + label + "_entry_points.txt";

            struct stat idx_stat;
            if (stat(task.idx_file.c_str(), &idx_stat) == 0) {
                tasks.push_back(task);
                std::cout << "  Found: " << label << std::endl;
            } else {
                std::cerr << "  Warning: No .idx file for " << label << std::endl;
            }
        }
    }

    closedir(dir);

    std::sort(tasks.begin(), tasks.end(),
              [](const EntryPointTask& a, const EntryPointTask& b) {
                  return a.label_name < b.label_name;
              });

    std::cout << "Found " << tasks.size() << " subgraphs to process\n" << std::endl;
    return tasks;
}

int main(int argc, char** argv) {
    std::string base_fvecs_file;
    std::string graph_dir;
    std::string output_dir;
    int num_clusters = 60;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--base-fvecs" && i + 1 < argc) {
            base_fvecs_file = argv[++i];
        } else if (arg == "--graph-dir" && i + 1 < argc) {
            graph_dir = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--num-clusters" && i + 1 < argc) {
            num_clusters = std::stoi(argv[++i]);
        } else if (arg == "--policy" && i + 1 < argc) {
            std::string p = argv[++i];
            if (!parse_policy(p, g_policy)) {
                std::cerr << "Error: unknown --policy " << p
                          << " (qleaf | kmeans | outdeg | random | random1 | nsgep)" << std::endl;
                return -1;
            }
        } else if (arg == "--seed" && i + 1 < argc) {
            g_seed = std::stoull(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --base-fvecs <file>     Original base.fvecs file" << std::endl;
            std::cout << "  --graph-dir <dir>       Directory containing .graph and .idx files" << std::endl;
            std::cout << "  --output-dir <dir>      Output directory for entry points" << std::endl;
            std::cout << "  --num-clusters <K>      Entry-set size; 0 = by subgraph size (16/32/64) (default: 60)" << std::endl;
            std::cout << "  --policy <name>         qleaf (default) | kmeans | outdeg | random | random1 | nsgep" << std::endl;
            std::cout << "  --seed <S>              Seed for random / random1 (default: 42)" << std::endl;
            std::cout << "  --help                  Show this help message" << std::endl;
            return 0;
        } else {
            std::cerr << "Error: unknown or incomplete argument " << arg << std::endl;
            return -1;
        }
    }

    if (base_fvecs_file.empty() || graph_dir.empty() || output_dir.empty()) {
        std::cerr << "Error: Missing required arguments" << std::endl;
        std::cerr << "Usage: " << argv[0]
                  << " --base-fvecs <base.fvecs> --graph-dir <dir> --output-dir <dir> [--num-clusters <K>]"
                  << std::endl;
        return -1;
    }

    try {
        std::cout << "===========================================================" << std::endl;
        std::cout << "NSG Entry Point Batch Generation (Sequential Processing)" << std::endl;
        std::cout << "===========================================================" << std::endl;
        std::cout << "Base file: " << base_fvecs_file << std::endl;
        std::cout << "Graph directory: " << graph_dir << std::endl;
        std::cout << "Output directory: " << output_dir << std::endl;
        std::cout << "Target clusters: " << (num_clusters > 0 ? std::to_string(num_clusters) : "auto (16/32/64)")
                  << std::endl;
        std::cout << "Policy: " << policy_description(g_policy) << ", seed " << g_seed << std::endl;
        std::cout << "Processing: SEQUENTIAL (each K-Means uses all CPU cores)" << std::endl;
        std::cout << "===========================================================\n" << std::endl;

        long vmhwm_start_kb = get_vmhwm_kb();
        auto pipeline_start = std::chrono::high_resolution_clock::now();

        std::cout << "Loading global base.fvecs via mmap..." << std::endl;
        auto mmap_start = std::chrono::high_resolution_clock::now();
        auto global_base_data = std::make_shared<MmapVectorReader>(base_fvecs_file.c_str());
        auto mmap_end = std::chrono::high_resolution_clock::now();
        double mmap_time = std::chrono::duration<double>(mmap_end - mmap_start).count();

        std::cout << "✓ Loaded " << global_base_data->num_vectors() << " vectors, dim="
                  << global_base_data->dimension() << " (" << mmap_time << "s)\n" << std::endl;

        mkdir(output_dir.c_str(), 0755);

        auto tasks = scan_graph_files(graph_dir, output_dir);
        if (tasks.empty()) {
            std::cerr << "Error: No .graph files found!" << std::endl;
            return -1;
        }

        std::cout << "=== Starting Sequential Processing ===" << std::endl;
        std::cout << "Total subgraphs: " << tasks.size() << "\n" << std::endl;

        for (auto& task : tasks) {
            process_single_subgraph(task, base_fvecs_file, num_clusters, global_base_data);
        }

        auto pipeline_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(pipeline_end - pipeline_start).count();

        long vmhwm_end_kb = get_vmhwm_kb();

        std::cout << "\n===========================================================" << std::endl;
        std::cout << "Summary" << std::endl;
        std::cout << "===========================================================" << std::endl;

        int success_count = 0;
        int failure_count = 0;
        double total_load_time = 0.0;
        double total_kmeans_time = 0.0;
        double total_degree_time = 0.0;
        size_t total_vectors = 0;
        size_t total_entry_points = 0;

        for (const auto& task : tasks) {
            if (task.success) {
                success_count++;
                total_load_time += task.load_time;
                total_kmeans_time += task.kmeans_time;
                total_degree_time += task.degree_time;
                total_vectors += task.num_vectors;
                total_entry_points += task.num_entry_points;
                std::cout << "✓ " << task.label_name
                          << " (" << task.num_vectors << " vectors, "
                          << task.num_entry_points << " EPs, "
                          << std::fixed << std::setprecision(2) << task.total_time << "s)" << std::endl;
            } else {
                failure_count++;
                std::cout << "✗ " << task.label_name << " - " << task.error_message << std::endl;
            }
        }

        std::cout << "\n=== Overall Statistics ===" << std::endl;
        std::cout << "Total subgraphs: " << tasks.size() << std::endl;
        std::cout << "  Successful: " << success_count << std::endl;
        std::cout << "  Failed: " << failure_count << std::endl;
        std::cout << "Total vectors processed: " << total_vectors << std::endl;
        std::cout << "Total entry points generated: " << total_entry_points << std::endl;
        std::cout << std::endl;

        std::cout << "=== Timing Breakdown ===" << std::endl;
        std::cout << "  Global mmap load: " << std::fixed << std::setprecision(2)
                  << mmap_time << "s" << std::endl;
        std::cout << "  Vector extraction (all subgraphs): " << total_load_time << "s" << std::endl;
        std::cout << "  K-Means clustering (all subgraphs): " << total_kmeans_time << "s" << std::endl;
        std::cout << "  Degree selection (all subgraphs): " << total_degree_time << "s" << std::endl;
        std::cout << "  Total pipeline time: " << total_time << "s" << std::endl;

        if (success_count > 0) {
            std::cout << "\n=== Average Time (per subgraph) ===" << std::endl;
            std::cout << "  Vector extraction: " << (total_load_time / success_count) << "s" << std::endl;
            std::cout << "  K-Means: " << (total_kmeans_time / success_count) << "s" << std::endl;
            std::cout << "  Degree selection: " << (total_degree_time / success_count) << "s" << std::endl;
            std::cout << "  Total: " << (total_time / success_count) << "s" << std::endl;
        }

        if (vmhwm_start_kb > 0 && vmhwm_end_kb > 0) {
            std::cout << "\n=== Memory Usage (Peak RSS) ===" << std::endl;
            std::cout << "  Initial VmHWM: " << (vmhwm_start_kb / 1024.0) << " MB" << std::endl;
            std::cout << "  Final VmHWM: " << (vmhwm_end_kb / 1024.0) << " MB" << std::endl;
            std::cout << "  Peak memory usage: " << (vmhwm_end_kb / 1024.0) << " MB" << std::endl;
            std::cout << "  Note: Sequential processing with FAISS internal parallelization" << std::endl;
        }

        std::cout << "===========================================================" << std::endl;

        return (failure_count == 0) ? 0 : -1;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return -1;
    }
}
