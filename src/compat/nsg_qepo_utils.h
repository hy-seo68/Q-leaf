// Compatibility definitions for graph construction and entry generation.
#ifndef NSG_QEPO_UTILS_H
#define NSG_QEPO_UTILS_H

#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <faiss/utils/distances.h>

std::vector<uint32_t> load_entry_points(const std::string& filename);

inline float l2_squared_distance(const float* v1, const float* v2, size_t dim) {
    return faiss::fvec_L2sqr(v1, v2, dim);
}

inline float l2_distance(const float* v1, const float* v2, size_t dim) {
    return std::sqrt(faiss::fvec_L2sqr(v1, v2, dim));
}

std::pair<uint32_t, float> select_best_entry_point_fast(
    const float* query,
    const std::vector<uint32_t>& entry_points,
    const float* entry_points_data,
    size_t num_eps,
    size_t dim,
    uint64_t& ep_dist_counter,
    float* distances_buffer
);

std::vector<std::pair<uint32_t, float>> select_top_entry_points_fast(
    const float* query,
    const std::vector<uint32_t>& entry_points,
    const float* entry_points_data,
    size_t num_eps,
    size_t dim,
    int num_selected,
    uint64_t& ep_dist_counter,
    float* distances_buffer,
    int64_t* indexes_buffer
);

float calculate_recall(const std::vector<std::vector<uint32_t>>& results,
                      const std::vector<std::vector<uint32_t>>& id_mappings,
                      const std::vector<uint32_t>& ground_truth,
                      size_t gt_dim, int k);

float calculate_recall_flat(
    const uint32_t* results_flat,
    const std::vector<const std::vector<uint32_t>*>& id_mapping_ptrs,
    size_t num_queries,
    const std::vector<uint32_t>& ground_truth,
    size_t gt_dim, int k);

#endif
