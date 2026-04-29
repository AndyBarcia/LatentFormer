#include <torch/extension.h>

#include <algorithm>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

torch::Tensor normalize_last_dim(const torch::Tensor& values, double eps) {
  auto denom = values.pow(2).sum(-1, true).sqrt().clamp_min(eps);
  return values / denom;
}

torch::Tensor compute_similarity(
    const torch::Tensor& query_signatures,
    const std::string& metric,
    double eps,
    double temp) {
  torch::Tensor lhs = query_signatures;

  if (metric == "centered-cosine") {
    lhs = lhs - lhs.mean(-1, true);
    lhs = normalize_last_dim(lhs, eps);
  } else if (metric == "cosine") {
    lhs = normalize_last_dim(lhs, eps);
  } else if (metric == "softmax") {
    lhs = torch::softmax(lhs / temp, -1);
  } else if (metric != "dot" && metric != "dot-sigmoid") {
    throw std::invalid_argument(
        "LatentFormerSeedSelection only supports dot, dot-sigmoid, cosine, "
        "centered-cosine, and softmax metrics.");
  }

  auto similarity = torch::matmul(lhs, lhs.transpose(-1, -2));
  if (metric == "dot-sigmoid") {
    similarity = torch::sigmoid(similarity / temp);
  }
  return similarity.clamp(0.0, 1.0);
}

class DsuCounter {
 public:
  DsuCounter(const std::vector<uint8_t>& active, const std::vector<uint8_t>& matched)
      : parent_(active.size()),
        rank_(active.size(), 0),
        active_(active),
        has_match_(matched),
        tp_(0),
        fp_(0) {
    std::iota(parent_.begin(), parent_.end(), 0);
    for (size_t idx = 0; idx < active_.size(); ++idx) {
      if (!active_[idx]) {
        continue;
      }
      if (has_match_[idx]) {
        ++tp_;
      } else {
        ++fp_;
      }
    }
  }

  int64_t find(int64_t value) {
    if (parent_[value] != value) {
      parent_[value] = find(parent_[value]);
    }
    return parent_[value];
  }

  void unite(int64_t lhs, int64_t rhs) {
    if (!active_[lhs] || !active_[rhs]) {
      return;
    }

    int64_t lhs_root = find(lhs);
    int64_t rhs_root = find(rhs);
    if (lhs_root == rhs_root) {
      return;
    }

    remove_component(lhs_root);
    remove_component(rhs_root);

    if (rank_[lhs_root] < rank_[rhs_root]) {
      std::swap(lhs_root, rhs_root);
    }
    parent_[rhs_root] = lhs_root;
    if (rank_[lhs_root] == rank_[rhs_root]) {
      ++rank_[lhs_root];
    }
    active_[lhs_root] = 1;
    has_match_[lhs_root] = has_match_[lhs_root] || has_match_[rhs_root];
    add_component(lhs_root);
  }

  int64_t tp() const {
    return tp_;
  }

  int64_t fp() const {
    return fp_;
  }

 private:
  void remove_component(int64_t root) {
    if (has_match_[root]) {
      --tp_;
    } else {
      --fp_;
    }
  }

  void add_component(int64_t root) {
    if (has_match_[root]) {
      ++tp_;
    } else {
      ++fp_;
    }
  }

  std::vector<int64_t> parent_;
  std::vector<uint8_t> rank_;
  std::vector<uint8_t> active_;
  std::vector<uint8_t> has_match_;
  int64_t tp_;
  int64_t fp_;
};

}  // namespace

#ifdef WITH_CUDA
std::vector<torch::Tensor> clustering_seed_selection_cuda_forward(
    const torch::Tensor& query_signatures,
    const torch::Tensor& seed_scores,
    const torch::Tensor& selected_mask,
    const torch::Tensor& similarity,
    double duplicate_threshold);

std::vector<torch::Tensor> seed_cluster_precision_recall_cuda_forward(
    const torch::Tensor& seed_scores,
    const torch::Tensor& matched_query_mask,
    const torch::Tensor& seed_thresholds,
    const torch::Tensor& duplicate_thresholds,
    const torch::Tensor& similarity);
#endif

std::vector<torch::Tensor> clustering_seed_selection_forward(
    const torch::Tensor& query_signatures,
    const torch::Tensor& query_seed_logits,
    double seed_threshold,
    double duplicate_threshold,
    const std::string& metric,
    double eps,
    double temp) {
  TORCH_CHECK(query_signatures.dim() == 3, "query_signatures must be shaped [B,Q,C]");
  TORCH_CHECK(query_seed_logits.dim() == 2, "query_seed_logits must be shaped [B,Q]");
  TORCH_CHECK(
      query_signatures.size(0) == query_seed_logits.size(0) &&
          query_signatures.size(1) == query_seed_logits.size(1),
      "query_signatures and query_seed_logits must agree on [B,Q]");
  TORCH_CHECK(query_signatures.size(1) > 0, "query_signatures must contain at least one query");

  const auto batch_size = query_signatures.size(0);
  const auto num_queries = query_signatures.size(1);
  const auto sig_dim = query_signatures.size(2);
  const auto device = query_signatures.device();

  auto seed_scores = query_seed_logits.to(torch::kFloat).sigmoid();
  auto selected_mask = seed_scores >= seed_threshold;
  auto has_selection = selected_mask.any(1);
  if (!has_selection.all().item<bool>()) {
    selected_mask = selected_mask.clone();
    auto fallback_indices = std::get<1>(seed_scores.max(1));
    auto has_selection_cpu = has_selection.to(torch::kCPU).contiguous();
    auto fallback_cpu = fallback_indices.to(torch::kCPU).contiguous();
    auto has_selection_a = has_selection_cpu.accessor<bool, 1>();
    auto fallback_a = fallback_cpu.accessor<int64_t, 1>();
    for (int64_t b = 0; b < batch_size; ++b) {
      if (!has_selection_a[b]) {
        selected_mask[b][fallback_a[b]] = true;
      }
    }
  }

  auto similarity = compute_similarity(query_signatures, metric, eps, temp);
#ifdef WITH_CUDA
  return clustering_seed_selection_cuda_forward(
      query_signatures.to(torch::kFloat).contiguous(),
      seed_scores.to(torch::kFloat).contiguous(),
      selected_mask.contiguous(),
      similarity.to(torch::kFloat).contiguous(),
      duplicate_threshold);
#endif
  TORCH_WARN_ONCE(
      "LatentFormerSeedSelection clustering_seed_selection_forward is using the CPU "
      "fallback. The CUDA path requires the extension to be built with CUDA ");

  auto adjacency = similarity >= duplicate_threshold;
  adjacency = adjacency.logical_and(selected_mask.unsqueeze(2)).logical_and(selected_mask.unsqueeze(1));

  auto selected_cpu = selected_mask.to(torch::kCPU).contiguous();
  auto adjacency_cpu = adjacency.to(torch::kCPU).contiguous();
  auto scores_cpu = seed_scores.to(torch::kCPU).contiguous();
  auto selected_a = selected_cpu.accessor<bool, 2>();
  auto adjacency_a = adjacency_cpu.accessor<bool, 3>();
  auto scores_a = scores_cpu.accessor<float, 2>();

  std::vector<std::vector<int64_t>> best_indices(batch_size);
  int64_t max_num_seeds = 0;

  for (int64_t b = 0; b < batch_size; ++b) {
    std::vector<uint8_t> visited(num_queries, 0);
    for (int64_t start = 0; start < num_queries; ++start) {
      if (!selected_a[b][start] || visited[start]) {
        continue;
      }

      int64_t best_idx = start;
      float best_score = scores_a[b][start];
      std::queue<int64_t> queue;
      visited[start] = 1;
      queue.push(start);

      while (!queue.empty()) {
        const int64_t current = queue.front();
        queue.pop();

        const float current_score = scores_a[b][current];
        if (current_score > best_score) {
          best_score = current_score;
          best_idx = current;
        }

        for (int64_t neighbor = 0; neighbor < num_queries; ++neighbor) {
          if (selected_a[b][neighbor] && !visited[neighbor] && adjacency_a[b][current][neighbor]) {
            visited[neighbor] = 1;
            queue.push(neighbor);
          }
        }
      }

      best_indices[b].push_back(best_idx);
    }
    max_num_seeds = std::max<int64_t>(max_num_seeds, best_indices[b].size());
  }

  auto seed_signatures = torch::zeros(
      {batch_size, max_num_seeds, sig_dim}, query_signatures.options());
  auto out_scores = torch::zeros(
      {batch_size, max_num_seeds}, seed_scores.options().device(device));
  auto pad_mask = torch::zeros(
      {batch_size, max_num_seeds}, torch::TensorOptions().dtype(torch::kBool).device(device));

  for (int64_t b = 0; b < batch_size; ++b) {
    const auto count = static_cast<int64_t>(best_indices[b].size());
    if (count == 0) {
      continue;
    }

    auto idx = torch::from_blob(best_indices[b].data(), {count}, torch::kLong).clone().to(device);
    seed_signatures[b].narrow(0, 0, count).copy_(query_signatures[b].index_select(0, idx));
    out_scores[b].narrow(0, 0, count).copy_(seed_scores[b].index_select(0, idx));
    pad_mask[b].narrow(0, 0, count).fill_(true);
  }

  return {seed_signatures, pad_mask, out_scores};
}

std::vector<torch::Tensor> seed_cluster_precision_recall_forward(
    const torch::Tensor& query_signatures,
    const torch::Tensor& query_seed_logits,
    const torch::Tensor& matched_query_mask,
    const torch::Tensor& matched_gt_indices,
    const torch::Tensor& seed_thresholds,
    const torch::Tensor& duplicate_thresholds,
    const std::string& metric,
    double eps,
    double temp) {
  TORCH_CHECK(query_signatures.dim() == 3, "query_signatures must be shaped [B,Q,C]");
  TORCH_CHECK(query_seed_logits.dim() == 2, "query_seed_logits must be shaped [B,Q]");
  TORCH_CHECK(matched_query_mask.dim() == 2, "matched_query_mask must be shaped [B,Q]");
  TORCH_CHECK(matched_gt_indices.dim() == 2, "matched_gt_indices must be shaped [B,Q]");
  TORCH_CHECK(seed_thresholds.dim() == 1, "seed_thresholds must be a 1D tensor");
  TORCH_CHECK(duplicate_thresholds.dim() == 1, "duplicate_thresholds must be a 1D tensor");
  TORCH_CHECK(
      query_signatures.size(0) == query_seed_logits.size(0) &&
          query_signatures.size(1) == query_seed_logits.size(1),
      "query_signatures and query_seed_logits must agree on [B,Q]");
  TORCH_CHECK(
      matched_query_mask.sizes() == query_seed_logits.sizes(),
      "matched_query_mask must match query_seed_logits shape");
  TORCH_CHECK(
      matched_gt_indices.sizes() == query_seed_logits.sizes(),
      "matched_gt_indices must match query_seed_logits shape");

  const auto batch_size = query_signatures.size(0);
  const auto num_queries = query_signatures.size(1);
  const auto num_seed_thresholds = seed_thresholds.size(0);
  const auto num_duplicate_thresholds = duplicate_thresholds.size(0);
  const auto device = query_signatures.device();

#ifdef WITH_CUDA
  auto seed_scores = query_seed_logits.to(torch::kFloat).sigmoid();
  auto similarity = compute_similarity(query_signatures, metric, eps, temp).to(torch::kFloat);
  return seed_cluster_precision_recall_cuda_forward(
      seed_scores.contiguous(),
      matched_query_mask.to(torch::kBool).contiguous(),
      seed_thresholds.to(torch::kFloat).contiguous(),
      duplicate_thresholds.to(torch::kFloat).contiguous(),
      similarity.contiguous());
#endif
  TORCH_WARN_ONCE(
      "LatentFormerSeedSelection seed_cluster_precision_recall_forward is using the "
      "CPU fallback. The CUDA path requires the extension to be built with CUDA ");

  auto seed_scores_cpu = query_seed_logits.to(torch::kFloat).sigmoid().to(torch::kCPU).contiguous();
  auto matched_cpu = matched_query_mask.to(torch::kBool).to(torch::kCPU).contiguous();
  auto matched_gt_cpu = matched_gt_indices.to(torch::kLong).to(torch::kCPU).contiguous();
  auto seed_thresholds_cpu = seed_thresholds.to(torch::kFloat).to(torch::kCPU).contiguous();
  auto duplicate_thresholds_cpu = duplicate_thresholds.to(torch::kFloat).to(torch::kCPU).contiguous();
  auto similarity_cpu = compute_similarity(query_signatures, metric, eps, temp)
                            .to(torch::kFloat)
                            .to(torch::kCPU)
                            .contiguous();

  auto scores_a = seed_scores_cpu.accessor<float, 2>();
  auto matched_a = matched_cpu.accessor<bool, 2>();
  auto matched_gt_a = matched_gt_cpu.accessor<int64_t, 2>();
  auto seed_thresholds_a = seed_thresholds_cpu.accessor<float, 1>();
  auto duplicate_thresholds_a = duplicate_thresholds_cpu.accessor<float, 1>();
  auto similarity_a = similarity_cpu.accessor<float, 3>();

  auto cpu_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
  auto tp_cpu = torch::zeros(
      {batch_size, num_seed_thresholds, num_duplicate_thresholds}, cpu_options);
  auto fp_cpu = torch::zeros_like(tp_cpu);
  auto fn_cpu = torch::zeros_like(tp_cpu);
  auto tp_a = tp_cpu.accessor<float, 3>();
  auto fp_a = fp_cpu.accessor<float, 3>();
  auto fn_a = fn_cpu.accessor<float, 3>();

  std::vector<int64_t> total_matched(batch_size, 0);
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t q = 0; q < num_queries; ++q) {
      if (matched_a[b][q]) {
        TORCH_CHECK(
            matched_gt_a[b][q] >= 0,
            "matched_gt_indices must be non-negative where matched_query_mask is true");
        total_matched[b] += 1;
      }
    }
  }

  std::vector<int64_t> seed_order(num_seed_thresholds);
  std::iota(seed_order.begin(), seed_order.end(), 0);
  std::sort(seed_order.begin(), seed_order.end(), [&](int64_t lhs, int64_t rhs) {
    return seed_thresholds_a[lhs] > seed_thresholds_a[rhs];
  });

  std::vector<int64_t> duplicate_order(num_duplicate_thresholds);
  std::iota(duplicate_order.begin(), duplicate_order.end(), 0);
  std::sort(duplicate_order.begin(), duplicate_order.end(), [&](int64_t lhs, int64_t rhs) {
    return duplicate_thresholds_a[lhs] > duplicate_thresholds_a[rhs];
  });

  for (int64_t b = 0; b < batch_size; ++b) {
    std::vector<std::tuple<float, int64_t, int64_t>> edges;
    edges.reserve(num_queries * (num_queries - 1) / 2);
    for (int64_t lhs = 0; lhs < num_queries; ++lhs) {
      for (int64_t rhs = lhs + 1; rhs < num_queries; ++rhs) {
        edges.emplace_back(similarity_a[b][lhs][rhs], lhs, rhs);
      }
    }
    std::sort(edges.begin(), edges.end(), [](const auto& lhs, const auto& rhs) {
      return std::get<0>(lhs) > std::get<0>(rhs);
    });

    std::vector<uint8_t> matched(num_queries, 0);
    for (int64_t q = 0; q < num_queries; ++q) {
      matched[q] = matched_a[b][q] ? 1 : 0;
    }

    for (const int64_t seed_idx : seed_order) {
      const float seed_threshold = seed_thresholds_a[seed_idx];
      std::vector<uint8_t> active(num_queries, 0);
      for (int64_t q = 0; q < num_queries; ++q) {
        active[q] = scores_a[b][q] >= seed_threshold ? 1 : 0;
      }

      DsuCounter dsu(active, matched);
      size_t edge_cursor = 0;
      for (const int64_t duplicate_idx : duplicate_order) {
        const float duplicate_threshold = duplicate_thresholds_a[duplicate_idx];
        while (edge_cursor < edges.size() &&
               std::get<0>(edges[edge_cursor]) >= duplicate_threshold) {
          dsu.unite(std::get<1>(edges[edge_cursor]), std::get<2>(edges[edge_cursor]));
          ++edge_cursor;
        }

        const int64_t image_tp = dsu.tp();
        const int64_t image_fp = dsu.fp();
        tp_a[b][seed_idx][duplicate_idx] = static_cast<float>(image_tp);
        fp_a[b][seed_idx][duplicate_idx] = static_cast<float>(image_fp);
        fn_a[b][seed_idx][duplicate_idx] = static_cast<float>(total_matched[b] - image_tp);
      }
    }
  }

  auto tp = tp_cpu.to(device);
  auto fp = fp_cpu.to(device);
  auto fn = fn_cpu.to(device);
  auto precision = tp / (tp + fp).clamp_min(1.0);
  auto recall = tp / (tp + fn).clamp_min(1.0);
  auto total_tp = tp.sum(0);
  auto total_fp = fp.sum(0);
  auto total_fn = fn.sum(0);
  auto micro_precision = total_tp / (total_tp + total_fp).clamp_min(1.0);
  auto micro_recall = total_tp / (total_tp + total_fn).clamp_min(1.0);

  return {
      tp,
      fp,
      fn,
      precision,
      recall,
      total_tp,
      total_fp,
      total_fn,
      micro_precision,
      micro_recall,
  };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "clustering_seed_selection_forward",
      &clustering_seed_selection_forward,
      "LatentFormer clustering seed selection forward");
  m.def(
      "seed_cluster_precision_recall_forward",
      &seed_cluster_precision_recall_forward,
      "LatentFormer seed cluster precision/recall forward");
}
