#include <torch/extension.h>

#include <queue>
#include <stdexcept>
#include <string>
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

}  // namespace

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "clustering_seed_selection_forward",
      &clustering_seed_selection_forward,
      "LatentFormer clustering seed selection forward");
}
