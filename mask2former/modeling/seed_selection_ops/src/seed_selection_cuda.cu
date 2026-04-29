#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <limits>
#include <vector>

namespace {

constexpr int kThreads = 256;

__global__ void init_labels_kernel(
    const bool* __restrict__ selected,
    int* __restrict__ labels,
    int total,
    int num_queries) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  labels[idx] = selected[idx] ? idx % num_queries : -1;
}

__global__ void propagate_labels_kernel(
    const bool* __restrict__ selected,
    const float* __restrict__ similarity,
    int* __restrict__ labels,
    int* __restrict__ changed,
    int batch_size,
    int num_queries,
    float duplicate_threshold) {
  int b = blockIdx.y;
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch_size || q >= num_queries) {
    return;
  }

  int offset = b * num_queries;
  int idx = offset + q;
  if (!selected[idx]) {
    return;
  }

  int best = labels[idx];
  const float* sim_row = similarity + (static_cast<long long>(b) * num_queries + q) * num_queries;
  for (int neighbor = 0; neighbor < num_queries; ++neighbor) {
    int neighbor_idx = offset + neighbor;
    int neighbor_label = labels[neighbor_idx];
    if (selected[neighbor_idx] && neighbor_label >= 0 && sim_row[neighbor] >= duplicate_threshold) {
      best = min(best, neighbor_label);
    }
  }

  if (best < labels[idx]) {
    labels[idx] = best;
    atomicExch(changed, 1);
  }
}

__global__ void select_component_best_kernel(
    const int* __restrict__ labels,
    const float* __restrict__ scores,
    int* __restrict__ best_indices,
    float* __restrict__ best_scores,
    int batch_size,
    int num_queries) {
  int b = blockIdx.y;
  int root = blockIdx.x;
  if (b >= batch_size || root >= num_queries) {
    return;
  }

  __shared__ float thread_scores[kThreads];
  __shared__ int thread_indices[kThreads];

  int offset = b * num_queries;
  float best_score = -1.0f;
  int best_idx = -1;
  for (int q = threadIdx.x; q < num_queries; q += blockDim.x) {
    int idx = offset + q;
    if (labels[idx] == root) {
      float score = scores[idx];
      if (score > best_score || (score == best_score && (best_idx < 0 || q < best_idx))) {
        best_score = score;
        best_idx = q;
      }
    }
  }

  thread_scores[threadIdx.x] = best_score;
  thread_indices[threadIdx.x] = best_idx;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      float other_score = thread_scores[threadIdx.x + stride];
      int other_idx = thread_indices[threadIdx.x + stride];
      if (other_score > thread_scores[threadIdx.x] ||
          (other_score == thread_scores[threadIdx.x] &&
           other_idx >= 0 &&
           (thread_indices[threadIdx.x] < 0 || other_idx < thread_indices[threadIdx.x]))) {
        thread_scores[threadIdx.x] = other_score;
        thread_indices[threadIdx.x] = other_idx;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && thread_indices[0] >= 0) {
    best_indices[offset + root] = thread_indices[0];
    best_scores[offset + root] = thread_scores[0];
  }
}

__global__ void gather_seed_outputs_kernel(
    const float* __restrict__ query_signatures,
    const int* __restrict__ best_indices,
    const float* __restrict__ best_scores,
    float* __restrict__ seed_signatures,
    bool* __restrict__ pad_mask,
    float* __restrict__ out_scores,
    int batch_size,
    int num_queries,
    int sig_dim) {
  long long total = static_cast<long long>(batch_size) * num_queries * sig_dim;
  long long linear = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (linear >= total) {
    return;
  }

  int c = linear % sig_dim;
  long long tmp = linear / sig_dim;
  int root = tmp % num_queries;
  int b = tmp / num_queries;
  int flat = b * num_queries + root;
  int best_idx = best_indices[flat];

  if (c == 0 && best_idx >= 0) {
    pad_mask[flat] = true;
    out_scores[flat] = best_scores[flat];
  }
  if (best_idx >= 0) {
    seed_signatures[linear] =
        query_signatures[(static_cast<long long>(b) * num_queries + best_idx) * sig_dim + c];
  }
}

}  // namespace

std::vector<torch::Tensor> clustering_seed_selection_cuda_forward(
    const torch::Tensor& query_signatures,
    const torch::Tensor& seed_scores,
    const torch::Tensor& selected_mask,
    const torch::Tensor& similarity,
    double duplicate_threshold) {
  TORCH_CHECK(query_signatures.is_cuda(), "query_signatures must be CUDA");
  TORCH_CHECK(seed_scores.is_cuda(), "seed_scores must be CUDA");
  TORCH_CHECK(selected_mask.is_cuda(), "selected_mask must be CUDA");
  TORCH_CHECK(similarity.is_cuda(), "similarity must be CUDA");
  TORCH_CHECK(query_signatures.scalar_type() == torch::kFloat, "CUDA seed selection expects float signatures");
  TORCH_CHECK(seed_scores.scalar_type() == torch::kFloat, "CUDA seed selection expects float seed scores");
  TORCH_CHECK(selected_mask.scalar_type() == torch::kBool, "CUDA seed selection expects bool selected mask");
  TORCH_CHECK(similarity.scalar_type() == torch::kFloat, "CUDA seed selection expects float similarity");

  const c10::cuda::CUDAGuard device_guard(query_signatures.device());
  const int batch_size = static_cast<int>(query_signatures.size(0));
  const int num_queries = static_cast<int>(query_signatures.size(1));
  const int sig_dim = static_cast<int>(query_signatures.size(2));
  auto stream = at::cuda::getCurrentCUDAStream();

  auto labels = torch::empty({batch_size, num_queries}, query_signatures.options().dtype(torch::kInt));
  auto best_indices = torch::full(
      {batch_size, num_queries},
      -1,
      query_signatures.options().dtype(torch::kInt));
  auto best_scores = torch::zeros({batch_size, num_queries}, seed_scores.options());
  auto seed_signatures = torch::zeros(
      {batch_size, num_queries, sig_dim},
      query_signatures.options());
  auto out_scores = torch::zeros({batch_size, num_queries}, seed_scores.options());
  auto pad_mask = torch::zeros(
      {batch_size, num_queries},
      selected_mask.options());
  auto changed = torch::empty({1}, query_signatures.options().dtype(torch::kInt));

  int total_nodes = batch_size * num_queries;
  int init_blocks = (total_nodes + kThreads - 1) / kThreads;
  init_labels_kernel<<<init_blocks, kThreads, 0, stream>>>(
      selected_mask.data_ptr<bool>(),
      labels.data_ptr<int>(),
      total_nodes,
      num_queries);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 prop_blocks((num_queries + kThreads - 1) / kThreads, batch_size);
  for (int iter = 0; iter < num_queries; ++iter) {
    changed.zero_();
    propagate_labels_kernel<<<prop_blocks, kThreads, 0, stream>>>(
        selected_mask.data_ptr<bool>(),
        similarity.data_ptr<float>(),
        labels.data_ptr<int>(),
        changed.data_ptr<int>(),
        batch_size,
        num_queries,
        static_cast<float>(duplicate_threshold));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto changed_cpu = changed.cpu();
    if (changed_cpu.data_ptr<int>()[0] == 0) {
      break;
    }
  }

  dim3 best_blocks(num_queries, batch_size);
  select_component_best_kernel<<<best_blocks, kThreads, 0, stream>>>(
      labels.data_ptr<int>(),
      seed_scores.data_ptr<float>(),
      best_indices.data_ptr<int>(),
      best_scores.data_ptr<float>(),
      batch_size,
      num_queries);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  long long total_values = static_cast<long long>(batch_size) * num_queries * sig_dim;
  int gather_blocks = static_cast<int>((total_values + kThreads - 1) / kThreads);
  gather_seed_outputs_kernel<<<gather_blocks, kThreads, 0, stream>>>(
      query_signatures.data_ptr<float>(),
      best_indices.data_ptr<int>(),
      best_scores.data_ptr<float>(),
      seed_signatures.data_ptr<float>(),
      pad_mask.data_ptr<bool>(),
      out_scores.data_ptr<float>(),
      batch_size,
      num_queries,
      sig_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {seed_signatures, pad_mask, out_scores};
}
