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
    int batch_size,
    int num_queries,
    float duplicate_threshold) {
  int b = blockIdx.x;
  if (b >= batch_size) {
    return;
  }

  extern __shared__ int shared[];
  int* prev_labels = shared;
  int* next_labels = shared + num_queries;
  int* changed = shared + 2 * num_queries;

  int offset = b * num_queries;
  for (int q = threadIdx.x; q < num_queries; q += blockDim.x) {
    int idx = offset + q;
    prev_labels[q] = selected[idx] ? labels[idx] : -1;
  }
  __syncthreads();

  for (int iter = 0; iter < num_queries; ++iter) {
    if (threadIdx.x == 0) {
      *changed = 0;
    }
    __syncthreads();

    for (int q = threadIdx.x; q < num_queries; q += blockDim.x) {
      int current_label = prev_labels[q];
      int best = current_label;
      if (current_label >= 0) {
        const float* sim_row = similarity + (static_cast<long long>(b) * num_queries + q) * num_queries;
        for (int neighbor = 0; neighbor < num_queries; ++neighbor) {
          int neighbor_label = prev_labels[neighbor];
          if (neighbor_label >= 0 && sim_row[neighbor] >= duplicate_threshold) {
            best = min(best, neighbor_label);
          }
        }
        if (best < current_label) {
          atomicExch(changed, 1);
        }
      }
      next_labels[q] = best;
    }
    __syncthreads();

    int* tmp = prev_labels;
    prev_labels = next_labels;
    next_labels = tmp;
    if (*changed == 0) {
      break;
    }
    __syncthreads();
  }

  for (int q = threadIdx.x; q < num_queries; q += blockDim.x) {
    labels[offset + q] = prev_labels[q];
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

__global__ void init_pr_labels_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ seed_thresholds,
    int* __restrict__ labels,
    int num_combos,
    int num_seed_thresholds,
    int num_duplicate_thresholds,
    int num_queries) {
  long long total = static_cast<long long>(num_combos) * num_queries;
  long long linear = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (linear >= total) {
    return;
  }

  int q = linear % num_queries;
  int combo = linear / num_queries;
  int b = combo / (num_seed_thresholds * num_duplicate_thresholds);
  int rem = combo - b * num_seed_thresholds * num_duplicate_thresholds;
  int seed_idx = rem / num_duplicate_thresholds;
  float score = scores[b * num_queries + q];
  labels[linear] = score >= seed_thresholds[seed_idx] ? q : -1;
}

__global__ void propagate_pr_labels_kernel(
    const float* __restrict__ similarity,
    const float* __restrict__ duplicate_thresholds,
    int* __restrict__ labels,
    int num_combos,
    int num_seed_thresholds,
    int num_duplicate_thresholds,
    int num_queries) {
  int combo = blockIdx.x;
  if (combo >= num_combos) {
    return;
  }

  extern __shared__ int shared[];
  int* prev_labels = shared;
  int* next_labels = shared + num_queries;
  int* changed = shared + 2 * num_queries;

  int label_offset = combo * num_queries;
  for (int q = threadIdx.x; q < num_queries; q += blockDim.x) {
    prev_labels[q] = labels[label_offset + q];
  }
  __syncthreads();

  int b = combo / (num_seed_thresholds * num_duplicate_thresholds);
  int duplicate_idx = combo % num_duplicate_thresholds;
  float duplicate_threshold = duplicate_thresholds[duplicate_idx];

  for (int iter = 0; iter < num_queries; ++iter) {
    if (threadIdx.x == 0) {
      *changed = 0;
    }
    __syncthreads();

    for (int q = threadIdx.x; q < num_queries; q += blockDim.x) {
      int current_label = prev_labels[q];
      int best = current_label;
      if (current_label >= 0) {
        const float* sim_row = similarity + (static_cast<long long>(b) * num_queries + q) * num_queries;
        for (int neighbor = 0; neighbor < num_queries; ++neighbor) {
          int neighbor_label = prev_labels[neighbor];
          if (neighbor_label >= 0 && sim_row[neighbor] >= duplicate_threshold) {
            best = min(best, neighbor_label);
          }
        }
        if (best < current_label) {
          atomicExch(changed, 1);
        }
      }
      next_labels[q] = best;
    }
    __syncthreads();

    int* tmp = prev_labels;
    prev_labels = next_labels;
    next_labels = tmp;
    if (*changed == 0) {
      break;
    }
    __syncthreads();
  }

  for (int q = threadIdx.x; q < num_queries; q += blockDim.x) {
    labels[label_offset + q] = prev_labels[q];
  }
}

__global__ void mark_pr_components_kernel(
    const int* __restrict__ labels,
    const bool* __restrict__ matched,
    int* __restrict__ component_active,
    int* __restrict__ component_matched,
    int num_combos,
    int num_seed_thresholds,
    int num_duplicate_thresholds,
    int num_queries) {
  long long total = static_cast<long long>(num_combos) * num_queries;
  long long linear = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (linear >= total) {
    return;
  }

  int q = linear % num_queries;
  int combo = linear / num_queries;
  int root = labels[linear];
  if (root < 0) {
    return;
  }

  int b = combo / (num_seed_thresholds * num_duplicate_thresholds);
  int component_idx = combo * num_queries + root;
  atomicExch(component_active + component_idx, 1);
  if (matched[b * num_queries + q]) {
    atomicExch(component_matched + component_idx, 1);
  }
}

__global__ void count_total_matched_kernel(
    const bool* __restrict__ matched,
    float* __restrict__ total_matched,
    int batch_size,
    int num_queries) {
  int b = blockIdx.x;
  if (b >= batch_size) {
    return;
  }

  __shared__ int thread_counts[kThreads];
  int count = 0;
  for (int q = threadIdx.x; q < num_queries; q += blockDim.x) {
    if (matched[b * num_queries + q]) {
      ++count;
    }
  }
  thread_counts[threadIdx.x] = count;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      thread_counts[threadIdx.x] += thread_counts[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    total_matched[b] = static_cast<float>(thread_counts[0]);
  }
}

__global__ void reduce_pr_counts_kernel(
    const int* __restrict__ component_active,
    const int* __restrict__ component_matched,
    const float* __restrict__ total_matched,
    float* __restrict__ tp,
    float* __restrict__ fp,
    float* __restrict__ fn,
    int num_combos,
    int num_seed_thresholds,
    int num_duplicate_thresholds,
    int num_queries) {
  int combo = blockIdx.x;
  if (combo >= num_combos) {
    return;
  }

  __shared__ int thread_tp[kThreads];
  __shared__ int thread_fp[kThreads];
  int local_tp = 0;
  int local_fp = 0;
  int component_offset = combo * num_queries;
  for (int root = threadIdx.x; root < num_queries; root += blockDim.x) {
    int idx = component_offset + root;
    if (component_active[idx]) {
      if (component_matched[idx]) {
        ++local_tp;
      } else {
        ++local_fp;
      }
    }
  }

  thread_tp[threadIdx.x] = local_tp;
  thread_fp[threadIdx.x] = local_fp;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      thread_tp[threadIdx.x] += thread_tp[threadIdx.x + stride];
      thread_fp[threadIdx.x] += thread_fp[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    int b = combo / (num_seed_thresholds * num_duplicate_thresholds);
    float image_tp = static_cast<float>(thread_tp[0]);
    float image_fp = static_cast<float>(thread_fp[0]);
    tp[combo] = image_tp;
    fp[combo] = image_fp;
    fn[combo] = total_matched[b] - image_tp;
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

  int total_nodes = batch_size * num_queries;
  int init_blocks = (total_nodes + kThreads - 1) / kThreads;
  init_labels_kernel<<<init_blocks, kThreads, 0, stream>>>(
      selected_mask.data_ptr<bool>(),
      labels.data_ptr<int>(),
      total_nodes,
      num_queries);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  size_t shared_label_bytes = static_cast<size_t>(2 * num_queries + 1) * sizeof(int);
  propagate_labels_kernel<<<batch_size, kThreads, shared_label_bytes, stream>>>(
      selected_mask.data_ptr<bool>(),
      similarity.data_ptr<float>(),
      labels.data_ptr<int>(),
      batch_size,
      num_queries,
      static_cast<float>(duplicate_threshold));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

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

std::vector<torch::Tensor> seed_cluster_precision_recall_cuda_forward(
    const torch::Tensor& seed_scores,
    const torch::Tensor& matched_query_mask,
    const torch::Tensor& seed_thresholds,
    const torch::Tensor& duplicate_thresholds,
    const torch::Tensor& similarity) {
  TORCH_CHECK(seed_scores.is_cuda(), "seed_scores must be CUDA");
  TORCH_CHECK(matched_query_mask.is_cuda(), "matched_query_mask must be CUDA");
  TORCH_CHECK(seed_thresholds.is_cuda(), "seed_thresholds must be CUDA");
  TORCH_CHECK(duplicate_thresholds.is_cuda(), "duplicate_thresholds must be CUDA");
  TORCH_CHECK(similarity.is_cuda(), "similarity must be CUDA");
  TORCH_CHECK(seed_scores.scalar_type() == torch::kFloat, "CUDA PR expects float seed scores");
  TORCH_CHECK(matched_query_mask.scalar_type() == torch::kBool, "CUDA PR expects bool matched mask");
  TORCH_CHECK(seed_thresholds.scalar_type() == torch::kFloat, "CUDA PR expects float seed thresholds");
  TORCH_CHECK(duplicate_thresholds.scalar_type() == torch::kFloat, "CUDA PR expects float duplicate thresholds");
  TORCH_CHECK(similarity.scalar_type() == torch::kFloat, "CUDA PR expects float similarity");

  const c10::cuda::CUDAGuard device_guard(seed_scores.device());
  const int batch_size = static_cast<int>(seed_scores.size(0));
  const int num_queries = static_cast<int>(seed_scores.size(1));
  const int num_seed_thresholds = static_cast<int>(seed_thresholds.size(0));
  const int num_duplicate_thresholds = static_cast<int>(duplicate_thresholds.size(0));
  const int num_combos = batch_size * num_seed_thresholds * num_duplicate_thresholds;
  auto stream = at::cuda::getCurrentCUDAStream();

  auto labels = torch::empty(
      {num_combos, num_queries},
      seed_scores.options().dtype(torch::kInt));
  auto component_active = torch::zeros(
      {num_combos, num_queries},
      seed_scores.options().dtype(torch::kInt));
  auto component_matched = torch::zeros_like(component_active);
  auto total_matched = torch::zeros({batch_size}, seed_scores.options());
  auto flat_shape = std::vector<int64_t>{
      batch_size,
      num_seed_thresholds,
      num_duplicate_thresholds,
  };
  auto tp = torch::zeros(flat_shape, seed_scores.options());
  auto fp = torch::zeros_like(tp);
  auto fn = torch::zeros_like(tp);

  long long total_labels = static_cast<long long>(num_combos) * num_queries;
  int init_blocks = static_cast<int>((total_labels + kThreads - 1) / kThreads);
  init_pr_labels_kernel<<<init_blocks, kThreads, 0, stream>>>(
      seed_scores.data_ptr<float>(),
      seed_thresholds.data_ptr<float>(),
      labels.data_ptr<int>(),
      num_combos,
      num_seed_thresholds,
      num_duplicate_thresholds,
      num_queries);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  size_t shared_label_bytes = static_cast<size_t>(2 * num_queries + 1) * sizeof(int);
  propagate_pr_labels_kernel<<<num_combos, kThreads, shared_label_bytes, stream>>>(
      similarity.data_ptr<float>(),
      duplicate_thresholds.data_ptr<float>(),
      labels.data_ptr<int>(),
      num_combos,
      num_seed_thresholds,
      num_duplicate_thresholds,
      num_queries);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  mark_pr_components_kernel<<<init_blocks, kThreads, 0, stream>>>(
      labels.data_ptr<int>(),
      matched_query_mask.data_ptr<bool>(),
      component_active.data_ptr<int>(),
      component_matched.data_ptr<int>(),
      num_combos,
      num_seed_thresholds,
      num_duplicate_thresholds,
      num_queries);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  count_total_matched_kernel<<<batch_size, kThreads, 0, stream>>>(
      matched_query_mask.data_ptr<bool>(),
      total_matched.data_ptr<float>(),
      batch_size,
      num_queries);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  reduce_pr_counts_kernel<<<num_combos, kThreads, 0, stream>>>(
      component_active.data_ptr<int>(),
      component_matched.data_ptr<int>(),
      total_matched.data_ptr<float>(),
      tp.data_ptr<float>(),
      fp.data_ptr<float>(),
      fn.data_ptr<float>(),
      num_combos,
      num_seed_thresholds,
      num_duplicate_thresholds,
      num_queries);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

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
