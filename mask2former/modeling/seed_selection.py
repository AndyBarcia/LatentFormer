import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from .seed_cluster_metrics import compute_seed_cluster_precision_recall
from .seed_selection_ops.functions import clustering_seed_selection_native
from .similarity import pairwise_similarity


class SeedSelectionBase(nn.Module):
    requires_gt_signatures = False

    @staticmethod
    def _flatten_query_features(values: torch.Tensor, name: str) -> torch.Tensor:
        if values.dim() == 3:
            return values
        if values.dim() == 4:
            layers, batch, queries = values.shape[:3]
            return values.permute(1, 0, 2, 3).reshape(batch, layers * queries, values.shape[-1])
        raise ValueError(f"{name} must be shaped [B,Q,C] or [L,B,Q,C], got {values.shape}.")

    @staticmethod
    def _flatten_query_logits(values: torch.Tensor, name: str) -> torch.Tensor:
        if values.dim() == 2:
            return values
        if values.dim() == 3:
            layers, batch, queries = values.shape
            return values.permute(1, 0, 2).reshape(batch, layers * queries)
        raise ValueError(f"{name} must be shaped [B,Q] or [L,B,Q], got {values.shape}.")

    def _require_gt_signatures(self, gt_signatures, gt_pad_mask):
        if gt_signatures is None or gt_pad_mask is None:
            raise ValueError(
                f"{self.__class__.__name__} requires GT signatures. "
                "Set MODEL.LATENT_FORMER.TEST.LOAD_GT_FOR_EVAL=True."
            )


class ClusteringSeedSelection(SeedSelectionBase):
    """Select seed queries and merge duplicate detections with connected components."""

    _NATIVE_SUPPORTED_METRICS = {
        "centered-cosine",
        "cosine",
        "dot",
        "dot-sigmoid",
        "softmax",
    }

    def __init__(
        self,
        *,
        similarity_metric: str = "cosine",
        seed_cluster_pr_num_seed_thresholds: int = 3,
        seed_cluster_pr_num_duplicate_thresholds: int = 3,
        seed_cluster_pr_seed_threshold_range=(0.0, 1.0),
        seed_cluster_pr_duplicate_threshold_range=(0.0, 1.0),
        seed_cluster_pr_hidden_dim: int = 32,
        seed_cluster_pr_inference_num_points: int = 201,
    ):
        super().__init__()
        self.similarity_metric = similarity_metric
        self.seed_cluster_pr_num_seed_thresholds = seed_cluster_pr_num_seed_thresholds
        self.seed_cluster_pr_num_duplicate_thresholds = seed_cluster_pr_num_duplicate_thresholds
        self.seed_cluster_pr_seed_threshold_range = seed_cluster_pr_seed_threshold_range
        self.seed_cluster_pr_duplicate_threshold_range = seed_cluster_pr_duplicate_threshold_range
        self.seed_cluster_pr_inference_num_points = seed_cluster_pr_inference_num_points
        self.threshold_pr_mlp = ThresholdPrecisionRecallMLP(
            hidden_dim=seed_cluster_pr_hidden_dim,
        )

    def _sample_seed_cluster_thresholds(self, device):
        seed_min, seed_max = self.seed_cluster_pr_seed_threshold_range
        duplicate_min, duplicate_max = self.seed_cluster_pr_duplicate_threshold_range
        seed_thresholds = torch.empty(
            self.seed_cluster_pr_num_seed_thresholds,
            device=device,
        ).uniform_(float(seed_min), float(seed_max))
        duplicate_thresholds = torch.empty(
            self.seed_cluster_pr_num_duplicate_thresholds,
            device=device,
        ).uniform_(float(duplicate_min), float(duplicate_max))
        return seed_thresholds, duplicate_thresholds

    def threshold_pr_loss(
        self,
        *,
        query_signatures: torch.Tensor,
        query_seed_logits: torch.Tensor,
        matched_query_mask: torch.Tensor,
        matched_gt_indices: torch.Tensor,
    ) -> torch.Tensor:
        seed_thresholds, duplicate_thresholds = self._sample_seed_cluster_thresholds(
            query_signatures.device
        )
        if seed_thresholds.numel() == 0 or duplicate_thresholds.numel() == 0:
            return query_signatures.sum() * 0.0

        seed_cluster_metrics = compute_seed_cluster_precision_recall(
            query_signatures=query_signatures,
            query_seed_logits=query_seed_logits,
            matched_query_mask=matched_query_mask,
            matched_gt_indices=matched_gt_indices,
            seed_threshold=seed_thresholds,
            duplicate_threshold=duplicate_thresholds,
            metric=self.similarity_metric,
        )
        seed_cluster_pr_pred = self.threshold_pr_mlp(
            seed_cluster_metrics["seed_thresholds"],
            seed_cluster_metrics["duplicate_thresholds"],
        )
        seed_cluster_pr_target = torch.stack(
            (
                seed_cluster_metrics["micro_precision"],
                seed_cluster_metrics["micro_recall"],
            ),
            dim=-1,
        ).to(device=seed_cluster_pr_pred.device, dtype=seed_cluster_pr_pred.dtype)
        return F.mse_loss(seed_cluster_pr_pred, seed_cluster_pr_target.detach())

    def best_thresholds(self) -> tuple[float, float]:
        num_points = int(self.seed_cluster_pr_inference_num_points)
        if num_points <= 0:
            raise ValueError(
                "seed_cluster_pr_inference_num_points must be positive when selecting "
                "clustering seed thresholds."
            )

        seed_min, seed_max = self.seed_cluster_pr_seed_threshold_range
        duplicate_min, duplicate_max = self.seed_cluster_pr_duplicate_threshold_range
        device = next(self.threshold_pr_mlp.parameters()).device
        seed_thresholds = torch.linspace(float(seed_min), float(seed_max), num_points, device=device)
        duplicate_thresholds = torch.linspace(
            float(duplicate_min),
            float(duplicate_max),
            num_points,
            device=device,
        )
        with torch.no_grad():
            pr = self.threshold_pr_mlp(seed_thresholds, duplicate_thresholds)
            precision = pr[..., 0]
            recall = pr[..., 1]
            rq = 2.0 * precision * recall / (precision + recall).clamp_min(1e-6)
            best_idx = rq.reshape(-1).argmax()
            duplicate_idx = best_idx % duplicate_thresholds.numel()
            seed_idx = best_idx // duplicate_thresholds.numel()

        return (
            float(seed_thresholds[seed_idx].detach().cpu()),
            float(duplicate_thresholds[duplicate_idx].detach().cpu()),
        )

    @staticmethod
    def _connected_components(adjacency: torch.Tensor) -> list[list[int]]:
        num_nodes = adjacency.shape[0]
        visited = torch.zeros(num_nodes, dtype=torch.bool, device=adjacency.device)
        components = []

        for node_idx in range(num_nodes):
            if visited[node_idx]:
                continue

            stack = [node_idx]
            visited[node_idx] = True
            component = []

            while stack:
                current = stack.pop()
                component.append(current)
                neighbors = adjacency[current].nonzero(as_tuple=False).flatten().tolist()
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)

            components.append(component)

        return components

    def forward(
        self,
        query_signatures: torch.Tensor,
        query_seed_logits: torch.Tensor,
        *,
        gt_signatures: torch.Tensor = None,
        gt_pad_mask: torch.Tensor = None,
        seed_threshold: float = None,
        duplicate_threshold: float = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del gt_signatures, gt_pad_mask
        if seed_threshold is None or duplicate_threshold is None:
            seed_threshold, duplicate_threshold = self.best_thresholds()

        q_sig_flat = self._flatten_query_features(query_signatures, "query_signatures")
        q_seed_logits_flat = self._flatten_query_logits(query_seed_logits, "query_seed_logits").float()

        if self.similarity_metric.lower() in self._NATIVE_SUPPORTED_METRICS:
            native_result = clustering_seed_selection_native(
                q_sig_flat,
                q_seed_logits_flat,
                seed_threshold=seed_threshold,
                duplicate_threshold=duplicate_threshold,
                similarity_metric=self.similarity_metric,
            )
            if native_result is not None:
                return native_result

        q_seed_scores = q_seed_logits_flat.sigmoid()

        selected_mask = q_seed_scores >= seed_threshold
        empty_selection = ~selected_mask.any(dim=1)
        if empty_selection.any():
            fallback_indices = q_seed_scores.argmax(dim=1)
            selected_mask = selected_mask.clone()
            selected_mask[empty_selection, fallback_indices[empty_selection]] = True

        similarity = pairwise_similarity(
            q_sig_flat,
            q_sig_flat,
            metric=self.similarity_metric,
        )
        adjacency = similarity >= duplicate_threshold
        adjacency = adjacency & selected_mask[:, :, None] & selected_mask[:, None, :]
        query_indices = torch.arange(q_sig_flat.shape[1], device=q_sig_flat.device)
        adjacency[:, query_indices, query_indices] = selected_mask

        batch_seed_signatures = []
        batch_seed_scores = []
        max_num_seeds = 0

        for signatures_per_image, seed_scores_per_image, selected, adjacency_per_image in zip(
            q_sig_flat,
            q_seed_scores,
            selected_mask,
            adjacency,
        ):
            selected = selected.nonzero(as_tuple=False).flatten()
            selected_adjacency = adjacency_per_image[selected][:, selected]
            components = self._connected_components(selected_adjacency)
            component_seed_signatures = []
            component_seed_scores = []
            for component in components:
                component_indices = selected.new_tensor(component, dtype=torch.long)
                component_query_indices = selected[component_indices]
                component_scores = seed_scores_per_image[component_query_indices]
                best_idx = component_query_indices[component_scores.argmax()]
                component_seed_signatures.append(signatures_per_image[best_idx])
                component_seed_scores.append(seed_scores_per_image[best_idx])

            image_seed_signatures = torch.stack(component_seed_signatures, dim=0)
            image_seed_scores = torch.stack(component_seed_scores, dim=0)
            batch_seed_signatures.append(image_seed_signatures)
            batch_seed_scores.append(image_seed_scores)
            max_num_seeds = max(max_num_seeds, image_seed_signatures.shape[0])

        batch_size = q_sig_flat.shape[0]
        sig_dim = q_sig_flat.shape[-1]
        seed_signatures = q_sig_flat.new_zeros((batch_size, max_num_seeds, sig_dim))
        seed_scores = q_seed_scores.new_zeros((batch_size, max_num_seeds))
        pad_mask = torch.zeros((batch_size, max_num_seeds), dtype=torch.bool, device=q_sig_flat.device)

        for batch_idx, (image_seed_signatures, image_seed_scores) in enumerate(
            zip(batch_seed_signatures, batch_seed_scores)
        ):
            num_seeds = image_seed_signatures.shape[0]
            seed_signatures[batch_idx, :num_seeds] = image_seed_signatures
            seed_scores[batch_idx, :num_seeds] = image_seed_scores
            pad_mask[batch_idx, :num_seeds] = True

        return seed_signatures, pad_mask, seed_scores


SeedSelection = ClusteringSeedSelection


class ThresholdPrecisionRecallMLP(nn.Module):
    """Predict seed-cluster precision/recall from a threshold pair."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

    def forward(
        self,
        seed_thresholds: torch.Tensor,
        duplicate_thresholds: torch.Tensor,
    ) -> torch.Tensor:
        seed_thresholds = seed_thresholds.to(dtype=torch.float32)
        duplicate_thresholds = duplicate_thresholds.to(
            device=seed_thresholds.device,
            dtype=torch.float32,
        )
        seed_grid, duplicate_grid = torch.meshgrid(
            seed_thresholds,
            duplicate_thresholds,
            indexing="ij",
        )
        threshold_pairs = torch.stack((seed_grid, duplicate_grid), dim=-1)
        return self.net(threshold_pairs)


class GoldenSeedSelection(SeedSelectionBase):
    """Use one-to-one matching to choose the closest predicted query for each GT object."""

    requires_gt_signatures = True

    def __init__(self, *, similarity_metric: str = "cosine"):
        super().__init__()
        self.similarity_metric = similarity_metric

    def forward(
        self,
        query_signatures: torch.Tensor,
        query_seed_logits: torch.Tensor,
        gt_signatures: torch.Tensor = None,
        gt_pad_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._require_gt_signatures(gt_signatures, gt_pad_mask)
        q_sig_flat = self._flatten_query_features(query_signatures, "query_signatures")
        q_seed_logits_flat = self._flatten_query_logits(query_seed_logits, "query_seed_logits").float()
        gt_pad_mask = gt_pad_mask.to(device=gt_signatures.device)

        sim = pairwise_similarity(
            q_sig_flat,
            gt_signatures,
            metric=self.similarity_metric,
        )
        seed_signatures = q_sig_flat.new_zeros(gt_signatures.shape)
        seed_scores = q_seed_logits_flat.new_zeros(gt_pad_mask.shape)

        for batch_idx in range(q_sig_flat.shape[0]):
            valid_gt_idx = gt_pad_mask[batch_idx].nonzero(as_tuple=False).flatten()
            if valid_gt_idx.numel() == 0:
                continue

            cost = 1.0 - sim[batch_idx, :, valid_gt_idx]
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            if len(row_ind) == 0:
                continue

            query_idx = torch.as_tensor(row_ind, device=q_sig_flat.device, dtype=torch.long)
            gt_idx = valid_gt_idx[
                torch.as_tensor(col_ind, device=q_sig_flat.device, dtype=torch.long)
            ]
            seed_signatures[batch_idx, gt_idx] = q_sig_flat[batch_idx, query_idx]
            seed_scores[batch_idx, gt_idx] = q_seed_logits_flat[batch_idx, query_idx].sigmoid()

        return seed_signatures, gt_pad_mask, seed_scores


class GTOracleSeedSelection(SeedSelectionBase):
    """Use encoded GT signatures directly as seeds."""

    requires_gt_signatures = True

    def forward(
        self,
        query_signatures: torch.Tensor,
        query_seed_logits: torch.Tensor,
        gt_signatures: torch.Tensor = None,
        gt_pad_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del query_signatures, query_seed_logits
        self._require_gt_signatures(gt_signatures, gt_pad_mask)
        gt_pad_mask = gt_pad_mask.to(device=gt_signatures.device)
        return gt_signatures, gt_pad_mask, gt_pad_mask.to(dtype=gt_signatures.dtype)


def build_seed_selection_modules(
    *,
    eval_modes,
    similarity_metric: str = "cosine",
    seed_cluster_pr_num_seed_thresholds: int = 3,
    seed_cluster_pr_num_duplicate_thresholds: int = 3,
    seed_cluster_pr_seed_threshold_range=(0.0, 1.0),
    seed_cluster_pr_duplicate_threshold_range=(0.0, 1.0),
    seed_cluster_pr_hidden_dim: int = 32,
    seed_cluster_pr_inference_num_points: int = 201,
):
    modules = nn.ModuleDict()
    for mode in eval_modes:
        if mode == "ClusteringSeedSelection":
            modules[mode] = ClusteringSeedSelection(
                similarity_metric=similarity_metric,
                seed_cluster_pr_num_seed_thresholds=seed_cluster_pr_num_seed_thresholds,
                seed_cluster_pr_num_duplicate_thresholds=seed_cluster_pr_num_duplicate_thresholds,
                seed_cluster_pr_seed_threshold_range=seed_cluster_pr_seed_threshold_range,
                seed_cluster_pr_duplicate_threshold_range=seed_cluster_pr_duplicate_threshold_range,
                seed_cluster_pr_hidden_dim=seed_cluster_pr_hidden_dim,
                seed_cluster_pr_inference_num_points=seed_cluster_pr_inference_num_points,
            )
        elif mode == "GoldenSeedSelection":
            modules[mode] = GoldenSeedSelection(similarity_metric=similarity_metric)
        elif mode == "GTOracleSeedSelection":
            modules[mode] = GTOracleSeedSelection()
        else:
            raise ValueError(f"Unsupported LatentFormer eval mode: {mode}")
    return modules
