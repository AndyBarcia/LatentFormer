import torch
from torch import nn

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

    def __init__(
        self,
        *,
        seed_threshold: float,
        duplicate_threshold: float,
        similarity_metric: str = "cosine",
    ):
        super().__init__()
        self.seed_threshold = seed_threshold
        self.duplicate_threshold = duplicate_threshold
        self.similarity_metric = similarity_metric

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
        gt_signatures: torch.Tensor = None,
        gt_pad_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del gt_signatures, gt_pad_mask
        q_sig_flat = self._flatten_query_features(query_signatures, "query_signatures")
        q_seed_logits_flat = self._flatten_query_logits(query_seed_logits, "query_seed_logits").float()
        q_seed_scores = q_seed_logits_flat.sigmoid()

        selected_mask = q_seed_scores >= self.seed_threshold
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
        adjacency = similarity >= self.duplicate_threshold
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


class GoldenSeedSelection(SeedSelectionBase):
    """Use GT signatures to choose the closest predicted query for each GT object."""

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
        sim = sim.masked_fill(~gt_pad_mask[:, None, :], float("-inf"))
        best_query = sim.argmax(dim=1)
        gather_idx = best_query[:, :, None].expand(-1, -1, q_sig_flat.shape[-1])
        seed_signatures = q_sig_flat.gather(1, gather_idx) * gt_pad_mask[:, :, None].to(
            dtype=q_sig_flat.dtype
        )
        seed_scores = q_seed_logits_flat.sigmoid().gather(1, best_query).masked_fill(~gt_pad_mask, 0.0)
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
    seed_threshold: float,
    duplicate_threshold: float,
    similarity_metric: str = "cosine",
):
    modules = nn.ModuleDict()
    for mode in eval_modes:
        if mode == "ClusteringSeedSelection":
            modules[mode] = ClusteringSeedSelection(
                seed_threshold=seed_threshold,
                duplicate_threshold=duplicate_threshold,
                similarity_metric=similarity_metric,
            )
        elif mode == "GoldenSeedSelection":
            modules[mode] = GoldenSeedSelection(similarity_metric=similarity_metric)
        elif mode == "GTOracleSeedSelection":
            modules[mode] = GTOracleSeedSelection()
        else:
            raise ValueError(f"Unsupported LatentFormer eval mode: {mode}")
    return modules
