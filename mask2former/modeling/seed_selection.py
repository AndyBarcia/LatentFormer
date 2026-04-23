import torch
from torch import nn

from .similarity import pairwise_similarity


class SeedSelection(nn.Module):
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_sig_flat = self._flatten_query_features(query_signatures, "query_signatures")
        q_seed_logits_flat = self._flatten_query_logits(query_seed_logits, "query_seed_logits").float()
        q_seed_scores = q_seed_logits_flat.sigmoid()

        batch_seed_signatures = []
        batch_seed_scores = []
        max_num_seeds = 0

        for signatures_per_image, seed_scores_per_image in zip(q_sig_flat, q_seed_scores):
            selected = torch.nonzero(
                seed_scores_per_image >= self.seed_threshold,
                as_tuple=False,
            ).flatten()
            if selected.numel() == 0:
                selected = seed_scores_per_image.argmax().view(1)

            selected_signatures = signatures_per_image[selected]
            selected_scores = seed_scores_per_image[selected]

            similarity = pairwise_similarity(
                selected_signatures.unsqueeze(0),
                selected_signatures.unsqueeze(0),
                metric=self.similarity_metric,
            ).squeeze(0)
            adjacency = similarity >= self.duplicate_threshold
            adjacency.fill_diagonal_(True)

            components = self._connected_components(adjacency)
            component_seed_signatures = []
            component_seed_scores = []
            for component in components:
                component_indices = selected.new_tensor(component, dtype=torch.long)
                component_scores = selected_scores[component_indices]
                best_idx = component_indices[component_scores.argmax()]
                component_seed_signatures.append(selected_signatures[best_idx])
                component_seed_scores.append(selected_scores[best_idx])

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
