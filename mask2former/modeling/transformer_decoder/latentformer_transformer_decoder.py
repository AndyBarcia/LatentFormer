# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List, Optional

import fvcore.nn.weight_init as weight_init
import torch
from torch import Tensor, nn

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .mask2former_transformer_decoder import (
    CrossAttentionLayer,
    FFNLayer,
    MLP,
    SelfAttentionLayer,
)
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from .position_encoding import PositionEmbeddingSine


def build_latentformer_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build the LatentFormer transformer decoder from ``cfg.MODEL.LATENT_FORMER``.
    """
    name = cfg.MODEL.LATENT_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


class AttnResidualDecoder(nn.Module):
    """Depth-wise attention residual mixer around a stack of decoder blocks."""

    def __init__(
        self,
        decoder_layers: nn.ModuleList,
        hidden_dim: int,
        use_attention_residuals: bool = True,
    ):
        super().__init__()
        self.layers = decoder_layers
        self.num_layers = len(decoder_layers)
        self.use_attention_residuals = use_attention_residuals

        self.attn_residual_queries = nn.Parameter(
            torch.randn(self.num_layers, hidden_dim) * (hidden_dim**-0.5)
        )
        self.attn_residual_key_scale = nn.Parameter(torch.ones(hidden_dim))
        self.attn_residual_eps = 1e-6

    def _depth_attention_residual(self, history: List[Tensor], layer_idx: int) -> Tensor:
        if not self.use_attention_residuals or len(history) == 1:
            return history[-1]

        history_stack = torch.stack(history, dim=0)  # D x B x Q x C
        rms = history_stack.pow(2).mean(dim=-1, keepdim=True).add(self.attn_residual_eps).rsqrt()
        keys = history_stack * rms * self.attn_residual_key_scale.view(1, 1, 1, -1)
        logits = torch.einsum("c,dbqc->dbq", self.attn_residual_queries[layer_idx], keys)
        weights = logits.softmax(dim=0)
        return torch.einsum("dbq,dbqc->bqc", weights, history_stack)

    def forward_layer(
        self,
        history: List[Tensor],
        layer_idx: int,
        memory: Tensor,
        memory_mask: Optional[Tensor],
        pos: Tensor,
        query_pos: Tensor,
    ) -> Tensor:
        output = self._depth_attention_residual(history, layer_idx).transpose(0, 1)
        layer = self.layers[layer_idx]

        output = layer["cross_attn"](
            output,
            memory,
            memory_mask=memory_mask,
            memory_key_padding_mask=None,
            pos=pos,
            query_pos=query_pos,
        )
        output = layer["self_attn"](
            output,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=query_pos,
        )
        output = layer["ffn"](output)
        return output.transpose(0, 1)


@TRANSFORMER_DECODER_REGISTRY.register()
class LatentTransformerDecoder(nn.Module):
    """Multi-scale LatentFormer transformer decoder."""

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        sig_dim: int,
        num_queries: int,
        nheads: int,
        dropout: float,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        num_feature_levels: int,
        enforce_input_project: bool,
        use_attention_residuals: bool = True,
    ):
        super().__init__()
        assert mask_classification, "LatentFormer always uses a class head."
        assert num_feature_levels >= 1, "LatentFormer requires at least one feature level."

        self.mask_classification = mask_classification
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.attn_bias_eps = 1e-6

        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        decoder_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            decoder_layers.append(
                nn.ModuleDict(
                    {
                        "self_attn": SelfAttentionLayer(
                            d_model=hidden_dim,
                            nhead=nheads,
                            dropout=dropout,
                            normalize_before=pre_norm,
                        ),
                        "cross_attn": CrossAttentionLayer(
                            d_model=hidden_dim,
                            nhead=nheads,
                            dropout=dropout,
                            normalize_before=pre_norm,
                        ),
                        "ffn": FFNLayer(
                            d_model=hidden_dim,
                            dim_feedforward=dim_feedforward,
                            dropout=dropout,
                            normalize_before=pre_norm,
                        ),
                    }
                )
            )
        self.decoder = AttnResidualDecoder(
            decoder_layers,
            hidden_dim=hidden_dim,
            use_attention_residuals=use_attention_residuals,
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, in_channels, 3)
        self.sig_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, sig_dim),
        )
        self.seed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.LATENT_FORMER.HIDDEN_DIM
        ret["sig_dim"] = cfg.MODEL.LATENT_FORMER.GT_ENCODER.SIG_DIM
        ret["num_queries"] = cfg.MODEL.LATENT_FORMER.NUM_OBJECT_QUERIES
        ret["nheads"] = cfg.MODEL.LATENT_FORMER.NHEADS
        ret["dropout"] = cfg.MODEL.LATENT_FORMER.DROPOUT
        ret["dim_feedforward"] = cfg.MODEL.LATENT_FORMER.DIM_FEEDFORWARD
        ret["dec_layers"] = cfg.MODEL.LATENT_FORMER.DEC_LAYERS
        ret["pre_norm"] = cfg.MODEL.LATENT_FORMER.PRE_NORM
        ret["num_feature_levels"] = cfg.MODEL.SEM_SEG_HEAD.get("LATENT_FPN_NUM_LEVELS", 3)
        ret["enforce_input_project"] = cfg.MODEL.LATENT_FORMER.ENFORCE_INPUT_PROJ
        return ret

    def forward(self, multi_scale_features, mask=None):
        assert len(multi_scale_features) == self.num_feature_levels
        del mask

        src = []
        pos = []
        for i, feature in enumerate(multi_scale_features):
            pos_i = self.pe_layer(feature, None).flatten(2).permute(2, 0, 1)
            src_i = self.input_proj[i](feature).flatten(2)
            src_i = src_i + self.level_embed.weight[i][None, :, None]
            src_i = src_i.permute(2, 0, 1)

            pos.append(pos_i)
            src.append(src_i)

        _, bs, _ = src[0].shape
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(0).repeat(bs, 1, 1)
        history = [output]

        predictions_class = []
        predictions_mask = []
        predictions_sig = []
        predictions_seed = []

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_bias = attn_biases[level_index]

            output = self.decoder.forward_layer(
                history,
                i,
                memory=src[level_index],
                memory_mask=attn_bias,
                pos=pos[level_index],
                query_pos=query_pos,
            )
            history.append(output)

            outputs_class, outputs_mask, outputs_sig, outputs_seed, attn_biases = (
                self.forward_prediction_heads(output, multi_scale_features)
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_sig.append(outputs_sig)
            predictions_seed.append(outputs_seed)

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "pred_signatures": predictions_sig[-1],
            "pred_seed_logits": predictions_seed[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class,
                predictions_mask,
                predictions_sig,
                predictions_seed,
            ),
        }
        return out

    def forward_prediction_heads(self, output, multi_scale_features):
        decoder_output = self.decoder_norm(output)
        outputs_class = self.class_embed(decoder_output)
        outputs_sig = self.sig_head(decoder_output)
        outputs_seed = self.seed_head(decoder_output).squeeze(-1)

        attn_biases = []
        mask_embed = self.mask_embed(decoder_output)
        for feature in multi_scale_features:
            output_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, feature)
            
            attn_bias = output_mask.flatten(2)
            attn_bias = attn_bias - attn_bias.mean(dim=-1, keepdim=True)
            attn_bias = attn_bias / attn_bias.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.attn_bias_eps)
            attn_bias = (
                attn_bias.unsqueeze(1)
                .repeat(1, self.num_heads, 1, 1)
                .flatten(0, 1)
            )
            attn_biases.append(attn_bias)

        return outputs_class, mask_embed, outputs_sig, outputs_seed, attn_biases

    @torch.jit.unused
    def _set_aux_loss(
        self,
        outputs_class,
        outputs_seg_masks,
        outputs_sig,
        outputs_seed,
    ):
        return [
            {
                "pred_logits": a,
                "pred_masks": b,
                "pred_signatures": c,
                "pred_seed_logits": d,
            }
            for a, b, c, d in zip(
                outputs_class[:-1],
                outputs_seg_masks[:-1],
                outputs_sig[:-1],
                outputs_seed[:-1],
            )
        ]
