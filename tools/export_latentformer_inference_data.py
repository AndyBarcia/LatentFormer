#!/usr/bin/env python
"""Export compact LatentFormer inference fixtures for notebook experiments."""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.structures import ImageList

from mask2former.latentformer_model import LatentAggregator
from mask2former.utils.padding import image_padding_mask
from train_net import Trainer, setup


def _resolve_run_dir(run_dir):
    path = Path(run_dir)
    if path.exists():
        return path
    parts = path.parts
    if len(parts) > 1:
        candidate = Path("outputs") / parts[-1]
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve run directory: {run_dir}")


def _latest_checkpoint(run_dir):
    last_checkpoint = run_dir / "last_checkpoint"
    if last_checkpoint.is_file():
        name = last_checkpoint.read_text().strip()
        if name:
            checkpoint = Path(name)
            if checkpoint.is_absolute():
                return checkpoint
            candidate = run_dir / name
            if candidate.is_file():
                return candidate
    checkpoints = sorted(run_dir.glob("model_*.pth"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pth checkpoint found in {run_dir}")
    return checkpoints[-1]


def _default_dataset_override(config_file, explicit_dataset):
    if explicit_dataset:
        return explicit_dataset
    with open(config_file) as handle:
        config = yaml.safe_load(handle) or {}
    test_datasets = config.get("DATASETS", {}).get("TEST", ())
    if not test_datasets:
        return None
    dataset_name = test_datasets[0]
    match = re.match(r"^(.*)_dev_subset_\d+$", dataset_name)
    if match:
        return match.group(1)
    return None


def _cpu_detach(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_detach(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_detach(item) for item in value)
    return value


def _instances_to_dict(instances):
    if instances is None:
        return None
    data = {"image_size": tuple(instances.image_size)}
    for field, value in instances.get_fields().items():
        if torch.is_tensor(value):
            data[field] = value.detach().cpu()
        elif hasattr(value, "tensor"):
            data[field] = value.tensor.detach().cpu()
        else:
            data[field] = value
    return data


def _prediction_to_dict(prediction):
    converted = {}
    if "sem_seg" in prediction:
        converted["sem_seg"] = prediction["sem_seg"].detach().cpu()
        converted["sem_seg_labels"] = prediction["sem_seg"].argmax(dim=0).detach().cpu()
    if "panoptic_seg" in prediction:
        panoptic_seg, segments_info = prediction["panoptic_seg"]
        converted["panoptic_seg"] = (panoptic_seg.detach().cpu(), segments_info)
    if "instances" in prediction:
        converted["instances"] = _instances_to_dict(prediction["instances"])
    if "latentformer_signature_eval" in prediction:
        converted["latentformer_signature_eval"] = _cpu_detach(
            prediction["latentformer_signature_eval"]
        )
    return converted


def _slice_batch(value, idx, batch_size):
    if not torch.is_tensor(value):
        return value
    if value.dim() >= 2 and value.shape[1] == batch_size:
        return value[:, idx : idx + 1].clone()
    if value.dim() >= 1 and value.shape[0] == batch_size:
        return value[idx : idx + 1].clone()
    return value


def _selection_state(model, seed_selection, outputs, targets, mode):
    gt_signatures = outputs.get("gt_mask_signatures")
    gt_pad_mask = targets["pad_mask"] if targets is not None else None
    seed_signatures, seed_pad_mask, seed_scores = seed_selection(
        outputs["pred_mask_signatures"],
        outputs["pred_mask_seed_logits"],
        gt_signatures=gt_signatures,
        gt_pad_mask=gt_pad_mask,
        policy="mask",
    )

    if mode == "ClusteringSeedSelection":
        class_seed_signatures, class_seed_pad_mask, class_seed_scores = (
            model.clustering_class_seed_selection(
                seed_selection,
                outputs["pred_class_signatures"],
                outputs["pred_class_seed_logits"],
                outputs["class_signatures"],
            )
        )
    else:
        gt_class_signatures = outputs.get("gt_semantic_class_signatures")
        gt_class_pad_mask = targets.get("semantic_pad_mask") if targets is not None else None
        class_seed_signatures, class_seed_pad_mask, class_seed_scores = seed_selection(
            outputs["pred_class_signatures"],
            outputs["pred_class_seed_logits"],
            gt_signatures=gt_class_signatures,
            gt_pad_mask=gt_class_pad_mask,
            policy="class",
        )

    class_seed_labels = model.class_labels_from_signatures(
        class_seed_signatures,
        class_seed_pad_mask,
        outputs["class_signatures"],
    )
    state = {
        "mask_seed_signatures": seed_signatures.detach().cpu(),
        "mask_seed_pad_mask": seed_pad_mask.detach().cpu(),
        "mask_seed_scores": seed_scores.detach().cpu(),
        "num_mask_seeds": seed_pad_mask.detach().cpu().sum(dim=1),
        "class_seed_signatures": class_seed_signatures.detach().cpu(),
        "class_seed_pad_mask": class_seed_pad_mask.detach().cpu(),
        "class_seed_labels": class_seed_labels.detach().cpu(),
        "class_seed_scores": class_seed_scores.detach().cpu(),
        "num_class_seeds": class_seed_pad_mask.detach().cpu().sum(dim=1),
    }
    if mode == "ClusteringSeedSelection" and hasattr(seed_selection, "best_thresholds"):
        state["best_thresholds"] = seed_selection.best_thresholds()
    return state


def _capture_model_state(model, batched_inputs):
    images = [x["image"].to(model.device) for x in batched_inputs]
    normalized = [(x - model.pixel_mean) / model.pixel_std for x in images]

    image_list = ImageList.from_tensors(normalized, model.size_divisibility)
    features = model.backbone(image_list.tensor)
    outputs = model.sem_seg_head(features, mask=image_padding_mask(image_list))
    outputs["class_signatures"] = model.gt_encoder.all_class_signatures()

    targets = None
    if model._needs_gt_signatures():
        targets = model._prepare_gt_signatures(batched_inputs, image_list, outputs)

    predictions = model.inference(
        outputs,
        batched_inputs,
        image_list.image_sizes,
        padded_image_size=image_list.tensor.shape[-2:],
        targets=targets,
    )

    query_mask_signatures = LatentAggregator._flatten_layer_queries(
        outputs["pred_mask_signatures"], "pred_mask_signatures"
    )
    query_mask_seed_scores = LatentAggregator._flatten_layer_logits(
        outputs["pred_mask_seed_logits"], "pred_mask_seed_logits"
    ).sigmoid()
    query_class_signatures = LatentAggregator._flatten_layer_queries(
        outputs["pred_class_signatures"], "pred_class_signatures"
    )
    query_class_seed_scores = LatentAggregator._flatten_layer_logits(
        outputs["pred_class_seed_logits"], "pred_class_seed_logits"
    ).sigmoid()

    modes = {
        mode: _selection_state(model, seed_selection, outputs, targets, mode)
        for mode, seed_selection in model.seed_selection_modules.items()
    }

    return {
        "image_sizes": list(image_list.image_sizes),
        "padded_image_size": tuple(image_list.tensor.shape[-2:]),
        "outputs": _cpu_detach(outputs),
        "targets": _cpu_detach(targets),
        "query_mask_signatures": query_mask_signatures.detach().cpu(),
        "query_mask_seed_scores": query_mask_seed_scores.detach().cpu(),
        "query_class_signatures": query_class_signatures.detach().cpu(),
        "query_class_seed_scores": query_class_seed_scores.detach().cpu(),
        "modes": modes,
        "predictions": _cpu_detach(predictions),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a few validation images once and save all tensors needed to rerun "
            "LatentFormer seed selection, aggregation, and post-processing in a notebook."
        )
    )
    parser.add_argument(
        "--run-dir",
        default="outputs/latentformer_R50_no_seed_hungarian_pattern_loss_seed_masking_pr_prediction_fixed",
        help="Training output directory, or a host/name ending in the output directory name.",
    )
    parser.add_argument("--config-file", default=None, help="Config file. Defaults to <run-dir>/config.yaml.")
    parser.add_argument("--weights", default=None, help="Checkpoint. Defaults to last/latest in run dir.")
    parser.add_argument("--output", default=None, help="Output .pt file.")
    parser.add_argument("--num-images", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", default=None, help="Dataset name. Defaults to cfg.DATASETS.TEST[0].")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = _resolve_run_dir(args.run_dir)
    config_file = Path(args.config_file) if args.config_file else run_dir / "config.yaml"
    weights = Path(args.weights) if args.weights else _latest_checkpoint(run_dir)
    output = Path(args.output) if args.output else run_dir / "latentformer_inference_export.pt"
    dataset_override = _default_dataset_override(config_file, args.dataset)

    cfg_opts = [
        "OUTPUT_DIR",
        str(run_dir / "export_setup"),
        "MODEL.WEIGHTS",
        str(weights),
        "MODEL.DEVICE",
        args.device,
        "MODEL.LATENT_FORMER.TEST.LOAD_GT_FOR_EVAL",
        "True",
        "TEST.EVAL_MAX_IMAGES",
        str(max(args.num_images, 1)),
        "TEST.IMS_PER_BATCH",
        str(args.batch_size),
        "DATALOADER.NUM_WORKERS",
        "0",
    ]
    if dataset_override:
        cfg_opts.extend(["DATASETS.TEST", str((dataset_override,))])

    cfg_args = argparse.Namespace(
        config_file=str(config_file),
        eval_only=True,
        resume=False,
        num_gpus=1,
        num_machines=1,
        machine_rank=0,
        dist_url="tcp://127.0.0.1:49152",
        opts=cfg_opts,
    )
    cfg = setup(cfg_args)
    dataset_name = cfg.DATASETS.TEST[0]

    model = build_model(cfg)
    DetectionCheckpointer(model).load(str(weights))
    model.eval()

    loader = Trainer.build_test_loader(cfg, dataset_name)
    examples = []
    exported = 0
    with torch.no_grad():
        for batched_inputs in loader:
            if exported >= args.num_images:
                break
            batched_inputs = batched_inputs[: args.num_images - exported]
            state = _capture_model_state(model, batched_inputs)
            batch_size = len(batched_inputs)
            for idx, input_per_image in enumerate(batched_inputs):
                record = {
                    "file_name": input_per_image.get("file_name"),
                    "image_id": input_per_image.get("image_id"),
                    "height": input_per_image.get("height"),
                    "width": input_per_image.get("width"),
                    "raw_image": input_per_image["image"].detach().cpu(),
                    "sem_seg": input_per_image.get("sem_seg").detach().cpu()
                    if torch.is_tensor(input_per_image.get("sem_seg"))
                    else None,
                    "instances": _instances_to_dict(input_per_image.get("instances")),
                }
                per_image_state = {
                    "image_size": state["image_sizes"][idx],
                    "padded_image_size": state["padded_image_size"],
                    "outputs": {
                        key: _slice_batch(value, idx, batch_size)
                        for key, value in state["outputs"].items()
                    },
                    "targets": {
                        key: value[idx : idx + 1].clone()
                        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == batch_size
                        else value
                        for key, value in (state["targets"] or {}).items()
                    },
                    "query_mask_signatures": state["query_mask_signatures"][idx].clone(),
                    "query_mask_seed_scores": state["query_mask_seed_scores"][idx].clone(),
                    "query_class_signatures": state["query_class_signatures"][idx].clone(),
                    "query_class_seed_scores": state["query_class_seed_scores"][idx].clone(),
                    "modes": {
                        mode: {
                            key: value[idx : idx + 1].clone()
                            if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == batch_size
                            else value
                            for key, value in mode_state.items()
                        }
                        for mode, mode_state in state["modes"].items()
                    },
                    "predictions": {
                        mode: _prediction_to_dict(mode_predictions[idx])
                        for mode, mode_predictions in state["predictions"].items()
                    }
                    if isinstance(state["predictions"], dict)
                    else _prediction_to_dict(state["predictions"][idx]),
                }
                examples.append({"record": record, "state": per_image_state})
            exported += len(batched_inputs)

    artifact = {
        "format": "latentformer-inference-export-v2",
        "run_dir": str(run_dir),
        "config_file": str(config_file),
        "weights": str(weights),
        "dataset_name": dataset_name,
        "eval_modes": tuple(cfg.MODEL.LATENT_FORMER.TEST.EVAL_MODES),
        "metadata": {
            "thing_classes": list(getattr(model.metadata, "thing_classes", [])),
            "stuff_classes": list(getattr(model.metadata, "stuff_classes", [])),
            "thing_dataset_id_to_contiguous_id": dict(
                getattr(model.metadata, "thing_dataset_id_to_contiguous_id", {})
            ),
            "stuff_dataset_id_to_contiguous_id": dict(
                getattr(model.metadata, "stuff_dataset_id_to_contiguous_id", {})
            ),
            "ignore_label": getattr(model.sem_seg_head, "ignore_value", None),
            "num_classes": model.sem_seg_head.num_classes,
        },
        "model_settings": {
            "semantic_on": model.semantic_on,
            "panoptic_on": model.panoptic_on,
            "instance_on": model.instance_on,
            "sem_seg_postprocess_before_inference": model.sem_seg_postprocess_before_inference,
            "score_threshold": model.score_threshold,
            "overlap_threshold": model.overlap_threshold,
        },
        "examples": examples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)

    summary = {
        "output": str(output),
        "num_examples": len(examples),
        "weights": str(weights),
        "dataset_name": dataset_name,
        "modes": list(artifact["eval_modes"]),
        "format": artifact["format"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
