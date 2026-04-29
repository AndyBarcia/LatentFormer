# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
    module=r"detectron2\.engine\.train_loop",
)

import copy
import itertools
import json
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.build import trivial_batch_collator
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler
from detectron2.data import build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.evaluation.evaluator import DatasetEvaluator, inference_on_dataset
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    LatentFormerSignatureEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_latentformer_config,
    add_maskformer2_config,
)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _pareto_frontier_indices(precision, recall):
    order = torch.argsort(recall, descending=True)
    best_precision = precision.new_tensor(-1.0)
    pareto = []
    for idx in order.tolist():
        if precision[idx] > best_precision + 1e-7:
            pareto.append(idx)
            best_precision = precision[idx]
    if not pareto:
        return torch.empty(0, dtype=torch.long, device=precision.device)
    pareto = torch.as_tensor(pareto, dtype=torch.long, device=precision.device)
    return pareto[torch.argsort(recall[pareto])]


def plot_latentformer_seed_cluster_pr_predictions(cfg, model, iteration=None):
    if cfg.MODEL.META_ARCHITECTURE != "LatentFormer":
        return None
    if not cfg.MODEL.LATENT_FORMER.SEED_CLUSTER_PR.PLOT_ON_EVAL:
        return None
    if not comm.is_main_process():
        return None

    model = _unwrap_model(model)
    seed_selection_modules = getattr(model, "seed_selection_modules", {})
    clustering_seed_selection = (
        seed_selection_modules["ClusteringSeedSelection"]
        if "ClusteringSeedSelection" in seed_selection_modules
        else None
    )
    predictor = getattr(clustering_seed_selection, "threshold_pr_mlp", None)
    if predictor is None:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pr_cfg = cfg.MODEL.LATENT_FORMER.SEED_CLUSTER_PR
    num_points = int(pr_cfg.PLOT_NUM_POINTS)
    seed_min, seed_max = pr_cfg.SEED_THRESHOLD_RANGE
    duplicate_min, duplicate_max = pr_cfg.DUPLICATE_THRESHOLD_RANGE

    device = next(predictor.parameters()).device
    seed_thresholds = torch.linspace(float(seed_min), float(seed_max), num_points, device=device)
    duplicate_thresholds = torch.linspace(
        float(duplicate_min),
        float(duplicate_max),
        num_points,
        device=device,
    )
    was_training = predictor.training
    predictor.eval()
    with torch.no_grad():
        predictions = predictor(seed_thresholds, duplicate_thresholds).detach().cpu()
    if was_training:
        predictor.train()

    precision = predictions[..., 0].reshape(-1)
    recall = predictions[..., 1].reshape(-1)
    seed_grid, _ = torch.meshgrid(
        seed_thresholds.detach().cpu(),
        duplicate_thresholds.detach().cpu(),
        indexing="ij",
    )
    flat_seed = seed_grid.reshape(-1)
    pareto = _pareto_frontier_indices(precision, recall)

    plot_dir = os.path.join(cfg.OUTPUT_DIR, "seed_cluster_pr_plots")
    os.makedirs(plot_dir, exist_ok=True)
    if iteration is None:
        name = "threshold_pr_mlp_eval"
    else:
        name = f"threshold_pr_mlp_iter_{int(iteration):07d}"
    output_path = os.path.join(plot_dir, f"{name}.jpg")
    latest_path = os.path.join(plot_dir, "threshold_pr_mlp_latest.jpg")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    scatter = ax.scatter(
        recall.numpy(),
        precision.numpy(),
        c=flat_seed.numpy(),
        s=4,
        alpha=0.22,
        cmap="viridis",
        linewidths=0,
    )
    if pareto.numel() > 0:
        ax.plot(
            recall[pareto].numpy(),
            precision[pareto].numpy(),
            color="#d62728",
            linewidth=2.2,
            label=f"Pareto frontier ({pareto.numel()} pts)",
        )
        ax.scatter(
            recall[pareto].numpy(),
            precision[pareto].numpy(),
            color="#d62728",
            s=10,
            zorder=3,
        )
    ax.set_xlabel("Predicted recall")
    ax.set_ylabel("Predicted precision")
    ax.set_title("Threshold MLP precision-recall predictions")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Seed threshold")
    if pareto.numel() > 0:
        ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, format="jpg", pil_kwargs={"quality": 85, "optimize": True})
    fig.savefig(latest_path, format="jpg", pil_kwargs={"quality": 85, "optimize": True})
    plt.close(fig)
    logging.getLogger(__name__).info("Wrote LatentFormer PR threshold plot to %s", output_path)
    return output_path


class LatentFormerMultiModeEvaluator(DatasetEvaluator):
    def __init__(self, evaluators_by_mode):
        self.evaluators_by_mode = evaluators_by_mode

    def reset(self):
        for evaluator in self.evaluators_by_mode.values():
            evaluator.reset()

    def process(self, inputs, outputs):
        for mode, evaluator in self.evaluators_by_mode.items():
            mode_outputs = outputs[mode] if isinstance(outputs, dict) else outputs
            evaluator.process(inputs, mode_outputs)

    def evaluate(self):
        results = OrderedDict()
        for mode, evaluator in self.evaluators_by_mode.items():
            mode_results = evaluator.evaluate()
            if mode_results is None:
                continue
            for key, value in mode_results.items():
                results[f"{mode}/{key}"] = value
        return results


def _limited_eval_image_ids(dataset_dicts):
    image_ids = {record["image_id"] for record in dataset_dicts if "image_id" in record}
    if not image_ids:
        return None
    return image_ids


def _write_limited_coco_json(json_file, limited_name, image_ids, output_dir, suffix):
    with open(json_file) as handle:
        annotations = json.load(handle)

    annotations["images"] = [
        image for image in annotations.get("images", []) if image.get("id") in image_ids
    ]
    annotations["annotations"] = [
        annotation
        for annotation in annotations.get("annotations", [])
        if annotation.get("image_id") in image_ids
    ]

    annotation_dir = os.path.join(output_dir, "dev_eval_annotations")
    os.makedirs(annotation_dir, exist_ok=True)
    limited_json = os.path.join(annotation_dir, f"{limited_name}_{suffix}.json")
    with open(limited_json, "w") as handle:
        json.dump(annotations, handle)
    return limited_json


def _update_limited_eval_metadata(metadata, limited_name, dataset_dicts, output_dir):
    image_ids = _limited_eval_image_ids(dataset_dicts)
    if image_ids is None:
        return

    if metadata.get("panoptic_json"):
        metadata["panoptic_json"] = _write_limited_coco_json(
            metadata["panoptic_json"], limited_name, image_ids, output_dir, "panoptic"
        )
    if metadata.get("json_file"):
        metadata["json_file"] = _write_limited_coco_json(
            metadata["json_file"], limited_name, image_ids, output_dir, "instances"
        )


def _register_limited_test_datasets(cfg, limit):
    if limit <= 0 or len(cfg.DATASETS.TEST) == 0:
        return

    logger = logging.getLogger(__name__)
    limited_names = []
    for dataset_name in cfg.DATASETS.TEST:
        limited_name = f"{dataset_name}_dev_subset_{limit}"
        if limited_name not in DatasetCatalog.list():
            dataset_dicts = DatasetCatalog.get(dataset_name)[:limit]
            DatasetCatalog.register(
                limited_name,
                lambda dataset_dicts=dataset_dicts: dataset_dicts,
            )
            metadata = MetadataCatalog.get(dataset_name).as_dict()
            metadata.pop("name", None)
            _update_limited_eval_metadata(metadata, limited_name, dataset_dicts, cfg.OUTPUT_DIR)
            MetadataCatalog.get(limited_name).set(**metadata)
        limited_names.append(limited_name)

    cfg.DATASETS.TEST = tuple(limited_names)
    logger.info("Using limited evaluation datasets: %s", ", ".join(limited_names))


def _build_detection_test_loader_with_batch_size(cfg, dataset_name, mapper):
    dataset = DatasetFromList(DatasetCatalog.get(dataset_name), copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler,
        cfg.TEST.IMS_PER_BATCH,
        drop_last=False,
    )
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    def test_and_save_results(self):
        results = super().test_and_save_results()
        plot_latentformer_seed_cluster_pr_predictions(self.cfg, self.model, iteration=self.iter)
        return results

    @staticmethod
    def _model_test_cfg(cfg):
        if cfg.MODEL.META_ARCHITECTURE == "LatentFormer":
            return cfg.MODEL.LATENT_FORMER.TEST
        return cfg.MODEL.MASK_FORMER.TEST

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        model_test_cfg = cls._model_test_cfg(cfg)
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if (
            cfg.MODEL.META_ARCHITECTURE == "LatentFormer"
            and cfg.MODEL.LATENT_FORMER.TEST.SIGNATURE_ON
        ):
            evaluator_list.append(
                LatentFormerSignatureEvaluator(
                    cfg.MODEL.LATENT_FORMER.MATCHING_SIMILARITY_METRIC,
                    output_dir=output_folder,
                )
            )
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if model_test_cfg.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and model_test_cfg.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and model_test_cfg.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and model_test_cfg.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and model_test_cfg.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if model_test_cfg.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if model_test_cfg.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and model_test_cfg.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, False)
            if cfg.TEST.IMS_PER_BATCH > 1:
                return _build_detection_test_loader_with_batch_size(cfg, dataset_name, mapper)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        return super().build_test_loader(cfg, dataset_name)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        eval_modes = ()
        if cfg.MODEL.META_ARCHITECTURE == "LatentFormer":
            eval_modes = tuple(cfg.MODEL.LATENT_FORMER.TEST.EVAL_MODES)
        if len(eval_modes) <= 1:
            return super().test(cfg, model, evaluators)

        logger = logging.getLogger(__name__)
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                evaluators_by_mode = OrderedDict()
                for mode in eval_modes:
                    evaluators_by_mode[mode] = cls.build_evaluator(
                        cfg,
                        dataset_name,
                        output_folder=os.path.join(cfg.OUTPUT_DIR, "inference", mode),
                    )
                evaluator = LatentFormerMultiModeEvaluator(evaluators_by_mode)
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                logger.info("Evaluation results for %s in csv format:", dataset_name)
                from detectron2.evaluation.testing import print_csv_format

                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_latentformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if (
        cfg.MODEL.META_ARCHITECTURE == "LatentFormer"
        and (
            cfg.MODEL.LATENT_FORMER.TEST.SIGNATURE_ON
            or any(
                mode in {"GoldenSeedSelection", "GTOracleSeedSelection"}
                for mode in cfg.MODEL.LATENT_FORMER.TEST.EVAL_MODES
            )
        )
    ):
        cfg.MODEL.LATENT_FORMER.TEST.LOAD_GT_FOR_EVAL = True
    _register_limited_test_datasets(cfg, cfg.TEST.EVAL_MAX_IMAGES)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        plot_latentformer_seed_cluster_pr_predictions(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
