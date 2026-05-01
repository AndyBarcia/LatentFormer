# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import copy
import io
import itertools
import json
import logging
import os
import tempfile
from collections import OrderedDict
from typing import Optional

import numpy as np
from PIL import Image
from tabulate import tabulate

from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)


class COCOPanopticEvaluator(DatasetEvaluator):
    """
    Evaluate Panoptic Quality metrics using PanopticAPI.

    In addition to the standard class-aware COCO panoptic metrics, this evaluator
    reports class-agnostic PQ/SQ/RQ by remapping all predicted and ground-truth
    segments to a single shared category before running PanopticAPI.
    """

    _CLASS_AGNOSTIC_CATEGORY_ID = 1

    def __init__(self, dataset_name: str, output_dir: Optional[str] = None):
        self._metadata = MetadataCatalog.get(dataset_name)
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }

        self._output_dir = output_dir
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)

    def reset(self):
        self._predictions = []

    def _convert_category_id(self, segment_info):
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        return segment_info

    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb

        for input, output in zip(inputs, outputs):
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy()
            if segments_info is None:
                label_divisor = self._metadata.label_divisor
                segments_info = []
                for panoptic_label in np.unique(panoptic_img):
                    if panoptic_label == -1:
                        continue
                    pred_class = panoptic_label // label_divisor
                    isthing = (
                        pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    )
                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": int(pred_class),
                            "isthing": bool(isthing),
                        }
                    )
                panoptic_img += 1

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )

    @classmethod
    def _make_class_agnostic_json(cls, json_data):
        json_data = copy.deepcopy(json_data)
        for annotation in json_data["annotations"]:
            for segment_info in annotation["segments_info"]:
                segment_info["category_id"] = cls._CLASS_AGNOSTIC_CATEGORY_ID
        json_data["categories"] = [
            {
                "id": cls._CLASS_AGNOSTIC_CATEGORY_ID,
                "name": "object",
                "supercategory": "object",
                "isthing": 1,
            }
        ]
        return json_data

    @staticmethod
    def _pq_compute(gt_json, pred_json, gt_folder, pred_folder):
        from panopticapi.evaluation import pq_compute

        with contextlib.redirect_stdout(io.StringIO()):
            return pq_compute(
                gt_json,
                pred_json,
                gt_folder=gt_folder,
                pred_folder=pred_folder,
            )

    @staticmethod
    def _pq_average(pq_stat, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in categories.items():
            if isthing is not None and isthing != (label_info["isthing"] == 1):
                continue
            iou = pq_stat.pq_per_cat[label].iou
            tp = pq_stat.pq_per_cat[label].tp
            fp = pq_stat.pq_per_cat[label].fp
            fn = pq_stat.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {"pq": 0.0, "sq": 0.0, "rq": 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {"pq": pq_class, "sq": sq_class, "rq": rq_class}
            pq += pq_class
            sq += sq_class
            rq += rq_class
        if n == 0:
            return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "n": 0}, per_class_results
        return {"pq": pq / n, "sq": sq / n, "rq": rq / n, "n": n}, per_class_results

    @classmethod
    def _pq_compute_class_agnostic(cls, gt_json, pred_json, gt_folder, pred_folder):
        from panopticapi.evaluation import pq_compute_multi_core

        with open(gt_json, "r") as f:
            gt_json_data = json.load(f)
        with open(pred_json, "r") as f:
            pred_json_data = json.load(f)

        categories = {el["id"]: el for el in gt_json_data["categories"]}
        pred_annotations = {el["image_id"]: el for el in pred_json_data["annotations"]}
        matched_annotations_list = []
        for gt_ann in gt_json_data["annotations"]:
            image_id = gt_ann["image_id"]
            if image_id not in pred_annotations:
                raise Exception("no prediction for the image with id: {}".format(image_id))
            matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

        with contextlib.redirect_stdout(io.StringIO()):
            pq_stat = pq_compute_multi_core(
                matched_annotations_list,
                gt_folder,
                pred_folder,
                categories,
            )

        results = {}
        for name, isthing in [("All", None), ("Things", True), ("Stuff", False)]:
            results[name], per_class_results = cls._pq_average(
                pq_stat, categories, isthing=isthing
            )
            if name == "All":
                results["per_class"] = per_class_results
        return results

    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            predictions = []
            for p in self._predictions:
                p = p.copy()
                png_string = p.pop("png_string")
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(png_string)
                predictions.append(p)

            with open(gt_json, "r") as f:
                gt_json_data = json.load(f)

            pred_json_data = copy.deepcopy(gt_json_data)
            pred_json_data["annotations"] = predictions

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(pred_json_data))

            pq_res = self._pq_compute(
                gt_json,
                PathManager.get_local_path(predictions_json),
                gt_folder=gt_folder,
                pred_folder=pred_dir,
            )

            class_agnostic_gt_json = os.path.join(pred_dir, "gt_class_agnostic.json")
            class_agnostic_predictions_json = os.path.join(
                output_dir, "predictions_class_agnostic.json"
            )
            with open(class_agnostic_gt_json, "w") as f:
                f.write(json.dumps(self._make_class_agnostic_json(gt_json_data)))
            with PathManager.open(class_agnostic_predictions_json, "w") as f:
                f.write(json.dumps(self._make_class_agnostic_json(pred_json_data)))

            pq_res_class_agnostic = self._pq_compute_class_agnostic(
                class_agnostic_gt_json,
                PathManager.get_local_path(class_agnostic_predictions_json),
                gt_folder=gt_folder,
                pred_folder=pred_dir,
            )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]
        res["PQ_class_agnostic"] = 100 * pq_res_class_agnostic["All"]["pq"]
        res["SQ_class_agnostic"] = 100 * pq_res_class_agnostic["All"]["sq"]
        res["RQ_class_agnostic"] = 100 * pq_res_class_agnostic["All"]["rq"]

        results = OrderedDict({"panoptic_seg": res})
        _print_panoptic_results(pq_res)
        _print_panoptic_results(pq_res_class_agnostic, title="Class-Agnostic Panoptic Evaluation Results")

        return results


def _print_panoptic_results(pq_res, title="Panoptic Evaluation Results"):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info(title + ":\n" + table)
