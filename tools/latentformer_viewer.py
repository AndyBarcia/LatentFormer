#!/usr/bin/env python3
"""Small interactive web viewer for LatentFormer checkpoints."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LatentFormer Viewer</title>
  <style>
    :root {
      --bg: #f3efe7;
      --panel: #fffaf2;
      --panel-2: #f7f1e7;
      --ink: #1d2433;
      --muted: #645d54;
      --accent: #9f3d2f;
      --accent-2: #1d6b6b;
      --line: #dbcdbb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(159,61,47,0.13), transparent 28%),
        radial-gradient(circle at top right, rgba(29,107,107,0.12), transparent 24%),
        linear-gradient(180deg, #f7f2e9 0%, var(--bg) 100%);
    }
    .shell {
      max-width: 1500px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero, .panel {
      background: rgba(255, 250, 242, 0.92);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 18px 40px rgba(39, 34, 24, 0.08);
      backdrop-filter: blur(8px);
    }
    .hero {
      padding: 24px 28px;
      margin-bottom: 18px;
    }
    .hero h1 {
      margin: 0 0 6px;
      font-size: clamp(28px, 4vw, 46px);
      line-height: 1;
      letter-spacing: -0.03em;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      font-size: 17px;
    }
    .panel {
      padding: 18px;
      margin-bottom: 18px;
    }
    .controls {
      display: grid;
      grid-template-columns: 2.5fr 1fr 1fr auto;
      gap: 14px;
      align-items: end;
    }
    .row2 {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr auto;
      gap: 14px;
      margin-top: 14px;
      align-items: end;
    }
    label {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    input, select, button {
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      padding: 12px 14px;
      font: inherit;
    }
    input, select {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    button {
      cursor: pointer;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent), #bf5a2d);
      color: white;
      border: none;
      box-shadow: 0 10px 24px rgba(159, 61, 47, 0.22);
    }
    button.secondary {
      background: linear-gradient(135deg, var(--accent-2), #2c8d8d);
      box-shadow: 0 10px 24px rgba(29, 107, 107, 0.2);
    }
    button:disabled {
      opacity: 0.6;
      cursor: wait;
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }
    .chip {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
    }
    .chip .k {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }
    .chip .v {
      font-size: 14px;
      word-break: break-word;
    }
    #status {
      margin-top: 14px;
      padding: 12px 14px;
      border-radius: 14px;
      background: var(--panel-2);
      color: var(--muted);
      min-height: 46px;
      white-space: pre-wrap;
    }
    #errorBox {
      margin-top: 12px;
      padding: 14px;
      border-radius: 14px;
      border: 1px solid rgba(159, 61, 47, 0.28);
      background: rgba(159, 61, 47, 0.08);
      display: none;
    }
    #errorBox strong {
      display: block;
      margin-bottom: 8px;
      color: #7d2118;
    }
    #errorTrace {
      margin: 0;
      overflow-x: auto;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 12px;
      line-height: 1.45;
      color: #4f1b15;
      white-space: pre-wrap;
    }
    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 16px;
    }
    .wide-card {
      grid-column: 1 / -1;
    }
    .card {
      background: rgba(255,255,255,0.94);
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
    }
    .card .title {
      padding: 12px 14px;
      font-weight: 700;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(90deg, rgba(159,61,47,0.08), rgba(29,107,107,0.06));
    }
    .card img {
      display: block;
      width: 100%;
      height: auto;
      background: #ebe3d5;
    }
    .mode-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
    }
    .mode-panel {
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      background: rgba(247, 241, 231, 0.62);
    }
    .mode-panel h3 {
      margin: 0 0 10px;
      font-size: 20px;
    }
    .pred-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    .small {
      color: var(--muted);
      font-size: 14px;
    }
    @media (max-width: 980px) {
      .controls, .row2, .meta {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>LatentFormer Viewer</h1>
      <p>Load a model folder, pick a validation image, and compare every LatentFormer eval mode on the same sample.</p>
    </section>

    <section class="panel">
      <div class="controls">
        <div>
          <label for="modelDir">Model Folder</label>
          <input id="modelDir" type="text" />
        </div>
        <div>
          <label for="configFile">Config File</label>
          <select id="configFile"></select>
        </div>
        <div>
          <label for="checkpoint">Checkpoint</label>
          <select id="checkpoint"></select>
        </div>
        <div>
          <label>&nbsp;</label>
          <button id="loadBtn" class="secondary">Load Folder</button>
        </div>
      </div>

      <div class="row2">
        <div>
          <label for="datasetName">Validation Dataset</label>
          <select id="datasetName"></select>
        </div>
        <div>
          <label for="imageIndex">Image</label>
          <select id="imageIndex"></select>
        </div>
        <div>
          <label for="imageSearch">Jump by Name</label>
          <input id="imageSearch" type="text" placeholder="type part of file name" />
        </div>
        <div>
          <label>&nbsp;</label>
          <button id="runBtn">Run Inference</button>
        </div>
      </div>

      <div class="meta" id="meta"></div>
      <div id="status">Waiting for a model folder.</div>
      <div id="errorBox">
        <strong>Backend Trace</strong>
        <pre id="errorTrace"></pre>
      </div>
    </section>

    <section class="panel">
      <div class="gallery" id="staticGallery"></div>
    </section>

    <section class="panel">
      <div class="mode-grid" id="modeGrid"></div>
    </section>

    <section class="panel">
      <div class="gallery" id="signatureGallery"></div>
    </section>
  </div>

  <script>
    const state = { info: null, images: [] };
    const modelDir = document.getElementById("modelDir");
    const configFile = document.getElementById("configFile");
    const checkpoint = document.getElementById("checkpoint");
    const datasetName = document.getElementById("datasetName");
    const imageIndex = document.getElementById("imageIndex");
    const imageSearch = document.getElementById("imageSearch");
    const loadBtn = document.getElementById("loadBtn");
    const runBtn = document.getElementById("runBtn");
    const statusBox = document.getElementById("status");
    const errorBox = document.getElementById("errorBox");
    const errorTrace = document.getElementById("errorTrace");
    const meta = document.getElementById("meta");
    const staticGallery = document.getElementById("staticGallery");
    const modeGrid = document.getElementById("modeGrid");
    const signatureGallery = document.getElementById("signatureGallery");

    function ellipsizeMiddle(text, maxLen = 48) {
      const value = String(text ?? "");
      if (value.length <= maxLen) return value;
      const keep = Math.max(8, Math.floor((maxLen - 1) / 2));
      return `${value.slice(0, keep)}…${value.slice(-keep)}`;
    }

    function basename(path) {
      const value = String(path ?? "");
      const parts = value.split(/[\\\\/]/);
      return parts[parts.length - 1] || value;
    }

    function setStatus(text) {
      statusBox.textContent = text;
    }

    function clearError() {
      errorBox.style.display = "none";
      errorTrace.textContent = "";
    }

    function showError(message, trace) {
      setStatus(message);
      if (trace) {
        errorTrace.textContent = trace;
        errorBox.style.display = "block";
      } else {
        clearError();
      }
    }

    function setBusy(flag) {
      loadBtn.disabled = flag;
      runBtn.disabled = flag;
    }

    function fillSelect(node, values, labelFn) {
      node.innerHTML = "";
      values.forEach((value, idx) => {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = labelFn ? labelFn(value, idx) : value;
        opt.title = String(value);
        node.appendChild(opt);
      });
    }

    function renderMeta(info) {
      const items = [
        ["Dataset", info.default_dataset || "-"],
        ["Eval Modes", (info.eval_modes || []).join(", ") || "-"],
        ["Images", String(info.image_count ?? "-")],
        ["Device", info.device || "-"],
      ];
      meta.innerHTML = items.map(([k, v]) => `
        <div class="chip">
          <div class="k">${k}</div>
          <div class="v">${v}</div>
        </div>
      `).join("");
    }

    function imageOptionLabel(item, idx) {
      return ellipsizeMiddle(`${idx}: ${item.label}`, 56);
    }

    function refreshImages() {
      const current = state.info.datasets[datasetName.value] || [];
      state.images = current;
      fillSelect(imageIndex, current.map((_, idx) => String(idx)), (_, idx) => imageOptionLabel(current[idx], idx));
    }

    async function loadModel() {
      setBusy(true);
      setStatus("Scanning model folder and dataset registrations...");
      clearError();
      staticGallery.innerHTML = "";
      modeGrid.innerHTML = "";
      signatureGallery.innerHTML = "";
      try {
        const params = new URLSearchParams({ model_dir: modelDir.value.trim() });
        const res = await fetch(`/api/model-info?${params.toString()}`);
        const data = await res.json();
        if (!res.ok) {
          showError(data.error || "Failed to load model folder.", data.traceback || "");
          return;
        }

        state.info = data;
        fillSelect(configFile, data.config_files, (value, idx) => data.display_config_files?.[idx] || basename(value));
        configFile.value = data.selected_config;
        configFile.title = data.selected_config;
        fillSelect(checkpoint, data.checkpoints, (value, idx) => data.display_checkpoints?.[idx] || basename(value));
        checkpoint.value = data.selected_checkpoint;
        checkpoint.title = data.selected_checkpoint;
        fillSelect(
          datasetName,
          Object.keys(data.datasets),
          (value) => data.display_datasets?.[value] || ellipsizeMiddle(value, 44),
        );
        datasetName.value = data.default_dataset;
        datasetName.title = data.default_dataset;
        refreshImages();
        renderMeta(data);
        setStatus(`Loaded ${data.model_dir}\n${data.image_count} images in ${data.default_dataset}`);
        clearError();
      } catch (err) {
        showError(String(err), "");
      } finally {
        setBusy(false);
      }
    }

    async function runInference() {
      setBusy(true);
      setStatus("Running model on the selected image...");
      clearError();
      staticGallery.innerHTML = "";
      modeGrid.innerHTML = "";
      signatureGallery.innerHTML = "";
      try {
        const body = {
          model_dir: modelDir.value.trim(),
          config_file: configFile.value,
          checkpoint: checkpoint.value,
          dataset_name: datasetName.value,
          image_index: Number(imageIndex.value),
        };
        const res = await fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (!res.ok) {
          showError(data.error || "Inference failed.", data.traceback || "");
          return;
        }

        staticGallery.innerHTML = data.static_panels.map(panel => `
          <div class="card ${panel.wide ? "wide-card" : ""}">
            <div class="title">${panel.title}</div>
            <img src="data:image/png;base64,${panel.image_base64}" alt="${panel.title}" />
          </div>
        `).join("");

        modeGrid.innerHTML = data.mode_panels.map(mode => `
          <div class="mode-panel">
            <h3>${mode.mode}</h3>
            <div class="small">${mode.summary}</div>
            <div class="pred-grid">
              ${mode.images.map(panel => `
                <div class="card">
                  <div class="title">${panel.title}</div>
                  <img src="data:image/png;base64,${panel.image_base64}" alt="${panel.title}" />
                </div>
              `).join("")}
            </div>
          </div>
        `).join("");

        signatureGallery.innerHTML = data.signature_panels.map(panel => `
          <div class="card ${panel.wide ? "wide-card" : ""}">
            <div class="title">${panel.title}</div>
            <img src="data:image/png;base64,${panel.image_base64}" alt="${panel.title}" />
          </div>
        `).join("");

        setStatus(`Rendered ${data.record_label}\nCheckpoint: ${data.checkpoint}`);
        clearError();
      } catch (err) {
        showError(String(err), "");
      } finally {
        setBusy(false);
      }
    }

    datasetName.addEventListener("change", refreshImages);
    datasetName.addEventListener("change", () => {
      datasetName.title = datasetName.value;
    });
    configFile.addEventListener("change", () => {
      configFile.title = configFile.value;
    });
    checkpoint.addEventListener("change", () => {
      checkpoint.title = checkpoint.value;
    });
    imageSearch.addEventListener("input", () => {
      const query = imageSearch.value.trim().toLowerCase();
      if (!query) return;
      const idx = state.images.findIndex(item => item.label.toLowerCase().includes(query));
      if (idx >= 0) imageIndex.value = String(idx);
    });
    loadBtn.addEventListener("click", loadModel);
    runBtn.addEventListener("click", runInference);

    fetch("/api/defaults").then(r => r.json()).then(data => {
      modelDir.value = data.default_model_dir;
      loadModel();
    }).catch(err => setStatus(String(err)));
  </script>
</body>
</html>
"""


def rgb_to_id(color):
    import numpy as np

    color = np.asarray(color, dtype=np.int64)
    if color.ndim == 3:
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return color[0] + 256 * color[1] + 256 * 256 * color[2]


def image_to_base64(image):
    from PIL import Image

    if image.dtype != "uint8":
        image = image.clip(0, 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class AppConfig:
    default_model_dir: str
    host: str
    port: int


class ViewerState:
    def __init__(self):
        self._lock = threading.Lock()
        self._cache_key = None
        self._bundle = None

    def get_or_load_bundle(self, model_dir: str, config_file: str, checkpoint: str):
        key = (os.path.abspath(model_dir), os.path.abspath(config_file), os.path.abspath(checkpoint))
        with self._lock:
            if self._cache_key == key and self._bundle is not None:
                return self._bundle
            bundle = self._load_bundle(model_dir, config_file, checkpoint)
            self._cache_key = key
            self._bundle = bundle
            return bundle

    def _load_bundle(self, model_dir: str, config_file: str, checkpoint: str):
        imports = _lazy_imports()
        cfg = load_cfg(config_file, model_dir, checkpoint, imports)
        ensure_test_datasets_registered(cfg)

        model = imports["Trainer"].build_model(cfg)
        model.eval()
        imports["DetectionCheckpointer"](model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)

        dataset_records = {
            name: imports["DatasetCatalog"].get(name)
            for name in cfg.DATASETS.TEST
        }
        mappers = {name: build_inference_mapper(cfg, imports) for name in cfg.DATASETS.TEST}
        return {
            "cfg": cfg,
            "model": model,
            "imports": imports,
            "dataset_records": dataset_records,
            "mappers": mappers,
        }


def _lazy_imports():
    import numpy as np
    import torch
    from PIL import Image

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data import detection_utils as utils
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.utils.visualizer import ColorMode, Visualizer

    from mask2former import (
        COCOPanopticNewBaselineDatasetMapper,
        add_latentformer_config,
        add_maskformer2_config,
    )
    from train_net import Trainer
    from detectron2.config import get_cfg

    return {
        "np": np,
        "torch": torch,
        "Image": Image,
        "DetectionCheckpointer": DetectionCheckpointer,
        "DatasetCatalog": DatasetCatalog,
        "MetadataCatalog": MetadataCatalog,
        "utils": utils,
        "add_deeplab_config": add_deeplab_config,
        "Visualizer": Visualizer,
        "ColorMode": ColorMode,
        "COCOPanopticNewBaselineDatasetMapper": COCOPanopticNewBaselineDatasetMapper,
        "Trainer": Trainer,
        "get_cfg": get_cfg,
        "add_maskformer2_config": add_maskformer2_config,
        "add_latentformer_config": add_latentformer_config,
    }


def find_model_assets(model_dir: str):
    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {root}")

    config_files = sorted(str(p) for p in root.glob("*.yaml"))
    if not config_files:
        config_files = sorted(str(p) for p in root.glob("*.yml"))
    checkpoints = sorted((str(p) for p in root.glob("model_*.pth")), reverse=True)
    final_checkpoint = root / "model_final.pth"
    if final_checkpoint.is_file():
        checkpoints.insert(0, str(final_checkpoint))

    last_checkpoint_path = root / "last_checkpoint"
    selected_checkpoint = checkpoints[0] if checkpoints else ""
    if last_checkpoint_path.is_file():
        last_name = last_checkpoint_path.read_text().strip()
        candidate = root / last_name
        if candidate.is_file():
            selected_checkpoint = str(candidate)
            if str(candidate) not in checkpoints:
                checkpoints.insert(0, str(candidate))

    if not config_files:
        raise FileNotFoundError(f"No config YAML file found in {root}")
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {root}")

    return {
        "model_dir": str(root),
        "config_files": config_files,
        "selected_config": config_files[0],
        "checkpoints": checkpoints,
        "selected_checkpoint": selected_checkpoint,
    }


def load_cfg(config_file: str, model_dir: str, checkpoint: str, imports):
    cfg = imports["get_cfg"]()
    imports["add_deeplab_config"](cfg)
    imports["add_maskformer2_config"](cfg)
    imports["add_latentformer_config"](cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = checkpoint
    cfg.OUTPUT_DIR = model_dir
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
    cfg.freeze()
    return cfg


def ensure_test_datasets_registered(cfg):
    imports = _lazy_imports()
    dataset_catalog = imports["DatasetCatalog"]
    metadata_catalog = imports["MetadataCatalog"]

    missing = [name for name in cfg.DATASETS.TEST if name not in dataset_catalog.list()]
    for name in missing:
        marker = "_dev_subset_"
        if marker not in name:
            raise KeyError(f"Dataset '{name}' is not registered.")
        base_name, limit_str = name.rsplit(marker, 1)
        if base_name not in dataset_catalog.list():
            raise KeyError(
                f"Dataset '{name}' is not registered and base dataset '{base_name}' was not found."
            )
        limit = int(limit_str)
        dataset_dicts = dataset_catalog.get(base_name)[:limit]
        dataset_catalog.register(name, lambda dataset_dicts=dataset_dicts: dataset_dicts)
        metadata = metadata_catalog.get(base_name).as_dict()
        metadata.pop("name", None)
        metadata_catalog.get(name).set(**metadata)


def build_inference_mapper(cfg, imports):
    if cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
        return imports["COCOPanopticNewBaselineDatasetMapper"](cfg, False)
    return None


def summarize_record(record: dict, idx: int) -> dict:
    label = Path(record.get("file_name", f"image_{idx}")).name
    if "image_id" in record:
        label = f"{label} [id={record['image_id']}]"
    return {
        "label": label,
        "file_name": record.get("file_name", ""),
        "image_id": record.get("image_id"),
    }


def prepare_single_input(record: dict, mapper):
    import copy

    if mapper is None:
        raise NotImplementedError(
            "Only INPUT.DATASET_MAPPER_NAME=coco_panoptic_lsj is supported in this viewer for now."
        )
    return mapper(copy.deepcopy(record))


def draw_input_image(record: dict):
    from PIL import Image
    import numpy as np

    image = np.asarray(Image.open(record["file_name"]).convert("RGB"))
    return image


def draw_gt_overlay(record: dict, metadata, imports):
    from PIL import Image

    image = draw_input_image(record)
    visualizer = imports["Visualizer"](image, metadata=metadata, instance_mode=imports["ColorMode"].IMAGE)

    if "pan_seg_file_name" in record and record.get("segments_info"):
        pan_seg = rgb_to_id(imports["np"].asarray(Image.open(record["pan_seg_file_name"]).convert("RGB")))
        pan_seg = imports["torch"].as_tensor(pan_seg.astype("int64"))
        return visualizer.draw_panoptic_seg(pan_seg, record["segments_info"]).get_image()
    if record.get("annotations"):
        return visualizer.draw_dataset_dict(record).get_image()
    return image


def draw_prediction_panels(record: dict, outputs, metadata, imports):
    panels = []
    image = draw_input_image(record)
    visualizer_kwargs = {"metadata": metadata, "instance_mode": imports["ColorMode"].IMAGE}

    if "panoptic_seg" in outputs:
        panoptic_seg, segments_info = outputs["panoptic_seg"]
        image_vis = imports["Visualizer"](image, **visualizer_kwargs)
        panels.append(
            {
                "title": "Panoptic Prediction",
                "image_base64": image_to_base64(
                    image_vis.draw_panoptic_seg(
                        panoptic_seg.to("cpu"),
                        segments_info,
                    ).get_image()
                ),
            }
        )
    if "instances" in outputs:
        image_vis = imports["Visualizer"](image, **visualizer_kwargs)
        panels.append(
            {
                "title": "Instance Prediction",
                "image_base64": image_to_base64(
                    image_vis.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
                ),
            }
        )
    if "sem_seg" in outputs:
        image_vis = imports["Visualizer"](image, **visualizer_kwargs)
        panels.append(
            {
                "title": "Semantic Prediction",
                "image_base64": image_to_base64(
                    image_vis.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
                ),
            }
        )
    return panels


def draw_signature_projection_panel(processed: dict, imports):
    diagnostics = processed.get("latentformer_signature_eval")
    if diagnostics is None:
        return None

    det = diagnostics.get("det_signatures")
    gt = diagnostics.get("gt_signatures")
    seeds = diagnostics.get("selected_seed_signatures")
    seed_scores = diagnostics.get("selected_seed_scores")
    if det is None or gt is None or seeds is None:
        return None

    det_np = det.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    seeds_np = seeds.detach().cpu().numpy()
    if det_np.size == 0 and gt_np.size == 0 and seeds_np.size == 0:
        return None

    figure = render_signature_projection_figure(
        det_signatures=det_np,
        gt_signatures=gt_np,
        selected_seed_signatures=seeds_np,
        selected_seed_scores=None if seed_scores is None else seed_scores.detach().cpu().numpy(),
    )
    if figure is None:
        return None

    image, method_name = figure
    return {
        "title": f"Signature Projection ({method_name})",
        "image_base64": image_to_base64(image),
    }


def draw_combined_signature_projection(outputs_by_mode: dict):
    golden = outputs_by_mode.get("GoldenSeedSelection")
    clustering = outputs_by_mode.get("ClusteringSeedSelection")
    if not golden or not clustering:
        return None

    golden_diag = golden[0].get("latentformer_signature_eval")
    clustering_diag = clustering[0].get("latentformer_signature_eval")
    if golden_diag is None or clustering_diag is None:
        return None

    gt = golden_diag.get("gt_signatures")
    queries = golden_diag.get("det_signatures")
    golden_seeds = golden_diag.get("selected_seed_signatures")
    clustering_seeds = clustering_diag.get("selected_seed_signatures")
    golden_scores = golden_diag.get("selected_seed_scores")
    clustering_scores = clustering_diag.get("selected_seed_scores")
    if gt is None or queries is None or golden_seeds is None or clustering_seeds is None:
        return None

    figure = render_combined_signature_projection_figure(
        gt_signatures=gt.detach().cpu().numpy(),
        query_signatures=queries.detach().cpu().numpy(),
        golden_seed_signatures=golden_seeds.detach().cpu().numpy(),
        clustering_seed_signatures=clustering_seeds.detach().cpu().numpy(),
        golden_seed_scores=None if golden_scores is None else golden_scores.detach().cpu().numpy(),
        clustering_seed_scores=None if clustering_scores is None else clustering_scores.detach().cpu().numpy(),
    )
    if figure is None:
        return None

    image, method_name = figure
    return {
        "title": f"Combined Signature Projection ({method_name})",
        "image_base64": image_to_base64(image),
    }


def render_signature_projection_figure(
    *,
    det_signatures,
    gt_signatures,
    selected_seed_signatures,
    selected_seed_scores,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    points = []
    if len(det_signatures):
        points.append(det_signatures)
    if len(gt_signatures):
        points.append(gt_signatures)
    if len(selected_seed_signatures):
        points.append(selected_seed_signatures)

    if not points:
        return None

    all_points = np.concatenate(points, axis=0)
    coords, method_name = project_signatures_2d(all_points)

    det_n = len(det_signatures)
    gt_n = len(gt_signatures)
    seed_n = len(selected_seed_signatures)
    det_xy = coords[:det_n]
    gt_xy = coords[det_n:det_n + gt_n]
    seed_xy = coords[det_n + gt_n:det_n + gt_n + seed_n]

    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=160)
    ax.set_facecolor("#fffaf2")
    fig.patch.set_facecolor("#fffaf2")

    if det_n:
        ax.scatter(
            det_xy[:, 0], det_xy[:, 1],
            s=22, c="#5b718b", alpha=0.45, label=f"Queries ({det_n})",
            edgecolors="none",
        )
    if gt_n:
        ax.scatter(
            gt_xy[:, 0], gt_xy[:, 1],
            s=90, c="#bb4f3b", alpha=0.95, marker="X", label=f"GT ({gt_n})",
            edgecolors="white", linewidths=0.6,
        )
    if seed_n:
        if selected_seed_scores is not None and len(selected_seed_scores) == seed_n:
            ax.scatter(
                seed_xy[:, 0], seed_xy[:, 1],
                s=110, c=selected_seed_scores, cmap="viridis", alpha=0.95,
                marker="o", label=f"Selected Seeds ({seed_n})",
                edgecolors="black", linewidths=0.5,
            )
            for idx, (x, y) in enumerate(seed_xy):
                ax.text(x, y, str(idx), fontsize=7, color="#14212b", ha="center", va="center")
            cbar = fig.colorbar(ax.collections[-1], ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Seed score", rotation=270, labelpad=12)
        else:
            ax.scatter(
                seed_xy[:, 0], seed_xy[:, 1],
                s=110, c="#1d8a89", alpha=0.95, marker="o",
                label=f"Selected Seeds ({seed_n})",
                edgecolors="black", linewidths=0.5,
            )

    ax.set_title("Query/GT Signature Layout", fontsize=12)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(True, color="#dbcdbb", alpha=0.45, linewidth=0.7)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return np.asarray(Image.open(buf).convert("RGB")), method_name


def project_signatures_2d(points):
    import numpy as np
    from sklearn.decomposition import PCA

    if points.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32), "Degenerate"

    try:
        import umap

        n_neighbors = max(2, min(15, points.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.15,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(points).astype(np.float32), "UMAP"
    except Exception:
        n_components = 2 if points.shape[0] >= 2 and points.shape[1] >= 2 else 1
        projected = PCA(n_components=n_components).fit_transform(points).astype(np.float32)
        if projected.shape[1] == 1:
            projected = np.concatenate([projected, np.zeros((projected.shape[0], 1), dtype=np.float32)], axis=1)
        return projected, "PCA"


def render_combined_signature_projection_figure(
    *,
    gt_signatures,
    query_signatures,
    golden_seed_signatures,
    clustering_seed_signatures,
    golden_seed_scores,
    clustering_seed_scores,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = []
    if len(query_signatures):
        groups.append(query_signatures)
    if len(gt_signatures):
        groups.append(gt_signatures)
    if len(golden_seed_signatures):
        groups.append(golden_seed_signatures)
    if len(clustering_seed_signatures):
        groups.append(clustering_seed_signatures)
    if not groups:
        return None

    all_points = np.concatenate(groups, axis=0)
    coords, method_name = project_signatures_2d(all_points)

    query_n = len(query_signatures)
    gt_n = len(gt_signatures)
    golden_n = len(golden_seed_signatures)
    clustering_n = len(clustering_seed_signatures)

    query_xy = coords[:query_n]
    gt_xy = coords[query_n:query_n + gt_n]
    golden_xy = coords[query_n + gt_n:query_n + gt_n + golden_n]
    clustering_xy = coords[
        query_n + gt_n + golden_n:query_n + gt_n + golden_n + clustering_n
    ]

    fig, ax = plt.subplots(figsize=(9.0, 6.2), dpi=170)
    ax.set_facecolor("#fffaf2")
    fig.patch.set_facecolor("#fffaf2")

    if query_n:
        ax.scatter(
            query_xy[:, 0], query_xy[:, 1],
            s=26, c="#62758a", alpha=0.30, marker=".",
            label=f"All queries ({query_n})",
            edgecolors="none", zorder=1,
        )

    if gt_n:
        ax.scatter(
            gt_xy[:, 0], gt_xy[:, 1],
            s=150, c="#bb4f3b", alpha=0.95, marker="X",
            label=f"GT signatures ({gt_n})",
            edgecolors="white", linewidths=0.8, zorder=4,
        )
        for idx, (x, y) in enumerate(gt_xy):
            ax.text(x, y, f"G{idx}", fontsize=8, color="#631f17", ha="left", va="bottom")

    if golden_n:
        golden_scatter = ax.scatter(
            golden_xy[:, 0], golden_xy[:, 1],
            s=120,
            c=golden_seed_scores if golden_seed_scores is not None and len(golden_seed_scores) == golden_n else "#c7a23a",
            cmap="YlOrBr",
            alpha=0.95, marker="P",
            label=f"Golden seeds ({golden_n})",
            edgecolors="black", linewidths=0.7, zorder=3,
        )
        for idx, (x, y) in enumerate(golden_xy):
            ax.text(x, y, f"Y{idx}", fontsize=8, color="#5a4200", ha="right", va="bottom")
        if golden_seed_scores is not None and len(golden_seed_scores) == golden_n:
            cbar = fig.colorbar(golden_scatter, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Golden seed score", rotation=270, labelpad=13)

    if clustering_n:
        cluster_scatter = ax.scatter(
            clustering_xy[:, 0], clustering_xy[:, 1],
            s=120,
            c=clustering_seed_scores if clustering_seed_scores is not None and len(clustering_seed_scores) == clustering_n else "#1d8a89",
            cmap="viridis",
            alpha=0.95, marker="o",
            label=f"Clustering seeds ({clustering_n})",
            edgecolors="black", linewidths=0.7, zorder=2,
        )
        for idx, (x, y) in enumerate(clustering_xy):
            ax.text(x, y, f"C{idx}", fontsize=8, color="#0f4d4c", ha="left", va="top")
        if clustering_seed_scores is not None and len(clustering_seed_scores) == clustering_n:
            cbar = fig.colorbar(cluster_scatter, ax=ax, fraction=0.03, pad=0.08)
            cbar.set_label("Clustering seed score", rotation=270, labelpad=15)

    ax.set_title("Query, GT, Golden, and Clustering Signature Layout", fontsize=13)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(True, color="#dbcdbb", alpha=0.45, linewidth=0.7)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return np.asarray(Image.open(buf).convert("RGB")), method_name


class App:
    def __init__(self, config: AppConfig):
        self.config = config
        self.state = ViewerState()

    def default_model_info(self):
        return {"default_model_dir": self.config.default_model_dir}

    def model_info(self, model_dir: str):
        assets = find_model_assets(model_dir)
        imports = _lazy_imports()
        cfg = load_cfg(assets["selected_config"], assets["model_dir"], assets["selected_checkpoint"], imports)
        ensure_test_datasets_registered(cfg)
        dataset_records = {
            name: imports["DatasetCatalog"].get(name)
            for name in cfg.DATASETS.TEST
        }
        default_dataset = cfg.DATASETS.TEST[0]
        eval_modes = list(cfg.MODEL.LATENT_FORMER.TEST.EVAL_MODES) if cfg.MODEL.META_ARCHITECTURE == "LatentFormer" else []
        return {
            **assets,
            "default_dataset": default_dataset,
            "config_files": assets["config_files"],
            "checkpoints": assets["checkpoints"],
            "selected_config": assets["selected_config"],
            "selected_checkpoint": assets["selected_checkpoint"],
            "display_config_files": [Path(path).name for path in assets["config_files"]],
            "display_checkpoints": [Path(path).name for path in assets["checkpoints"]],
            "display_datasets": {
                name: ellipsize_dataset_name(name)
                for name in dataset_records.keys()
            },
            "datasets": {
                name: [summarize_record(record, idx) for idx, record in enumerate(records)]
                for name, records in dataset_records.items()
            },
            "eval_modes": eval_modes,
            "image_count": len(dataset_records[default_dataset]),
            "device": cfg.MODEL.DEVICE,
        }

    def run(self, payload: dict):
        model_dir = payload["model_dir"]
        config_file = payload["config_file"]
        checkpoint = payload["checkpoint"]
        dataset_name = payload["dataset_name"]
        image_index = int(payload["image_index"])

        bundle = self.state.get_or_load_bundle(model_dir, config_file, checkpoint)
        cfg = bundle["cfg"]
        model = bundle["model"]
        imports = bundle["imports"]
        records = bundle["dataset_records"][dataset_name]
        metadata = imports["MetadataCatalog"].get(dataset_name)
        mapper = bundle["mappers"][dataset_name]
        record = records[image_index]
        mapped = prepare_single_input(record, mapper)

        with imports["torch"].no_grad():
            outputs = model([mapped])
        if not isinstance(outputs, dict):
            outputs = {"default": outputs}

        static_panels = [
            {
                "title": "Input Image",
                "image_base64": image_to_base64(draw_input_image(record)),
            },
            {
                "title": "Ground Truth",
                "image_base64": image_to_base64(draw_gt_overlay(record, metadata, imports)),
            },
        ]

        mode_panels = []
        for mode_name, mode_outputs in outputs.items():
            processed = mode_outputs[0]
            prediction_panels = draw_prediction_panels(record, processed, metadata, imports)
            mode_panels.append(
                {
                    "mode": mode_name,
                    "summary": summarize_outputs(processed),
                    "images": prediction_panels,
                }
            )

        combined_signature_panel = draw_combined_signature_projection(outputs)
        if combined_signature_panel is not None:
            combined_signature_panel["wide"] = True

        return {
            "checkpoint": checkpoint,
            "record_label": summarize_record(record, image_index)["label"],
            "static_panels": static_panels,
            "mode_panels": mode_panels,
            "signature_panels": ([combined_signature_panel] if combined_signature_panel is not None else []),
            "config_file": config_file,
            "dataset_name": dataset_name,
            "eval_modes": list(cfg.MODEL.LATENT_FORMER.TEST.EVAL_MODES),
        }


def summarize_outputs(processed: dict) -> str:
    parts = []
    if "panoptic_seg" in processed:
        _, segments = processed["panoptic_seg"]
        parts.append(f"{len(segments)} panoptic segments")
    if "instances" in processed:
        parts.append(f"{len(processed['instances'])} instances")
    if "sem_seg" in processed:
        parts.append(f"{processed['sem_seg'].shape[0]} semantic classes")
    return ", ".join(parts) if parts else "No visual outputs"


def ellipsize_dataset_name(name: str, max_len: int = 42) -> str:
    if len(name) <= max_len:
        return name
    keep = max(8, (max_len - 1) // 2)
    return f"{name[:keep]}…{name[-keep:]}"


class RequestHandler(BaseHTTPRequestHandler):
    app: App = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_PAGE)
            return
        if parsed.path == "/api/defaults":
            self._send_json(self.app.default_model_info())
            return
        if parsed.path == "/api/model-info":
            try:
                params = parse_qs(parsed.query)
                model_dir = params.get("model_dir", [""])[0]
                self._send_json(self.app.model_info(model_dir))
            except Exception as exc:  # pragma: no cover - surfaced to user
                self._send_error_json(exc)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            self._send_json(self.app.run(payload))
        except Exception as exc:  # pragma: no cover - surfaced to user
            self._send_error_json(exc)

    def log_message(self, format, *args):
        return

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, exc: Exception):
        self._send_json(
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive LatentFormer checkpoint viewer.")
    parser.add_argument(
        "--model-dir",
        default="outputs/h200/latentformer_R50_no_seed_hungarian_no_agg_for_seeds",
        help="Default model directory to open.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the web server.")
    parser.add_argument("--port", type=int, default=8877, help="Port to bind the web server.")
    return parser.parse_args()


def main():
    args = parse_args()
    app = App(
        AppConfig(
            default_model_dir=str(Path(args.model_dir).expanduser().resolve()),
            host=args.host,
            port=args.port,
        )
    )
    RequestHandler.app = app
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"LatentFormer viewer running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
