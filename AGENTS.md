This is a fork of the Mask2Former repo that is trying to implement a novel architecture called LatentFormer. It's design an principles can be read in `LATENT_ARCH.md`.

This is a shared work environment, so be careful.
- We have assigned an H200 node that we can use with `tsp gpu exec command`. We can only use an H200 at a time.
- Datasets are in `/data/datasets/`. Don't write or break anything there.
- In this environment we need to use `apptainer`. The SIF file `/data/andy.barcia/fcclip-torch27-cu126.sif` should have everything we need to execute this model.

For training/evaluation, `train.sh` is the LatentFormer launcher for this repo. It uses `apptainer` with the SIF above, bind-mounts the repo into the container, mounts `/data/datasets` read-only, and wraps execution in `tsp gpu exec` by default.
- Default mode is `RUN_MODE=smoke`, which runs `configs/latentformer/latentformer_R50_bs16_50ep.yaml` for a short wiring check (`MAX_ITER=20`, `IMS_PER_BATCH=1`).
- Use `RUN_MODE=train ./train.sh` for normal resumed training, or `RUN_MODE=eval ./train.sh` for `--eval-only`.
- Useful overrides include `OUTPUT_DIR=...`, `CONFIG_FILE=...`, `NUM_GPUS=...`, `IMS_PER_BATCH=...`, `MAX_ITER=...`, and `USE_TSP=0` if already inside an allocated GPU environment.
- Wrapper logs are written under `${OUTPUT_DIR}/logs/`, alongside Detectron outputs for the run.
