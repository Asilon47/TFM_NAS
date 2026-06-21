# `dataset/` — gate-pose data contract (D1 target)

The thesis target task (open decision **D1**, resolved 2026-06-18): **drone-racing gate
detection + 8-keypoint pose estimation**, in **Ultralytics YOLO-pose** format. This is the
training data for the user's deployed `yolo11n-pose` model and the dataset every NAS candidate
is fine-tuned + scored on (pose mAP / OKS), replacing the earlier ImageNet-classification framing.

This file is tracked; the image/label payload (~1.6 GB) is **gitignored** (`dataset/*`, like
`data/`). Never commit the images/labels.

## Task & classes

| Field | Value |
|---|---|
| `task` | `pose` |
| `nc` (classes) | `1` — `gate` |
| `kpt_shape` | `[8, 3]` — 8 keypoints, each `(x, y, visibility)` |
| input res (deployed) | 640 (FP16 TRT engine); training script default 320, multi-scale |
| source image size | 1640 × 1232 JPEG (RGB) |

## Keypoint schema

8 gate-corner keypoints (outer frame + inner aperture), with the horizontal-flip pairing
`flip_idx: [1, 0, 3, 2, 5, 4, 7, 6]` (swaps left↔right on flip augmentation):

| idx | name | idx | name |
|---|---|---|---|
| 0 | bottom_right_outer | 4 | bottom_right_inner |
| 1 | bottom_left_outer  | 5 | bottom_left_inner  |
| 2 | top_right_outer    | 6 | top_right_inner    |
| 3 | top_left_outer     | 7 | top_left_inner     |

## Label format

One `.txt` per image (`labels/{train,val}/<stem>.txt`), one line per gate instance, all
coordinates normalized to `[0, 1]`:

```
<cls> <cx> <cy> <w> <h>  <kx0> <ky0> <v0>  <kx1> <ky1> <v1>  ...  <kx7> <ky7> <v7>
```

- `cls` is always `0` (gate); `cx cy w h` is the box; then 8 × `(x, y, v)` triples.
- visibility `v`: `2` = labeled/visible, `0` = absent (its `x, y` are `0.0`). In the synthetic
  renders many views show only the 4 outer corners (inner set to `0 0 0`).

## Splits

`train: images/train`, `val: images/val`; `val_percentage ≈ 0.3334`, random split. Counts on
this machine: **2842 train / 140 val** images (1:1 with label files). No `test` split.

## Consuming the dataset

- Ultralytics reads `dataset/dataset.yaml` directly (`YOLO.val(data=...)` /
  `.train(data=...)`). **Gotcha:** the committed yaml's `path:` is a stale absolute path from
  the box it was authored on (`/root/workspace/20_jan_a2rl_WITH_SYNTH`). Ultralytics resolves
  `train`/`val` relative to `path`, so `detect.evaluate.resolve_data_yaml` rewrites it to the
  local `dataset/` into a temp copy at run time — **don't hand-edit the yaml**.
- Pose mAP (OKS) is reused from Ultralytics' validator (`metrics.pose.map`/`.map50`); we do not
  re-implement OKS. See `detect/evaluate.py`.

## Caveats

- **Synthetic source.** `…_WITH_SYNTH` — procedurally rendered gates (filenames encode
  altitude/azimuth/elevation sweeps). Expect a sim→real domain gap; real-image validation is a
  separate concern before deployment claims.
- **OKS sigmas.** COCO defines per-keypoint sigma (falloff) constants for 17 human joints; this
  custom 8-keypoint set has none, so Ultralytics falls back to a uniform default. Pose-mAP
  *absolute* values are therefore convention-dependent — fine for *ranking* archs (the NAS
  signal), but not directly comparable to COCO-pose numbers.
