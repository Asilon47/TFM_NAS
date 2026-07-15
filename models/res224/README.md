# models/res224/ — low-res probe exports (Phase 10 / CP 10.1)

Static @224 ONNX exports used as the GAP8 feasibility-probe inputs
(`mcu/probes/`). The `.onnx` files are gitignored (regeneratable, ~10 MB
each); the `.meta.json` sidecars and this README are tracked.

| file | what | opset | provenance |
|---|---|---|---|
| `graft_noneck_224.onnx` | winner-v1 arch (d=[2,2,4,3,3]) full graft: OFA backbone + adapters + YOLO11 pose head | 17 | exported 2026-07-15 (first low-res probe, same session as the CPU cross-device rank check — see procedure.md) |
| `graft_noneck_224_op12.onnx` | same graph, opset-12 re-export (importer-compat fallback) | 12 | CP 10.1, 2026-07-15 |
| `graft_backbone_224.onnx` | backbone-only (P3/P4/P5 taps), drops the pose head's Expand/Shape/Softmax risk set | 12 | CP 10.1, 2026-07-15 |
| `graft_noneck_224_mcu.onnx` | full graft, `--mcu-act` (OFA h-swish/h-sigmoid → HardSigmoid forms — the variant that passes nntool fusions+int8 aquant) | 12 | CP 10.1, 2026-07-15 |
| `graft_backbone_224_mcu.onnx` | backbone-only, `--mcu-act` — a 6-op-type graph (Conv/HardSigmoid/Mul/Add/ReduceMean/Relu) | 12 | CP 10.1, 2026-07-15 |
| `yolo11n_pose_224.onnx` | deployed baseline at probe res (probe B) | 12 | first low-res probe, 2026-07-15 |
| `dense_ctrl_n_224.onnx` | dense-family n-scale control (same op profile as yolo11n) | 12 | first low-res probe, 2026-07-15 |

## Regenerate

Graft exports (`.venv-nas`; winner record = `state/winner_v1/winner.json`):

```bash
python -m detect.export_grafted_onnx --imgsz 224 --out models/res224/graft_noneck_224.onnx
python -m detect.export_grafted_onnx --imgsz 224 --opset 12 --out models/res224/graft_noneck_224_op12.onnx
python -m detect.export_grafted_onnx --backbone-only --imgsz 224 --opset 12 --out models/res224/graft_backbone_224.onnx
python -m detect.export_grafted_onnx --imgsz 224 --opset 12 --mcu-act --out models/res224/graft_noneck_224_mcu.onnx
python -m detect.export_grafted_onnx --backbone-only --imgsz 224 --opset 12 --mcu-act --out models/res224/graft_backbone_224_mcu.onnx
```

yolo11n / dense control: Ultralytics ONNX export of the respective donor at
`imgsz=224, opset=12` (see procedure.md "CPU cross-device rank check").
