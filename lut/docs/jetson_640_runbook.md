# Jetson runbook — @640 LUT re-sweep + baseline anchor (CP 3.3)

Two Jetson-gated measurements CP 3.3 needs, both run from the **laptop** (`.venv`,
fabric over SSH); the Jetson only ever runs TensorRT inside the `lut-runner` docker.

1. **@640 LUT re-sweep** — the pose backbone deploys at 640, where every per-block
   input resolution re-keys vs the @224 ImageNet grid. The catalog already carries the
   91 new @640 MBConv rows (append-only; commit `988e543`); this measures them so
   `search.cost.cost(arch, lut, res=640)` and `search.bo --res 640` have real latencies.
2. **Baseline anchor** — the deployed **yolo11n-pose** @640 latency, which sets the
   hard ceiling `T_max = min(baseline, fps_to_ms(60)=16.7 ms)` for `search.bo` / D4.

Prerequisites (one-time): `ssh-copy-id <user>@<jetson>`, real endpoint in the
gitignored `config.local.yaml`, and `bash scripts/setup_jetson.sh` (builds the
`lut-runner` image). See `lut/docs/schema.md` for the row/precision schema.

---

## 0. Lock device state (laptop)

```bash
source .venv/bin/activate
bash scripts/setup_jetson.sh        # power mode + jetson_clocks (does NOT survive reboot)
```
Every command below re-probes the device at start and **refuses to measure** with
unlocked clocks or the wrong power mode (`run_sweep.preflight_verdict`), so a
post-reboot session fails fast instead of stamping bad rows.

## 1. @640 LUT re-sweep (idempotent — only the new rows)

```bash
python -m lut.orchestrate.run_sweep                 # skips the 2710 measured @224 rows;
                                                    # measures the 91 pending @640 rows
```
Resumable: Ctrl-C and re-run to continue; one bad row is logged and skipped, not fatal.
Verify when done:
```bash
python -m pytest tests/test_lut_keydrift.py -q      # completeness gate flips SKIP -> PASS at 2801/2801
python -m search.bo --structural --seeds 3 --budget 20 --res 640 --t-max-ms 16.7   # costs @640 cleanly
```

## 2. Baseline anchor (yolo11n-pose @640)

Export the ONNX where ultralytics lives (`.venv-nas` / Kaggle / Colab), then bench from
the laptop:
```bash
# (ultralytics host) static-shape ONNX, batch-1, 640x640:
python -m detect.export_baseline_onnx --weights yolo11n-pose.pt --imgsz 640
#   -> yolo11n_pose_640.onnx  (copy it to the laptop)

# (laptop) build the TRT engine on-device and time it:
python -m lut.orchestrate.bench_model --onnx yolo11n_pose_640.onnx --imgsz 640
#   -> data/baseline_anchor.json  {latency_ms, peak_mem_mib, ...}
```
**Precision:** defaults to `sweep.precision` (fp32) so the ceiling is like-for-like
with the fp32 LUT the candidates are costed against. The fp16 deploy figure is a
separate `--precision fp16` run (a Phase-8/9 number, not the `T_max` the search uses).

## 3. Restore + use

```bash
bash scripts/teardown_jetson.sh     # restore idle power mode / unlock clocks
```
Feed the result into the search:
```bash
T_max = min(<data/baseline_anchor.json latency_ms.mean>, 16.7)     # ms
python -m search.bo --res 640 --t-max-ms <T_max> --device cuda \
    --head-weights runs/pose/experiments/gate_baseline/weights/best.pt --freeze-head \
    --seeds 5 --budget 50          # (the accuracy half runs on GPU/Kaggle — see kaggle/README.md)
```

---

### Notes
- The @640 rows share the LUT's append-only schema (`sha1(block+cfg+input_shape)`); the
  measured @224 rows and the golden `row_key` hashes are untouched.
- `bench_model` results are **not** LUT rows (a whole model has no per-block key) — they
  live in their own JSON and must never be appended to `data/lut.jsonl`.
- Two iso-J λ-calibration reference backbones (MobileNetV3-large, EfficientNet-B0) can be
  benched the same way (`export_baseline_onnx --weights …` → `bench_model`) if you want
  measured λ anchors rather than published numbers — a CP 3.5 selection detail, optional here.
