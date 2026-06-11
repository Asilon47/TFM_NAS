# LUT Pipeline

Measures **latency, memory, params, and FLOPs** for every block in the catalog,
on a real Jetson Orin Nano running TensorRT (FP32), and writes them to
`data/lut.jsonl`.

- **Laptop** runs Python, holds the catalog, generates ONNX per block,
  orchestrates, and writes `data/lut.jsonl`.
- **Jetson** (connected by USB-C in device mode, `192.168.55.1`) runs a small
  Docker container that builds a TRT engine per block and times it.
- **No PyTorch on the Jetson** — keeps its 8 GB of RAM free for accurate memory
  measurements.

Schema details: [`lut/docs/schema.md`](docs/schema.md).

---

## 1. Physical setup

1. Flash JetPack 7.2 on the Jetson via NVIDIA SDK Manager.
2. Plug the Jetson's **USB-C data port** (the one on the module, not the barrel jack) into the laptop.
3. On the laptop, you should now see a new network interface and the Jetson reachable at `192.168.55.1`:
   ```bash
   ip addr | grep -A2 l4tbr0   # or similar interface name
   ping -c1 192.168.55.1
   ```
4. Copy your SSH key to the Jetson so the scripts can run without a password prompt:
   ```bash
   ssh-copy-id jetson@192.168.55.1
   ```
5. Edit `config.yaml` to match your Jetson's user/host/key path.

## 2. Pick a power mode (do this before benchmarking)

The LUT is only valid for the power mode that was active when it was measured. For Orin Nano 8GB the relevant modes are:

- `sudo nvpmodel -m 0`   — 15 W (maximum performance)
- `sudo nvpmodel -m 1`   — 7 W

Lock clocks to maximum within the chosen mode:
```bash
sudo jetson_clocks
```

Choose the mode you'll deploy under, and stick with it for the whole sweep.

## 3. Laptop bootstrap (LUT venv)

```bash
bash scripts/setup_laptop.sh
source .venv/bin/activate
```

Creates a CPU-only venv with `torch` (for ONNX export), `onnx`, `fabric`, etc.

## 4. Jetson bootstrap

```bash
bash scripts/setup_jetson.sh
```

Over SSH, this verifies Docker + the NVIDIA runtime, rsyncs `lut/bench/` to the
Jetson, pulls `nvcr.io/nvidia/l4t-tensorrt:r36.3.0-runtime`, and builds the
`lut-runner:latest` image on the Jetson. Smoke-tests `import tensorrt` inside
the container.

## 5. Probe the device once

```bash
python -m lut.orchestrate.probe_device
```

Writes `data/device_info.json` with peak DRAM bandwidth (measured with the CUDA
samples `bandwidthTest`), clocks, TRT version, and the active power mode.

Re-run this whenever you change power mode.

## 6. Sanity check

```bash
bash scripts/sanity_check.sh
```

End-to-end: 3 `conv3x3` configs go through the full pipeline. You should see
three rows appended to `data/lut.jsonl` with sub-millisecond `latency_ms.mean`
values. If this passes, the full sweep will work.

## 7. Full sweep

```bash
python -m lut.orchestrate.run_sweep
```

Prints the expected row count up-front. Appends to `data/lut.jsonl` as it goes.
Safe to Ctrl-C and re-run — already-measured rows (matched by `row_key`) are
skipped.

Narrow the sweep:
```bash
python -m lut.orchestrate.run_sweep --blocks mbconv conv3x3 dwconv
python -m lut.orchestrate.run_sweep --limit 50            # stop after 50 new rows
```

## 8. Extending the catalog

All blocks live in `catalog/` (at the project root — shared with the NAS
pipeline). To add a new block:

1. Add a tiny `nn.Module` to `catalog/mbconv.py` or `catalog/seg_det.py`.
2. Add a builder function + grid entry in `catalog/blocks.py`'s `BLOCK_REGISTRY`.
3. Re-run `python -m lut.orchestrate.run_sweep` — only the new rows are measured.

Existing LUT rows are never invalidated by adding new blocks or widening grids:
each row is keyed by `(block, cfg, input_shape)`, so the schema is append-only.

## 9. Consuming the LUT

```python
import pandas as pd
lut = pd.read_json("data/lut.jsonl", lines=True)

# All MBConv rows at 28x28
mbconv = lut[(lut.block == "mbconv") &
             (lut.input_shape.apply(lambda s: s[2] == 28))]

# Direct lookup for one cfg
row = lut[(lut.block == "mbconv") &
          (lut.cfg == {"in_c": 32, "out_c": 64, "kernel": 5,
                       "stride": 2, "expand": 6, "se": True})]
latency_ms = row.latency_ms.iloc[0]["mean"]
```

## 10. Generate a dummy LUT (no Jetson needed)

For NAS development when the Jetson is not available, generate a synthetic LUT
using roofline-model latency estimates:

```bash
python -m lut.orchestrate.gen_dummy_lut           # write data/lut.jsonl
python -m lut.orchestrate.gen_dummy_lut --overwrite  # replace existing file
```

The dummy rows use the same `row_key` scheme as the real pipeline, so NAS code
written against the dummy LUT will work unchanged against a real one.

## Layout

```
lut/
├── export/       torch module -> ONNX
├── bench/        TRT engine build + timed inference (runs in Docker on Jetson)
├── orchestrate/  laptop-side loop: push ONNX, run container, collect results
└── docs/         lut.jsonl + device_info.json schema

catalog/          block definitions + sweep grids  ← shared with NAS pipeline
data/             lut.jsonl + device_info.json (gitignored)
config.yaml       Jetson SSH + sweep settings
```
