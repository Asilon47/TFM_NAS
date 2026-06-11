# LUT schema (v1)

Two files, both under `data/`.

## `lut.jsonl` — one JSON row per measured `(block, cfg, input_shape)`

Append-only. Safe to delete and regenerate.

```json
{
  "row_key": "a7f1b9cc2de41b8c",
  "block": "mbconv",
  "cfg": {"in_c": 32, "out_c": 64, "kernel": 5, "stride": 2, "expand": 6, "se": true, "res": 56},
  "input_shape": [1, 32, 56, 56],
  "precision": "fp16",
  "latency_ms": {"mean": 0.412, "std": 0.008, "p50": 0.410, "p95": 0.425, "n": 200},
  "peak_mem_mib": 18.7,
  "params": 12384,
  "flops": 8123456,
  "achieved_bw_gbps": 24.1,
  "trt_version": "8.6.2",
  "power_mode": "15W",
  "jetpack": "6.0",
  "timestamp": "2026-04-22T10:30:00Z"
}
```

Notes:
- `row_key` is a 16-hex sha1 of `(block, cfg, input_shape)`; used for resumability.
  **`precision` is deliberately NOT part of the key**: rows measured at different
  precisions coexist in the file under the same `row_key`. Always filter to one
  precision before keying rows in memory (`lut/loader.py:load_lut` does this and
  raises on collisions); `completed_keys(path, precision=...)` makes resume
  precision-aware, so changing `sweep.precision` re-measures instead of skipping.
- Rows written by `gen_dummy_lut` carry `"source": "roofline_dummy"` so dummy
  estimates are distinguishable from real Jetson measurements in a mixed file.
  Real rows have no `source` field.
- The golden hashes in `tests/test_row_key.py` pin this key contract; a change
  that re-keys rows fails those tests by design.
- `latency_ms` values measured via CUDA events with `n` timed iterations after 50 warmups.
- `peak_mem_mib` comes from `cudaMemGetInfo` delta around the timed loop. Noisy: treat as an upper bound, not a precise reading.
- `flops` is a static estimate (Conv + Linear multiply-adds, counted via forward hooks). Useful as a predictive-model feature; not a deployment figure.
- `achieved_bw_gbps = sum(IO tensor bytes) / latency_mean`. Good sanity check against the device's peak DRAM bandwidth.

## `device_info.json` — device-level constants (one per power mode)

```json
{
  "device": "Jetson Orin Nano",
  "power_mode": "15W",
  "gpu_clock_mhz_max": 1020,
  "emc_clock_mhz": 3200,
  "peak_dram_gbps_measured": 51.2,
  "trt_version": "8.6.2",
  "cuda_version": "12.2",
  "probed_at": "2026-04-22T10:00:00Z"
}
```

If you switch the Jetson's power mode (`sudo nvpmodel -m N`), re-run
`python -m lut.orchestrate.probe_device` and archive the previous file — the LUT
rows only apply to the power mode that was active when they were measured.

## Reading the LUT from your NAS code

Use the validated loader (filters by precision, refuses duplicate keys):

```python
from pathlib import Path
from lut.loader import load_lut

lut = load_lut(Path("data/lut.jsonl"), precision="fp16")  # row_key -> row
lat = lut[some_row_key]["latency_ms"]["mean"]
```

Pandas works too for exploration
(`pd.read_json("data/lut.jsonl", lines=True)`), but remember to filter by
`precision` before keying anything by `row_key`.
