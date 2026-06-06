# LUT schema (v1)

Two files, both under `data/`.

## `lut.jsonl` — one JSON row per measured `(block, cfg, input_shape)`

Append-only. Safe to delete and regenerate.

```json
{
  "row_key": "a7f1b9cc2de41b8c",
  "block": "mbconv",
  "cfg": {"in_c": 32, "out_c": 64, "kernel": 5, "stride": 2, "expand": 6, "se": true},
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
`python -m orchestrate.probe_device` and archive the previous file — the LUT
rows only apply to the power mode that was active when they were measured.

## Reading the LUT from your NAS code

```python
import pandas as pd
lut = pd.read_json("data/lut.jsonl", lines=True)
lat = lut.query("block == 'mbconv' and cfg == @target_cfg").latency_ms.iloc[0]["mean"]
```

Or build an in-memory dict keyed by `row_key` for O(1) lookup.
