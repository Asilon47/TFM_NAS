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
  "trt_version": "10.3.0",
  "power_mode": "0",
  "jetpack": "R39.2.0",
  "clocks_locked": true,
  "source": "jetson_trt",
  "timestamp": "2026-06-12T10:30:00Z"
}
```

Notes:
- `row_key` is a 16-hex sha1 of `(block, cfg, input_shape)`; used for resumability.
  **`precision` is deliberately NOT part of the key**: rows measured at different
  precisions coexist in the file under the same `row_key`. Always filter to one
  precision before keying rows in memory (`lut/loader.py:load_lut` does this and
  raises on collisions); `completed_keys(path, precision=...)` makes resume
  precision-aware, so changing `sweep.precision` re-measures instead of skipping.
- **`precision: "fp32"` means TRT's default fp32, which allows TF32** tensor-core
  math on the Orin's Ampere GPU (trtexec is run without `--noTF32`). This is a
  deliberate decision (2026-06-12): the LUT predicts what a default TRT
  deployment actually does, not strict IEEE fp32. Rows older than this note
  (none — decided before the full sweep) would share the same semantics anyway.
- `source` tags provenance: `"jetson_trt"` for rows measured by
  `run_sweep` on the device, `"roofline_dummy"` for `gen_dummy_lut` estimates.
  Rows measured before 2026-06-12 lack the field (treat absent as real).
- The golden hashes in `tests/test_row_key.py` pin this key contract; a change
  that re-keys rows fails those tests by design.
- `latency_ms` is measured via CUDA events, queue depth 1 (one inference fully
  drained per sample — the right semantic for blocks executing sequentially in
  a net). `n` is the sample count: at least `timed_iters`, but sampling
  continues until the timed window spans `min_window_s` wall time (default
  0.5 s), so tiny blocks get many more samples than large ones. 50 warmup
  iterations precede the window. H2D/D2H copies are excluded.
- `peak_mem_mib` = TRT execution scratch (`engine.device_memory_size_v2`) +
  the block's resident IO buffers. Deterministic (TRT-reported), but it
  **excludes weights** (reconstruct as `params x bytes-per-param`) and the CUDA
  context/runtime overhead. **Do not sum it across blocks**: each inter-block
  tensor would be counted twice (output of block *i* = input of block *i+1*)
  and TRT reuses scratch across layers in a fused whole-net engine — a sane
  whole-net estimate is shaped like `sum(weights) + max_i(scratch_i + io_i)`,
  to be decided at cost-model time (CP 2.2+). Rows measured before 2026-06-12
  used a `cudaMemGetInfo` free-delta that is meaningless on Jetson's unified
  memory; none of those remain in the file.
- Dummy rows compute IO bytes at fp16 width (`FP16_BYTES`), while real engines
  keep fp32 IO bindings — dummy `peak_mem_mib`/`achieved_bw_gbps` are on a
  different basis and are not comparable to measured rows.
- `flops` is a static estimate (Conv + Linear multiply-adds, counted via forward hooks). Useful as a predictive-model feature; not a deployment figure.
- `achieved_bw_gbps = sum(IO tensor bytes) / latency_mean`. Good sanity check against the device's peak DRAM bandwidth.

## `device_info.json` — device-level constants (one per power mode)

```json
{
  "device": "Jetson Orin Nano",
  "power_mode": "0",
  "gpu_clock_mhz_max": 612,
  "gpu_clock_mhz_cur": 612,
  "clocks_locked": true,
  "emc_clock_mhz": 2133,
  "peak_dram_gbps_measured": 62.8,
  "trt_version": "10.3.0",
  "cuda_version": "12.6.68",
  "jetpack": "R39.2.0",
  "probed_at": "2026-06-12T10:00:00Z"
}
```

- `clocks_locked` is true when the GPU devfreq governor pins
  `min_freq == max_freq` (what `jetson_clocks` does). **`jetson_clocks` does
  not survive a reboot**, which is why `run_sweep` re-probes the device and
  rewrites this file at every sweep start (the preflight), refusing to measure
  when clocks are unlocked or the power mode doesn't match `config.yaml`.
  `--skip-preflight` bypasses the probe and trusts the file as-is.
- If you switch power mode (`sudo nvpmodel -m N`), the preflight will catch the
  mismatch; rows only apply to the power mode active when they were measured.

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
