"""The pure normalization in lut.orchestrate.bench_model (no SSH / Jetson).

The remote bench path is reused verbatim from run_sweep (Jetson-gated, integration-
only); what is unit-testable here is ``baseline_row`` — turning a bench result +
device metadata into the anchor JSON, including the achieved-bandwidth derivation
and the device-state stamps that make the latency comparable to the LUT.
"""
from lut.orchestrate.bench_model import baseline_row


def _bench(mean_ms: float = 8.0, peak: float = 120.0, io_bytes: int = 8_000_000) -> dict:
    return {
        "latency_ms": {"mean": mean_ms, "std": 0.1, "p50": mean_ms, "p95": mean_ms + 0.2, "n": 200},
        "peak_mem_mib": peak, "io_bytes": io_bytes, "trt_version": "10.3.0",
    }


_DEVICE = {"power_mode": 0, "jetpack": "6.0", "clocks_locked": True}


def test_baseline_row_passes_through_latency_and_stamps_device_state():
    row = baseline_row(_bench(), name="yolo11n_pose_640", precision="fp32",
                       imgsz=640, device_info=_DEVICE)
    assert row["name"] == "yolo11n_pose_640"
    assert row["precision"] == "fp32"
    assert row["imgsz"] == 640
    assert row["latency_ms"]["mean"] == 8.0          # full distribution passthrough
    assert row["peak_mem_mib"] == 120.0
    assert row["power_mode"] == 0 and row["clocks_locked"] is True   # device stamps
    assert row["trt_version"] == "10.3.0"
    assert row["source"] == "jetson_trt"


def test_baseline_row_derives_achieved_bandwidth():
    # achieved_bw = io_bytes / latency_s / 1e9 = 8e6 / (8e-3) / 1e9 = 1.0 GB/s
    row = baseline_row(_bench(mean_ms=8.0, io_bytes=8_000_000), name="m",
                       precision="fp32", imgsz=640, device_info=_DEVICE)
    assert row["achieved_bw_gbps"] == 1.0


def test_baseline_row_handles_zero_latency_without_dividing():
    row = baseline_row(_bench(mean_ms=0.0), name="m", precision="fp16",
                       imgsz=640, device_info={})
    assert row["achieved_bw_gbps"] == 0.0            # guarded, not a ZeroDivisionError
    assert row["power_mode"] is None                  # absent device info -> None, not a crash
