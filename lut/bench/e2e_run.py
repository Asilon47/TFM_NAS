#!/usr/bin/env python3
"""End-to-end frame timing on the Jetson: preprocess + TRT inference + postprocess.

The isolated bench times only ``execute_async_v3`` — but the deployed pipeline (why yolo11n
ran at ~12 FPS ≈ 83 ms while the engine benches at 7.7 ms) also pays for image preprocessing
(letterbox-resize, normalize, HWC→CHW) and pose postprocess (conf filter, NMS, keypoint
decode), both on the Orin's CPU. This runs the whole frame N times and reports the mean of
each stage, so we can see what fraction inference actually is — the measurement that speaks to
the 12 FPS observation. Preprocess uses cv2 if present (matches ultralytics), else numpy.

Runs INSIDE the lut-runner container (has tensorrt/pycuda/numpy). Output: one JSON line.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
from run_bench import allocate_io, load_engine  # sibling in /bench

try:
    import cv2
    _HAVE_CV2 = True
except Exception:  # noqa: BLE001
    _HAVE_CV2 = False


def preprocess(frame: np.ndarray, imgsz: int, dtype) -> np.ndarray:
    """Letterbox-resize a HWC uint8 camera frame → NCHW normalized tensor (the deploy path)."""
    h, w = frame.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if _HAVE_CV2:
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:                                        # numpy nearest-neighbour fallback
        yi = (np.arange(nh) / r).astype(np.int32).clip(0, h - 1)
        xi = (np.arange(nw) / r).astype(np.int32).clip(0, w - 1)
        resized = frame[yi][:, xi]
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    img = canvas.astype(np.float32) / 255.0                 # normalize
    img = img.transpose(2, 0, 1)[None]                       # HWC→CHW, add batch
    return np.ascontiguousarray(img, dtype=dtype)


def postprocess(out: np.ndarray, conf: float = 0.25, iou: float = 0.7) -> int:
    """Representative pose postprocess: conf filter → NMS → keypoint reshape (timed, not used).

    Adapts to the engine's output layout ((1,F,A) or (1,A,F)); F = 4 box + nc + nkpt*3.
    Returns the surviving detection count (keeps the work honest / un-optimizable-away).
    """
    a = out[0]
    preds = a.T if a.shape[0] < a.shape[1] else a          # → (anchors, features)
    boxes = preds[:, :4].astype(np.float32)
    scores = preds[:, 4]
    if scores.max() > 1.0 or scores.min() < 0.0:            # logits → sigmoid
        scores = 1.0 / (1.0 + np.exp(-scores))
    keep = scores > conf
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) == 0:
        return 0
    # xywh → xyxy, then greedy NMS (numpy)
    xy, wh = boxes[:, :2], boxes[:, 2:4]
    x1y1, x2y2 = xy - wh / 2, xy + wh / 2
    x1, y1, x2, y2 = x1y1[:, 0], x1y1[:, 1], x2y2[:, 0], x2y2[:, 1]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    kept = []
    while order.size:
        i = order[0]
        kept.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ov = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][ov <= iou]
    _ = preds[kept, 5:].reshape(len(kept), -1)              # keypoint slice (decode cost)
    return len(kept)


def run(engine_path: Path, imgsz: int, iters: int, warmup: int) -> dict:
    engine = load_engine(engine_path)
    ctx = engine.create_execution_context()
    inputs, outputs, bindings = allocate_io(engine)
    for i in range(engine.num_io_tensors):
        ctx.set_tensor_address(engine.get_tensor_name(i), bindings[i])
    stream = cuda.Stream()
    inp, outp = inputs[0], outputs[0]
    frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)   # synthetic camera frame

    pre_ms, inf_ms, post_ms = [], [], []
    for it in range(iters + warmup):
        t0 = time.perf_counter()
        x = preprocess(frame, imgsz, inp["dtype"])
        t1 = time.perf_counter()
        cuda.memcpy_htod_async(inp["dev"], np.ascontiguousarray(x), stream)
        ev0, ev1 = cuda.Event(), cuda.Event()
        ev0.record(stream)
        ctx.execute_async_v3(stream.handle)
        ev1.record(stream)
        ev1.synchronize()
        for o in outputs:
            cuda.memcpy_dtoh_async(o["host"], o["dev"], stream)
        stream.synchronize()
        t2 = time.perf_counter()
        postprocess(outp["host"].reshape(outp["shape"]))
        t3 = time.perf_counter()
        if it >= warmup:
            pre_ms.append((t1 - t0) * 1000)
            inf_ms.append(ev1.time_since(ev0))
            post_ms.append((t3 - t2) * 1000)

    def stat(v):
        return {"mean": statistics.fmean(v), "p50": sorted(v)[len(v) // 2],
                "std": statistics.pstdev(v) if len(v) > 1 else 0.0}

    pre, inf, post = stat(pre_ms), stat(inf_ms), stat(post_ms)
    total = pre["mean"] + inf["mean"] + post["mean"]
    return {
        "preprocess_ms": pre, "inference_ms": inf, "postprocess_ms": post,
        "total_ms": total, "fps": 1000.0 / total,
        "inference_fraction": inf["mean"] / total,
        "cv2": _HAVE_CV2, "n": len(inf_ms),
        "output_shape": list(outp["shape"]),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    a = ap.parse_args()
    sys.stdout.write(json.dumps(run(Path(a.engine), a.imgsz, a.iters, a.warmup)) + "\n")
