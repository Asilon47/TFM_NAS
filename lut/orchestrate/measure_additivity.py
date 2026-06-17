"""On-device whole-subnet benchmarking — the measured side of CP 2.2's additivity DoD.

``search.cost.cost`` SUMS per-block LUT latencies; the DoD compares that sum against
the *measured* whole-subnet latency, binned by depth (``search.validate_additivity``).
The summed side is already in ``data/additivity_subnets.json`` (``search.additivity_preview
manifest``); this driver fills the measured side by benchmarking each pinned subnet as
ONE TensorRT engine on the Jetson, reusing ``run_sweep``'s bench path verbatim.

It is the on-device sibling of ``run_sweep.py`` and the composition root for this step,
so it imports both ``search.*`` (the export + manifest) and ``lut.orchestrate.*`` (the
bench). Whole-net latencies are NOT LUT rows — a whole subnet has no per-block
``row_key`` — so they are written ONLY to the manifest (``measured_ms``) and, with
``--with-stem-head``, to ``data/stem_head_offset.json`` (the absolute cost offset).

Idempotent: entries already carrying ``measured_ms`` are skipped, so a re-run resumes.
The measured net is built from the same ``catalog`` blocks the per-block LUT timed
(see ``search.export_subnet``), so measured-minus-summed isolates cross-block fusion.

Usage:
  python -m lut.orchestrate.measure_additivity                  # measure all pending
  python -m lut.orchestrate.measure_additivity --limit 1        # smoke one subnet
  python -m lut.orchestrate.measure_additivity --with-stem-head # + calibrate the offset
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from catalog.flops import count_flops
from lut.orchestrate.probe_device import probe, write_device_info
from lut.orchestrate.run_sweep import preflight_verdict, run_remote_bench
from lut.orchestrate.ssh_client import connect, load_config
from search.additivity_preview import DEFAULT_MANIFEST, load_manifest, write_manifest
from search.cost import DEFAULT_OFFSET_PATH, offset_from_measurements
from search.export_subnet import (
    HEAD_INPUT_SHAPE,
    STEM_INPUT_SHAPE,
    build_head,
    build_stem,
    export_head,
    export_stem,
    export_subnet,
)

ROOT = Path(__file__).resolve().parents[2]


def bench_to_component(bench_result: dict, params: int, flops: int) -> dict:
    """Normalize a remote bench result + laptop-side params/flops into a cost component.

    The remote bench reports a latency distribution + working-set peak; params (from
    the export) and flops (a CPU count) are known laptop-side. Returns the flat
    ``{latency_ms, peak_mem_mib, params, flops}`` shape consumed by ``measured_ms`` and
    ``search.cost.offset_from_measurements``.
    """
    return {
        "latency_ms": bench_result["latency_ms"]["mean"],
        "peak_mem_mib": bench_result["peak_mem_mib"],
        "params": params,
        "flops": flops,
    }


def pending_entries(entries: list[dict]) -> list[dict]:
    """Manifest entries still needing a measurement (the idempotent-resume contract)."""
    return [e for e in entries if e.get("measured_ms") is None]


def _bench_onnx(conn, cfg, sweep_cfg: dict, job_id: str, onnx_path: Path,
                precision: str) -> dict:
    """Build + benchmark one ONNX on the device via run_sweep's reused bench path."""
    return run_remote_bench(
        conn, cfg, job_id, onnx_path, precision=precision,
        warmup=int(sweep_cfg.get("warmup_iters", 50)),
        iters=int(sweep_cfg.get("timed_iters", 200)),
        min_window_s=float(sweep_cfg.get("min_window_s", 0.5)),
    )


def _calibrate_stem_head(conn, cfg, sweep_cfg: dict, precision: str,
                         out_path: Path, tmp: Path) -> None:
    """Measure the fixed stem + head and write the absolute cost offset JSON."""
    components: dict[str, dict] = {}
    specs = [("stem", build_stem, STEM_INPUT_SHAPE, export_stem),
             ("head", build_head, HEAD_INPUT_SHAPE, export_head)]
    for name, builder, in_shape, exporter in specs:
        meta = exporter(tmp / f"{name}.onnx")
        flops = count_flops(builder().eval(), in_shape)
        bench = _bench_onnx(conn, cfg, sweep_cfg, name, tmp / f"{name}.onnx", precision)
        components[name] = bench_to_component(bench, meta["params"], flops)
        c = components[name]
        print(f"  {name}: latency={c['latency_ms']:.4g} ms  params={c['params']:,}  "
              f"peak={c['peak_mem_mib']:.4g} MiB", flush=True)
    offset = offset_from_measurements(components["stem"], components["head"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {**offset, "precision": precision, "components": components}, indent=2) + "\n")
    print(f"Stem/head offset -> {out_path}: latency={offset['latency_ms']:.4g} ms  "
          f"params={offset['params']:,}  peak={offset['peak_mem_mib']:.4g} MiB")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--precision", default=None,
                    help="override; default is sweep.precision from config (fp32)")
    ap.add_argument("--limit", type=int, default=0,
                    help="measure at most N pending subnets (0 = all). Use 1 to smoke.")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="trust device state instead of re-probing (bring-up/debug only)")
    ap.add_argument("--with-stem-head", action="store_true",
                    help="also measure the fixed stem+head -> data/stem_head_offset.json")
    ap.add_argument("--offset-out", type=Path, default=DEFAULT_OFFSET_PATH)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args(argv)

    cfg, sweep_cfg = load_config(Path(args.config))
    precision = args.precision or sweep_cfg.get("precision", "fp16")

    conn = connect(cfg)
    if not args.skip_preflight:
        # Same gate as run_sweep: a whole net measured under DVFS or the wrong power
        # mode is not comparable to the LUT it is validated against.
        device_info = probe(conn, cfg)
        write_device_info(
            device_info, ROOT / sweep_cfg.get("device_info_json", "data/device_info.json"))
        reason = preflight_verdict(device_info, cfg.power_mode,
                                   require_locked_clocks=cfg.lock_clocks)
        if reason is not None:
            raise SystemExit(f"[preflight] {reason}")
        print(f"Preflight OK: power_mode={device_info.get('power_mode')!r} "
              f"clocks_locked={device_info.get('clocks_locked')}", flush=True)

    conn.run(f"mkdir -p {cfg.remote_workdir}/job {cfg.remote_workdir}/cache", hide=True)

    entries = load_manifest(args.manifest)
    pending = pending_entries(entries)
    print(f"Manifest {args.manifest}: {len(entries)} subnets, {len(pending)} pending "
          f"(precision={precision}).", flush=True)

    n_new = 0
    with tempfile.TemporaryDirectory(prefix="additivity_onnx_") as tmpdir:
        tmp = Path(tmpdir)
        for e in pending:
            onnx_path = tmp / f"{e['id']}.onnx"
            meta = export_subnet(e["arch_dict"], onnx_path)
            bench = _bench_onnx(conn, cfg, sweep_cfg, e["id"], onnx_path, precision)
            comp = bench_to_component(bench, meta["params"], 0)
            e["measured_ms"] = comp["latency_ms"]
            e["measured_peak_mem_mib"] = comp["peak_mem_mib"]
            write_manifest(args.manifest, entries)   # persist after each (resume-safe)
            n_new += 1
            ratio = e["summed_ms"] / e["measured_ms"] if e["measured_ms"] else float("nan")
            print(f"  depth {e['depth']:2d} id={e['id']}: measured={e['measured_ms']:.4g} "
                  f"ms  summed={e['summed_ms']:.4g} ms  (summed/measured={ratio:.2f})",
                  flush=True)
            if args.limit and n_new >= args.limit:
                break

        print(f"\nMeasured {n_new} subnet(s); manifest -> {args.manifest}")
        if args.with_stem_head:
            print("Calibrating stem/head offset:")
            _calibrate_stem_head(conn, cfg, sweep_cfg, precision, args.offset_out, tmp)

    print("\nNext: python -m search.additivity_preview report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
