"""Winner-v2-OFA Track 2 — the remote entry's command composer (pure)."""
import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _entry():
    spec = importlib.util.spec_from_file_location(
        "run_prune_graft", ROOT / "colab" / "run_prune_graft.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ns(**over):
    base = dict(out_dir=Path("/content/tfm_out"), device="cuda", batch=16, lr=1e-3, epochs=100,
                seed=0, ckpt_every=10, max_steps=None, spec=None, ratios="0.50",
                technique="global_taylor", arch_json=None, kd=True, teacher_pt=None,
                kd_alpha=1.0)
    base.update(over)
    return argparse.Namespace(**base)


def test_compose_passes_batch_and_lr():
    m = _entry()
    cmd = m.compose_recover_cmd(_ns(batch=48, lr=0.003), donor=Path("/d/g.pt"),
                                data_yaml=Path("/d/d.yaml"), python="python")
    assert "--batch 48" in cmd and "--lr 0.003" in cmd


def test_compose_max_steps_only_when_set():
    m = _entry()
    donor, yaml = Path("/d/gate_best.pt"), Path("/d/dataset.yaml")
    assert "--max-steps" not in m.compose_recover_cmd(_ns(), donor=donor, data_yaml=yaml,
                                                      python="python")
    assert "--max-steps 8" in m.compose_recover_cmd(_ns(max_steps=8), donor=donor,
                                                     data_yaml=yaml, python="python")


def test_compose_spec_run_with_default_kd_teacher():
    m = _entry()
    cmd = m.compose_recover_cmd(_ns(spec="prune/specs/v2_act292.json"),
                                donor=Path("/d/gate_best.pt"),
                                data_yaml=Path("/d/dataset.yaml"), python="python")
    assert "--ratio-spec prune/specs/v2_act292.json" in cmd
    assert "--ratios" not in cmd                      # spec overrides the ladder
    assert "--teacher /d/gate_best.pt --kd-alpha 1.0" in cmd
    assert "--ckpt-every 10" in cmd and "--seed 0" in cmd and "--epochs 100" in cmd


def test_compose_ratio_run_and_probe_and_no_kd():
    m = _entry()
    cmd = m.compose_recover_cmd(_ns(), donor=Path("/d/gate_best.pt"),
                                data_yaml=Path("/d/dataset.yaml"), python="python")
    assert "--technique global_taylor" in cmd and "--ratios 0.50" in cmd

    probe = m.compose_recover_cmd(
        _ns(spec="prune/specs/u30.json", arch_json="prune/specs/minact_arch.json"),
        donor=Path("/d/gate_best.pt"), data_yaml=Path("/d/dataset.yaml"), python="python")
    assert "--arch-json prune/specs/minact_arch.json" in probe
    assert "--ratio-spec prune/specs/u30.json" in probe
    assert "--technique global_taylor" in probe   # spec runs still carry the importance

    nokd = m.compose_recover_cmd(_ns(kd=False), donor=Path("/d/gate_best.pt"),
                                 data_yaml=Path("/d/dataset.yaml"), python="python")
    assert "--teacher" not in nokd


def test_compose_teacher_override_for_track_2t():
    m = _entry()
    cmd = m.compose_recover_cmd(_ns(teacher_pt="/d/yolo11x_gate.pt", kd_alpha=0.5),
                                donor=Path("/d/gate_best.pt"),
                                data_yaml=Path("/d/dataset.yaml"), python="python")
    assert "--teacher /d/yolo11x_gate.pt --kd-alpha 0.5" in cmd
