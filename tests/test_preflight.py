"""preflight_verdict: which device states are fit to measure.

The verdict runs on a FRESH probe at sweep start. Hard conditions (unlocked
clocks, power-mode mismatch) must abort — they silently corrupt rows
otherwise. The bandwidth probe failing only degrades sanity checks, so it
warns. Policy ownership: TODO(user) in run_sweep.preflight_verdict.
"""
import pytest

pytest.importorskip("fabric", reason="run_sweep imports ssh_client -> fabric")
pytest.importorskip("tqdm", reason="run_sweep imports tqdm")
pytest.importorskip("torch", reason="run_sweep imports catalog -> torch")

from lut.orchestrate.run_sweep import preflight_verdict  # noqa: E402

GOOD = {
    "device": "Jetson Orin Nano",
    "power_mode": "0",
    "gpu_clock_mhz_max": 612,
    "gpu_clock_mhz_cur": 612,
    "clocks_locked": True,
    "emc_clock_mhz": 2133,
    "peak_dram_gbps_measured": 62.8,
}


def test_good_state_passes():
    assert preflight_verdict(GOOD, expected_power_mode=0) is None


def test_unlocked_clocks_abort():
    info = {**GOOD, "clocks_locked": False}
    reason = preflight_verdict(info, expected_power_mode=0)
    assert reason is not None and "setup_jetson" in reason


def test_missing_clocks_key_aborts():
    # An old probe payload without the field must not pass as "locked".
    info = {k: v for k, v in GOOD.items() if k != "clocks_locked"}
    assert preflight_verdict(info, expected_power_mode=0) is not None


def test_unlocked_tolerated_when_not_required():
    info = {**GOOD, "clocks_locked": False}
    assert preflight_verdict(info, 0, require_locked_clocks=False) is None


def test_power_mode_mismatch_aborts_naming_both():
    reason = preflight_verdict(GOOD, expected_power_mode=1)
    assert reason is not None and "'0'" in reason and "1" in reason


def test_unknown_power_mode_aborts():
    info = {**GOOD, "power_mode": None}
    assert preflight_verdict(info, expected_power_mode=0) is not None


def test_no_expected_mode_skips_the_check():
    info = {**GOOD, "power_mode": "unknown"}
    assert preflight_verdict(info, expected_power_mode=None) is None


def test_mode_comparison_normalizes_str_vs_int():
    info = {**GOOD, "power_mode": 0}  # int from a hypothetical probe variant
    assert preflight_verdict(info, expected_power_mode=0) is None


def test_bandwidth_zero_warns_but_proceeds(capsys):
    info = {**GOOD, "peak_dram_gbps_measured": 0.0}
    assert preflight_verdict(info, expected_power_mode=0) is None
    assert "WARN" in capsys.readouterr().err
