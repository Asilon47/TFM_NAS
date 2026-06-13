"""load_config: misconfiguration must name the config path and every gap."""
import pytest

pytest.importorskip("fabric", reason="ssh_client imports fabric")

import yaml  # noqa: E402

from lut.orchestrate.ssh_client import load_config  # noqa: E402

FULL = {
    "jetson": {
        "host": "192.168.55.1",
        "user": "nvidia",
        "ssh_key": "~/.ssh/id_ed25519",
        "remote_workdir": "/home/nvidia/lut_runner",
        "docker_image": "lut-runner:latest",
    },
    "sweep": {"precision": "fp16"},
}


def _write(tmp_path, data):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_happy_path_expands_user(tmp_path):
    cfg, sweep = load_config(_write(tmp_path, FULL))
    assert cfg.host == "192.168.55.1"
    assert "~" not in cfg.ssh_key  # expanduser applied
    assert cfg.port == 22  # FULL has no 'port' → default holds
    assert sweep == {"precision": "fp16"}


def test_port_override(tmp_path):
    data = {"jetson": {**FULL["jetson"], "port": 5100}}
    cfg, _ = load_config(_write(tmp_path, data))
    assert cfg.port == 5100
    assert isinstance(cfg.port, int)


def test_power_mode_and_lock_clocks_parsed(tmp_path):
    data = {"jetson": {**FULL["jetson"], "power_mode": 0, "lock_clocks": False}}
    cfg, _ = load_config(_write(tmp_path, data))
    assert cfg.power_mode == 0
    assert cfg.lock_clocks is False


def test_power_mode_defaults(tmp_path):
    cfg, _ = load_config(_write(tmp_path, FULL))
    assert cfg.power_mode is None  # preflight then skips the mode check
    assert cfg.lock_clocks is True  # but still requires locked clocks


def test_local_overlay_overrides_endpoint(tmp_path):
    _write(tmp_path, FULL)
    (tmp_path / "config.local.yaml").write_text(
        yaml.safe_dump({"jetson": {"host": "10.0.0.9", "port": 5100}}))
    cfg, sweep = load_config(tmp_path / "config.yaml")
    assert cfg.host == "10.0.0.9"
    assert cfg.port == 5100
    assert cfg.user == "nvidia"  # not in the overlay → inherited
    assert sweep == {"precision": "fp16"}  # untouched section survives


def test_local_overlay_can_override_sweep(tmp_path):
    _write(tmp_path, FULL)
    (tmp_path / "config.local.yaml").write_text(
        yaml.safe_dump({"sweep": {"precision": "fp32"}}))
    _, sweep = load_config(tmp_path / "config.yaml")
    assert sweep == {"precision": "fp32"}


def test_local_overlay_must_be_mapping(tmp_path):
    _write(tmp_path, FULL)
    (tmp_path / "config.local.yaml").write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="config.local.yaml"):
        load_config(tmp_path / "config.yaml")


def test_missing_jetson_section(tmp_path):
    with pytest.raises(ValueError, match="jetson"):
        load_config(_write(tmp_path, {"sweep": {}}))


def test_missing_keys_all_named_at_once(tmp_path):
    data = {"jetson": {k: v for k, v in FULL["jetson"].items()
                       if k not in ("ssh_key", "docker_image")}}
    with pytest.raises(ValueError) as exc:
        load_config(_write(tmp_path, data))
    msg = str(exc.value)
    assert "jetson.ssh_key" in msg
    assert "jetson.docker_image" in msg
    assert "config.yaml" in msg


def test_missing_sweep_section_defaults_empty(tmp_path):
    cfg, sweep = load_config(_write(tmp_path, {"jetson": FULL["jetson"]}))
    assert sweep == {}
