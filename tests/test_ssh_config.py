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
    assert sweep == {"precision": "fp16"}


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
