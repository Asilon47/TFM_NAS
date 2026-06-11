"""Thin wrapper around fabric.Connection for the orchestrator.

Centralizes host/user/key resolution so run_sweep.py and resume.py both read
the same config. Nothing here is Jetson-specific — it's just SSH.
"""
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional

import yaml
from fabric import Connection


@dataclass
class JetsonConfig:
    host: str
    user: str
    ssh_key: str
    remote_workdir: str
    docker_image: str


_REQUIRED_JETSON_KEYS = ("host", "user", "ssh_key", "remote_workdir", "docker_image")


def load_config(path: Optional[Path] = None) -> tuple[JetsonConfig, dict]:
    """Read config.yaml; fail with every missing key named at once.

    ``jetson.power_mode`` / ``jetson.lock_clocks`` are deliberately not part
    of JetsonConfig — they are consumed by scripts/setup_jetson.sh (awk),
    not by Python.
    """
    path = path or Path(__file__).resolve().parents[2] / "config.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or not isinstance(raw.get("jetson"), dict):
        raise ValueError(f"{path}: missing required 'jetson:' section")
    j = raw["jetson"]
    missing = [k for k in _REQUIRED_JETSON_KEYS if k not in j]
    if missing:
        raise ValueError(
            f"{path}: missing required key(s): "
            + ", ".join(f"jetson.{k}" for k in missing)
        )
    cfg = JetsonConfig(
        host=j["host"], user=j["user"],
        ssh_key=os.path.expanduser(j["ssh_key"]),
        remote_workdir=j["remote_workdir"],
        docker_image=j["docker_image"],
    )
    return cfg, raw.get("sweep", {})


def connect(cfg: JetsonConfig) -> Connection:
    return Connection(
        host=cfg.host, user=cfg.user,
        connect_kwargs={"key_filename": cfg.ssh_key},
    )
