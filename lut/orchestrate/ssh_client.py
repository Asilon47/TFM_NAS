"""Thin wrapper around fabric.Connection for the orchestrator.

Centralizes host/user/key resolution so run_sweep.py and resume.py both read
the same config. Nothing here is Jetson-specific — it's just SSH.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from fabric import Connection


@dataclass
class JetsonConfig:
    host: str
    user: str
    ssh_key: str
    remote_workdir: str
    docker_image: str
    port: int = 22
    # Expected device state, verified by run_sweep's preflight against what
    # the device actually reports (and applied by scripts/setup_jetson.sh).
    power_mode: int | None = None
    lock_clocks: bool = True


_REQUIRED_JETSON_KEYS = ("host", "user", "ssh_key", "remote_workdir", "docker_image")


def _merge_local_overrides(raw: dict, local_path: Path) -> dict:
    """Two-level merge: keys inside config.local.yaml's sections override the
    committed template's. The local file carries the real Jetson endpoint and
    is gitignored, so the public repo never learns it."""
    if not local_path.exists():
        return raw
    local = yaml.safe_load(local_path.read_text()) or {}
    if not isinstance(local, dict):
        raise ValueError(f"{local_path}: top level must be a mapping")
    merged = dict(raw)
    for section, overrides in local.items():
        base = merged.get(section)
        if isinstance(overrides, dict) and isinstance(base, dict):
            merged[section] = {**base, **overrides}
        else:
            merged[section] = overrides
    return merged


def load_config(path: Path | None = None) -> tuple[JetsonConfig, dict]:
    """Read config.yaml (+ sibling config.local.yaml overrides); fail with
    every missing key named at once.

    ``jetson.power_mode`` / ``jetson.lock_clocks`` are applied by
    scripts/setup_jetson.sh (awk) and verified by the run_sweep preflight
    (Python) — both read the same keys so the config stays the single
    source of truth for the measurement state.
    """
    path = path or Path(__file__).resolve().parents[2] / "config.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    raw = _merge_local_overrides(raw, path.with_name("config.local.yaml"))
    if not isinstance(raw.get("jetson"), dict):
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
        port=int(j.get("port", 22)),
        power_mode=int(j["power_mode"]) if "power_mode" in j else None,
        lock_clocks=bool(j.get("lock_clocks", True)),
    )
    return cfg, raw.get("sweep", {})


def connect(cfg: JetsonConfig) -> Connection:
    return Connection(
        host=cfg.host, user=cfg.user, port=cfg.port,
        connect_kwargs={"key_filename": cfg.ssh_key},
    )
