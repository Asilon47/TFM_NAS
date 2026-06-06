"""Run bench/device_probe.sh on the Jetson and stash the JSON in data/."""
import argparse
import json
from pathlib import Path

from lut.orchestrate.ssh_client import load_config, connect

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    cfg, sweep_cfg = load_config(Path(args.config))
    dest = ROOT / sweep_cfg.get("device_info_json", "data/device_info.json")
    dest.parent.mkdir(parents=True, exist_ok=True)

    conn = connect(cfg)
    cmd = (
        f"docker run --rm --runtime nvidia "
        f"-v {cfg.remote_workdir}/bench:/bench {cfg.docker_image} "
        f"bash /bench/device_probe.sh"
    )
    res = conn.run(cmd, hide=True, warn=True)
    if res.return_code != 0:
        raise SystemExit(f"device_probe.sh failed:\n{res.stderr}")
    info = json.loads(res.stdout)
    dest.write_text(json.dumps(info, indent=2))
    print(f"Wrote {dest}:\n{json.dumps(info, indent=2)}")


if __name__ == "__main__":
    main()
