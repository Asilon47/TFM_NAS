"""Run bench/device_probe.sh on the Jetson and stash the JSON in data/.

`probe()` + `write_device_info()` are also the sweep-start preflight's code
path (lut/orchestrate/run_sweep.py): one probe implementation, two callers.
"""
import argparse
import json
from pathlib import Path

from fabric import Connection

from lut.orchestrate.ssh_client import JetsonConfig, connect, load_config

ROOT = Path(__file__).resolve().parents[2]


def probe(conn: Connection, cfg: JetsonConfig) -> dict:
    """Run device_probe.sh in the runner container; return its parsed JSON."""
    cmd = (
            f"docker run --rm --runtime nvidia --privileged "
            f"-v /sys/kernel/debug:/sys/kernel/debug:ro "
            f"-v /var/lib/nvpmodel:/var/lib/nvpmodel:ro "
            f"-v /etc/nv_tegra_release:/etc/nv_tegra_release:ro "
            f"-v {cfg.remote_workdir}/bench:/bench {cfg.docker_image} "
            f"bash /bench/device_probe.sh"
        )
    res = conn.run(cmd, hide=True, warn=True)
    if res.return_code != 0:
        raise SystemExit(f"device_probe.sh failed:\n{res.stderr}")
    raw_output = res.stdout.strip()
    try:
        # Extract the JSON payload between the first '{' and the last '}' —
        # docker/driver noise may precede or follow it on stdout.
        start_idx = raw_output.find('{')
        end_idx = raw_output.rfind('}')
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON brackets found in the output.")
        return json.loads(raw_output[start_idx:end_idx + 1])
    except (json.JSONDecodeError, ValueError) as e:
        # Print exactly what the Jetson sent back so it's debuggable.
        raise SystemExit(
            f"Failed to parse JSON. Raw output from Jetson was:\n"
            f"--- START RAW OUTPUT ---\n{raw_output}\n--- END RAW OUTPUT ---\n"
            f"Error: {e}"
        ) from e


def write_device_info(info: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(info, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    cfg, sweep_cfg = load_config(Path(args.config))
    dest = ROOT / sweep_cfg.get("device_info_json", "data/device_info.json")

    info = probe(connect(cfg), cfg)
    write_device_info(info, dest)
    print(f"Wrote {dest}:\n{json.dumps(info, indent=2)}")


if __name__ == "__main__":
    main()
