#!/usr/bin/env python3
"""Read the GAP8 net_bench CPX console and print measured cyc + FPS.

The firmware emits integer-only lines (GAP8 newlib-nano %f is unreliable):
    BENCH model=<s> res=<i> cyc=<i> nodes=<i> clk_us=<i> n=<i> fcl=<i>
This host side derives ms = clk_us/1000 and fps = 1e6/clk_us, appends every row
to data/mcu/board/<model>.txt, and prints a readable table. cyc is RANKING-ONLY
vs the sim column; ms/fps is the honest absolute.

Two input paths:
  --stdin  (robust, version-proof): pipe ANY console source into it, e.g. a
           cfclient console capture or a serial dump. Always works.
  --uri    (turnkey, best-effort): read the CPX console live over CRTP via cflib.
           The exact cflib CPX API varies by version; if it errors, fall back to
           --stdin or just watch cfclient's Console tab (the same CPX stream).
"""
import argparse
import pathlib
import re
import sys

_BENCH = re.compile(
    r"BENCH\s+model=(?P<model>\S+)\s+res=(?P<res>\d+)\s+cyc=(?P<cyc>\d+)\s+"
    r"nodes=(?P<nodes>\d+)\s+clk_us=(?P<clk_us>\d+)\s+n=(?P<n>\d+)\s+fcl=(?P<fcl>\d+)"
)


def parse_bench_line(line):
    """Parse one firmware console line -> dict with derived ms/fps, or None.

    fps is None when clk_us == 0 (a not-yet-timed / degenerate line).
    """
    m = _BENCH.search(line or "")
    if not m:
        return None
    rec = {k: int(v) for k, v in m.groupdict().items() if k != "model"}
    rec["model"] = m.group("model")
    rec["ms"] = rec["clk_us"] / 1000.0
    rec["fps"] = (1_000_000.0 / rec["clk_us"]) if rec["clk_us"] else None
    return rec


def _fmt(rec):
    fps = f"{rec['fps']:.3f}" if rec["fps"] is not None else "n/a"
    return (f"{rec['model']:<22} res={rec['res']}  cyc={rec['cyc']:>12,}  "
            f"ms={rec['ms']:>8.2f}  fps={fps:>7}  (n={rec['n']}, fcl={rec['fcl']} MHz)")


def stream(lines, out_dir):
    """Funnel an iterable of console lines: parse, print, log per-model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for line in lines:
        line = line.rstrip("\n")
        rec = parse_bench_line(line)
        if rec is None:
            if line:
                print(f"  {line}")
            continue
        print(_fmt(rec))
        with (out_dir / f"{rec['model']}.txt").open("a") as fh:
            fh.write(line + "\n")


def _live_lines(uri):  # pragma: no cover - needs radio hardware
    """Yield console lines from the AI-deck over CRTP. Best-effort across cflib
    versions; on any import/API error, tell the user to use --stdin / cfclient."""
    try:
        import cflib.crtp
        from cflib.crazyflie import Crazyflie
    except Exception as exc:
        sys.exit(f"cflib unavailable ({exc}); use --stdin or cfclient's Console tab.")

    import queue
    cflib.crtp.init_drivers()
    q: queue.Queue[str] = queue.Queue()
    cf = Crazyflie(rw_cache="./cache")
    # The AI-deck CPX console arrives on the Crazyflie console receiver as text.
    cf.console.receivedChar.add_callback(q.put)
    print(f"connecting to {uri} (Ctrl-C to stop) ...")
    cf.open_link(uri)
    buf = ""
    try:
        while True:
            buf += q.get()
            while "\n" in buf:
                head, buf = buf.split("\n", 1)
                yield head
    finally:
        cf.close_link()


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser(description="GAP8 net_bench CPX console reader")
    ap.add_argument("--uri", default=None, help="Crazyradio URI (live cflib read)")
    ap.add_argument("--stdin", action="store_true", help="read console lines from stdin")
    ap.add_argument("--out", default="data/mcu/board", help="dir for per-model logs")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    if args.uri and not args.stdin:
        stream(_live_lines(args.uri), out)
    else:
        stream(sys.stdin, out)
