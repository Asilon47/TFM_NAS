# mcu/BOARD.md — Real-silicon bring-up on the Crazyflie AI-deck (GAP8)

Goal of this first pass: a **silicon latency bench** — flash the a5fddcc graph
(`models/res192/cand_a5fddcc354bd.onnx`, the CP 10.3 winner, sim 58.39 M cyc / 3.0 FPS)
to the real GAP8 with a fixed input and read the **hardware** performance counters. That
converts our GVSOC ranking-only number into a measured one. Camera + DFL-decode/NMS + radio
output are a later, separate effort (see the end).

## The GreenWaves problem, and the practical route

GreenWaves is defunct (see [[gap8-toolchain-recovery]]), so the vendor flashing toolchain
(openocd fork + AI-deck board configs) is gone the same way the AutoTiler was. **Do NOT try
to reconstruct flashing from our `tfm-gap8:cp10.1` image** — it is built for GVSOC, and JTAG
over USB into Docker is its own headache. Instead use **Bitcraze's `aideck-gap8-examples`**
kit: Bitcraze packaged a pinned GAP SDK + openocd + AI-deck board configs + a Docker
toolchain *specifically for this deck*, precisely because the vendor disappeared. We bring
our AutoTiler graph into their known-good flashing flow.

## Step 0 — the programmer (the one extra piece of hardware)

The AI-deck flashes GAP8 over **JTAG**. Bitcraze's recommended adapter is the
**Olimex ARM-USB-TINY-H** into the AI-deck's JTAG header. (The Crazyradio/OTA path exists
via the ESP32/NINA but is flaky for dev — JTAG is the reliable loop.) If you don't have the
Olimex, that's the thing to get before anything else works.

## Step 1 — stand up Bitcraze's toolchain (the "can I flash at all" gate)

On the Linux host the deck plugs into (native, not our Docker):

```bash
git clone https://github.com/bitcraze/aideck-gap8-examples
cd aideck-gap8-examples
# Bitcraze ships a toolchain Docker image; their README pins the exact tag.
docker pull bitcraze/aideck-toolchain          # (or the tag their README pins)
```

Then wire USB permissions for the Olimex FTDI (Linux):

```bash
# /etc/udev/rules.d/99-openocd.rules — Olimex ARM-USB-TINY-H
SUBSYSTEM=="usb", ATTR{idVendor}=="15ba", ATTR{idProduct}=="002a", MODE="0666"
# then: sudo udevadm control --reload && replug the adapter
```

## Step 2 — flash their `helloworld` (the hardware analog of our GVSOC "Test success!")

Follow the aideck-gap8-examples README to build + flash `examples/other/hello_world_gap8`
over JTAG. Success = UART output from the board. **Until this prints, nothing else matters** —
it isolates "can this host flash this GAP8" from anything about our net. This is the gate.

## Step 3 — our a5fddcc board probe (I build this once Step 2 prints)

Once you can flash helloworld, tell me and I build the board bench:
- take a5fddcc's AutoTiler codegen (same graph GVSOC priced) into the Bitcraze app skeleton,
- a board `net_cyc.c` that wraps the cluster inference in **`pi_perf`** hardware counters
  (the silicon equivalent of the `AT_GraphPerf` we already read in sim — `mcu/probes/cyc/
  net_cyc.c` is the starting point),
- fixed input tensor (no camera yet), UART out the measured cycles/time,
- and the same for the yolo11n baseline graph, so the on-silicon RATIO is directly comparable.

Deliverable: measured GAP8 cycles for a5fddcc vs baseline → confirms (or corrects) the sim
ranking that underpins the CP 10.3 domination claim.

## Honest expectations for the silicon number

- **Real latency will be HIGHER than the 58.39 M sim cycles.** GVSOC models graph cycles but
  not real HyperRAM latency, DMA contention, or (later) the camera pipeline. The *ranking*
  (a5fddcc < baseline) is the claim, and that is what silicon confirms — not the absolute ms.
- **Also measure power** — the AI-deck's whole point is the sub-100 mW class, and it is a
  number GVSOC cannot give at all. GAP8 exposes it via `pi_perf` / the board's power rail.

## Later (separate, bigger): the full camera application

HM01B0 grayscale capture → crop/quantize to 192 → inference → **DFL decode + anchor concat +
NMS on the fabric controller** (our cycle graph EXCLUDES these — they are unimplemented C,
see procedure.md CP 10.1) → gate output over the Crazyradio. Plus: on-device accuracy is a
real domain shift (HM01B0 frames vs the synthetic A2RL val set the 0.6299 was measured on).
Not needed for the latency claim; needed for a real robotics demo.
