#!/usr/bin/env python3
"""RUN THIS FIRST on the Windows machine, before anything else.

It answers the three questions that block everything downstream, and it never
touches the car, so it is safe to run with the game sitting at a menu.

    1. Is ViGEmBus installed and can we create a virtual gamepad?
    2. Is BeamNG emitting UDP telemetry, on which port, and how fast?
    3. What is actually in those packets -- do they match the LFS layouts?

Usage (in the repo folder, with BeamNG running and a vehicle spawned):

    py -m pip install vgamepad
    py windows_probe.py                    # listens on 4444 and 4445
    py windows_probe.py --ports 4444 5000  # if you configured other ports

BeamNG setup, to do before running:
  Options > Others (or Protocols) > enable OutGauge and Motion Sim / OutSim.
  Set the target IP to 127.0.0.1 and note the ports. Exact menu names vary by
  build -- if you cannot find them, run this anyway and tell me what it reports.
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from sim.telemetry import (  # noqa: E402
    MotionSimPacket,
    OutGaugePacket,
    OutSimPacket,
    TelemetryError,
    identify,
)


def check_gamepad() -> bool:
    print("=" * 68)
    print("1. VIRTUAL GAMEPAD")
    print("=" * 68)
    try:
        import vgamepad
    except ImportError:
        print("  vgamepad is NOT installed.        py -m pip install vgamepad")
        return False
    try:
        pad = vgamepad.VX360Gamepad()
    except Exception as error:  # ViGEmBus driver missing or not running
        print(f"  vgamepad imported but the pad would not open: {error}")
        print("  Install the ViGEmBus driver, then reboot:")
        print("    https://github.com/nefarius/ViGEmBus/releases")
        return False

    print("  OK - virtual Xbox 360 pad created.")
    print("  Sweeping steering left..right for 3 s - watch the car's wheels.")
    for i in range(150):
        pad.left_joystick_float(x_value_float=__import__("math").sin(i / 24), y_value_float=0.0)
        pad.update()
        time.sleep(0.02)
    pad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
    pad.update()
    print("  Sweep done. Did the front wheels move? If not, BeamNG is not")
    print("  binding the virtual pad - check Options > Controls.")
    return True


def describe(data: bytes) -> str:
    kind = identify(data)
    try:
        if kind == "outgauge":
            p = OutGaugePacket.parse(data)
            return (f"OutGauge   speed {p.speed_mps:6.2f} m/s  rpm {p.rpm:7.1f}  "
                    f"coolant {p.coolant_temp_c:5.1f}C  oil {p.oil_temp_c:5.1f}C  "
                    f"thr {p.throttle:4.2f}  brk {p.brake:4.2f}  gear {p.gear}")
        if kind == "motionsim":
            p = MotionSimPacket.parse(data)
            return (f"MotionSim  pos ({p.x_m:9.2f}, {p.y_m:9.2f}, {p.z_m:7.2f})  "
                    f"speed {p.speed_mps:6.2f} m/s  heading {p.heading_rad:6.3f}  "
                    f"yaw {p.yaw_rad:6.3f}")
        if kind == "outsim":
            p = OutSimPacket.parse(data)
            return (f"OutSim     pos ({p.x_m:9.2f}, {p.y_m:9.2f}, {p.z_m:7.2f})  "
                    f"heading {p.heading_rad:6.3f}  speed {p.speed_mps:6.2f} m/s")
    except TelemetryError as error:
        return f"{kind}-shaped but would not decode: {error}"

    floats = struct.unpack(f"<{len(data)//4}f", data[: len(data) // 4 * 4])
    return (f"UNKNOWN {len(data)} bytes  head={data[:4]!r}  floats: "
            + " ".join(f"{v:.4g}" for v in floats[:10]))


def listen(ports: list[int], seconds: float) -> None:
    print()
    print("=" * 68)
    print(f"2. UDP TELEMETRY  (listening on {ports} for {seconds:.0f}s)")
    print("=" * 68)
    print("  Drive the car around now - the numbers should change.\n")

    sockets = {}
    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
        except OSError as error:
            print(f"  port {port}: could not bind ({error}) - something else is using it")
            continue
        s.settimeout(0.2)
        sockets[port] = s

    counts = {port: 0 for port in sockets}
    sizes = {port: set() for port in sockets}
    kinds = {port: set() for port in sockets}
    last_print = {port: 0.0 for port in sockets}
    deadline = time.time() + seconds
    while time.time() < deadline:
        for port, sock in sockets.items():
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            counts[port] += 1
            sizes[port].add(len(data))
            kinds[port].add(identify(data) or f"unknown-{len(data)}")
            now = time.time()
            if now - last_print[port] > 0.5:
                last_print[port] = now
                print(f"  :{port}  {describe(data)}")

    print()
    print("  " + "-" * 64)
    for port in sockets:
        if counts[port]:
            print(f"  port {port}: {counts[port]} packets "
                  f"({counts[port]/seconds:.0f}/s), sizes {sorted(sizes[port])}, "
                  f"streams {sorted(kinds[port])}")
        else:
            print(f"  port {port}: nothing received (fine if both streams share a port)")
    found = set().union(*kinds.values()) if kinds else set()
    if "outgauge" in found and not {"motionsim", "outsim"} & found:
        print("\n  OutGauge is arriving but no pose stream is. Enable Motion Sim")
        print("  in BeamNG's protocol options - without pose the car cannot be")
        print("  steered along a route.")
    elif found and not any(k.startswith("unknown") for k in found):
        print("\n  Both streams present and decoding. You are ready to drive.")
    if not any(counts.values()):
        print("\n  No telemetry at all. Check that OutGauge / Motion Sim are enabled")
        print("  in BeamNG's options, that the target IP is 127.0.0.1, and that a")
        print("  vehicle is spawned and being driven.")
    for port in sockets:
        sockets[port].close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ports", type=int, nargs="+", default=[4444, 4445])
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--skip-gamepad", action="store_true")
    args = parser.parse_args()

    if not args.skip_gamepad:
        check_gamepad()
    listen(args.ports, args.seconds)

    print()
    print("=" * 68)
    print("Send me this whole output and I will wire up the real backend.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
