"""OutGauge and OutSim packet decoding.

Both are LFS-derived UDP formats that BeamNG re-emits. OutGauge carries the
engine channels; it carries **no position**, so pose for path following has to
come from OutSim. Parsing is pure and fully testable without the game.
"""

import math
import struct

import pytest

from sim.telemetry import (
    OUTGAUGE_LAYOUT,
    OUTSIM_LAYOUT,
    OutGaugePacket,
    OutSimPacket,
    TelemetryError,
)


def outgauge_bytes(speed=20.0, rpm=2500.0, eng_temp=88.0, throttle=0.5,
                   brake=0.0, gear=4, time_ms=1234, with_id=False):
    packed = struct.pack(
        OUTGAUGE_LAYOUT, time_ms, b"etk8", 0, gear, 0,
        speed, rpm, 0.0, eng_temp, 0.7, 3.5, 95.0,
        0, 0, throttle, brake, 0.0, b"", b"",
    )
    return packed + struct.pack("<i", 7) if with_id else packed


def outsim_bytes(x=10.0, y=-4.0, heading=0.5, vx=20.0, vy=1.0, with_id=False):
    packed = struct.pack(
        OUTSIM_LAYOUT, 1234, 0.0, 0.0, 0.0, heading, 0.0, 0.0,
        0.0, 0.0, 0.0, vx, vy, 0.0,
        int(x * 65536), int(y * 65536), 0,
    )
    return packed + struct.pack("<i", 7) if with_id else packed


def test_outgauge_decodes_the_engine_channels():
    packet = OutGaugePacket.parse(outgauge_bytes(speed=25.0, rpm=3100.0, eng_temp=91.5))
    assert packet.speed_mps == pytest.approx(25.0)
    assert packet.rpm == pytest.approx(3100.0)
    assert packet.coolant_temp_c == pytest.approx(91.5)


def test_outgauge_decodes_the_pedals():
    packet = OutGaugePacket.parse(outgauge_bytes(throttle=0.8, brake=0.25))
    assert packet.throttle == pytest.approx(0.8)
    assert packet.brake == pytest.approx(0.25)


def test_outgauge_decodes_the_gear_with_neutral_as_zero():
    # The wire format is reverse=0, neutral=1, first=2. Report a human gear.
    assert OutGaugePacket.parse(outgauge_bytes(gear=1)).gear == 0
    assert OutGaugePacket.parse(outgauge_bytes(gear=2)).gear == 1
    assert OutGaugePacket.parse(outgauge_bytes(gear=0)).gear == -1


def test_outgauge_accepts_the_optional_trailing_id():
    assert OutGaugePacket.parse(outgauge_bytes(with_id=True)).speed_mps == pytest.approx(20.0)


def test_outgauge_reports_its_timestamp():
    assert OutGaugePacket.parse(outgauge_bytes(time_ms=98765)).time_ms == 98765


def test_a_packet_of_the_wrong_size_is_rejected_with_its_length():
    with pytest.raises(TelemetryError, match="17 bytes"):
        OutGaugePacket.parse(b"x" * 17)


def test_outsim_decodes_position_in_metres():
    packet = OutSimPacket.parse(outsim_bytes(x=123.5, y=-7.25))
    assert packet.x_m == pytest.approx(123.5, abs=1e-3)
    assert packet.y_m == pytest.approx(-7.25, abs=1e-3)


def test_outsim_decodes_heading():
    assert OutSimPacket.parse(outsim_bytes(heading=1.25)).heading_rad == pytest.approx(1.25)


def test_outsim_reports_ground_speed_from_the_velocity_vector():
    packet = OutSimPacket.parse(outsim_bytes(vx=3.0, vy=4.0))
    assert packet.speed_mps == pytest.approx(5.0)


def test_outsim_accepts_the_optional_trailing_id():
    assert OutSimPacket.parse(outsim_bytes(with_id=True)).x_m == pytest.approx(10.0, abs=1e-3)


def test_an_outsim_packet_of_the_wrong_size_is_rejected():
    with pytest.raises(TelemetryError, match="bytes"):
        OutSimPacket.parse(b"x" * 30)


def test_the_two_formats_have_distinguishable_lengths():
    # The probe on the Windows machine identifies a stream by packet length.
    assert struct.calcsize(OUTGAUGE_LAYOUT) != struct.calcsize(OUTSIM_LAYOUT)


# --- BeamNG MotionSim -----------------------------------------------------

from sim.telemetry import MOTIONSIM_LAYOUT, MOTIONSIM_MAGIC, MotionSimPacket  # noqa: E402


def motionsim_bytes(pos=(29.1, -368.0, 128.0), vel=(0.0, 0.0, 0.0), yaw=0.3,
                    magic=MOTIONSIM_MAGIC):
    return struct.pack(
        MOTIONSIM_LAYOUT, magic, *pos, *vel, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, yaw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )


def test_the_motionsim_packet_is_eighty_eight_bytes():
    assert struct.calcsize(MOTIONSIM_LAYOUT) == 88


def test_the_motionsim_magic_is_bng1():
    # Observed on the wire as float 2.9e-09, which is the bytes "BNG1".
    assert MOTIONSIM_MAGIC == b"BNG1"
    assert struct.unpack("<f", MOTIONSIM_MAGIC)[0] == pytest.approx(2.9e-09, rel=0.01)


def test_motionsim_decodes_position():
    packet = MotionSimPacket.parse(motionsim_bytes(pos=(595.0, -233.0, 147.0)))
    assert (packet.x_m, packet.y_m, packet.z_m) == pytest.approx((595.0, -233.0, 147.0))


def test_motionsim_decodes_the_velocity_vector():
    packet = MotionSimPacket.parse(motionsim_bytes(vel=(-4.89, -1.37, 0.076)))
    assert packet.speed_mps == pytest.approx(math.hypot(4.89, 1.37), rel=1e-3)


def test_heading_comes_from_the_velocity_vector_while_moving():
    # Velocity is guaranteed to share the frame the positions are in; the sign
    # and zero convention of yawPos is not documented, and getting it wrong
    # inverts the steering.
    packet = MotionSimPacket.parse(motionsim_bytes(vel=(0.0, 10.0, 0.0), yaw=99.0))
    assert packet.heading_rad == pytest.approx(math.pi / 2)


def test_heading_falls_back_to_yaw_when_almost_stationary():
    packet = MotionSimPacket.parse(motionsim_bytes(vel=(0.01, 0.0, 0.0), yaw=1.1))
    assert packet.heading_rad == pytest.approx(1.1)


def test_the_reported_yaw_is_kept_so_the_two_can_be_compared():
    packet = MotionSimPacket.parse(motionsim_bytes(vel=(10.0, 0.0, 0.0), yaw=1.1))
    assert packet.yaw_rad == pytest.approx(1.1)
    assert packet.heading_rad == pytest.approx(0.0)


def test_a_packet_without_the_magic_is_rejected():
    with pytest.raises(TelemetryError, match="BNG1"):
        MotionSimPacket.parse(motionsim_bytes(magic=b"XXXX"))


def test_a_motionsim_packet_of_the_wrong_size_is_rejected():
    with pytest.raises(TelemetryError, match="bytes"):
        MotionSimPacket.parse(b"BNG1" + b"\x00" * 40)


# --- demultiplexing -------------------------------------------------------

from sim.telemetry import identify  # noqa: E402


def test_a_motionsim_datagram_is_identified():
    assert identify(motionsim_bytes()) == "motionsim"


def test_an_outgauge_datagram_is_identified():
    assert identify(outgauge_bytes()) == "outgauge"


def test_an_outgauge_datagram_with_a_trailing_id_is_identified():
    # This is what BeamNG actually sends: 96 bytes, not 92.
    assert identify(outgauge_bytes(with_id=True)) == "outgauge"


def test_an_unrecognised_datagram_is_reported_as_unknown():
    assert identify(b"nonsense") is None


def test_motionsim_is_not_confused_with_outgauge():
    # Both arrive on the same port in practice, so this is the whole game.
    assert identify(motionsim_bytes()) != identify(outgauge_bytes(with_id=True))
