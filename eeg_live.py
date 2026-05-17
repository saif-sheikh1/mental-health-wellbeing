#!/usr/bin/env python3
"""Live EEG serial reader for ThinkGear/NeuroSky-style packets."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import serial


BAND_NAMES = [
    "delta",
    "theta",
    "low_alpha",
    "high_alpha",
    "low_beta",
    "high_beta",
    "low_gamma",
    "mid_gamma",
]


def find_default_port() -> str | None:
    patterns = [
        "/dev/cu.usbserial*",
        "/dev/cu.wchusbserial*",
        "/dev/cu.SLAB_USBtoUART*",
        "/dev/cu.usbmodem*",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def read_byte(stream: BinaryIO) -> int | None:
    data = stream.read(1)
    if not data:
        return None
    return data[0]


def read_packet(stream: BinaryIO) -> bytes | None:
    """Read one validated ThinkGear packet payload."""
    while True:
        byte = read_byte(stream)
        if byte is None:
            return None
        if byte != 0xAA:
            continue

        second = read_byte(stream)
        if second is None:
            return None
        if second != 0xAA:
            continue

        length = read_byte(stream)
        if length is None:
            return None
        if length == 0 or length > 169:
            continue

        payload = stream.read(length)
        checksum_raw = stream.read(1)
        if len(payload) != length or len(checksum_raw) != 1:
            return None

        checksum = checksum_raw[0]
        expected = (~sum(payload)) & 0xFF
        if checksum == expected:
            return payload


def uint24(raw: bytes) -> int:
    return (raw[0] << 16) | (raw[1] << 8) | raw[2]


def int16(raw: bytes) -> int:
    value = (raw[0] << 8) | raw[1]
    if value & 0x8000:
        value -= 0x10000
    return value


def parse_payload(payload: bytes) -> dict:
    parsed: dict[str, object] = {}
    i = 0
    while i < len(payload):
        code = payload[i]
        i += 1

        while code == 0x55 and i < len(payload):
            code = payload[i]
            i += 1

        if code < 0x80:
            length = 1
        else:
            if i >= len(payload):
                break
            length = payload[i]
            i += 1

        data = payload[i : i + length]
        i += length
        if len(data) != length:
            break

        if code == 0x02 and length == 1:
            parsed["poor_signal"] = data[0]
        elif code == 0x04 and length == 1:
            parsed["attention"] = data[0]
        elif code == 0x05 and length == 1:
            parsed["meditation"] = data[0]
        elif code == 0x16 and length == 1:
            parsed["blink_strength"] = data[0]
        elif code == 0x80 and length == 2:
            parsed["raw_wave"] = int16(data)
        elif code == 0x83 and length == 24:
            parsed["eeg_power"] = {
                name: uint24(data[idx * 3 : idx * 3 + 3])
                for idx, name in enumerate(BAND_NAMES)
            }

    return parsed


def normalize_bands(bands: dict[str, int]) -> dict[str, float]:
    """Map large raw band powers to 0..1 for quick display/API experiments."""
    return {
        name: round(min(1.0, math.log10(max(0, value) + 1) / 6.0), 4)
        for name, value in bands.items()
    }


def flatten_reading(reading: dict) -> dict:
    flat = {
        "timestamp": reading["timestamp"],
        "poor_signal": reading.get("poor_signal"),
        "attention": reading.get("attention"),
        "meditation": reading.get("meditation"),
        "blink_strength": reading.get("blink_strength"),
        "raw_wave": reading.get("raw_wave"),
    }

    bands = reading.get("eeg_power") or {}
    norm = reading.get("normalized_eeg") or {}
    for name in BAND_NAMES:
        flat[name] = bands.get(name)
        flat[f"{name}_normalized"] = norm.get(name)
    return flat


def print_reading(reading: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(reading, sort_keys=True), flush=True)
        return

    bands = reading.get("eeg_power") or {}
    parts = [
        f"[{reading['timestamp']}]",
        f"signal={reading.get('poor_signal', '-')}",
        f"attention={reading.get('attention', '-')}",
        f"meditation={reading.get('meditation', '-')}",
    ]
    if bands:
        parts.extend(f"{name}={bands[name]}" for name in BAND_NAMES)
    elif "raw_wave" in reading:
        parts.append(f"raw={reading['raw_wave']}")
    print("  ".join(parts), flush=True)


def main() -> int:
    default_port = find_default_port()
    parser = argparse.ArgumentParser(description="Read live EEG packets from a USB serial module.")
    parser.add_argument("--port", default=default_port, help="Serial port, for example /dev/cu.usbserial-1420")
    parser.add_argument("--baud", type=int, default=57600, help="Serial baud rate")
    parser.add_argument("--duration", type=float, default=0, help="Seconds to run; 0 means run until stopped")
    parser.add_argument("--json", action="store_true", help="Print JSON readings")
    parser.add_argument("--raw", action="store_true", help="Also print high-rate raw wave samples")
    parser.add_argument("--csv", type=Path, help="Append readings to a CSV file")
    args = parser.parse_args()

    if not args.port:
        print("No USB serial port found. Connect the EEG USB adapter and try again.", file=sys.stderr)
        return 2

    csv_file = None
    writer = None
    try:
        if args.csv:
            new_file = not args.csv.exists()
            csv_file = args.csv.open("a", newline="")
            writer = csv.DictWriter(csv_file, fieldnames=list(flatten_reading({"timestamp": ""}).keys()))
            if new_file:
                writer.writeheader()

        print(f"Opening EEG serial stream on {args.port} @ {args.baud} baud", file=sys.stderr, flush=True)
        print("Press Ctrl+C to stop.", file=sys.stderr, flush=True)
        started = time.time()
        packet_count = 0
        printed_count = 0

        with serial.Serial(args.port, args.baud, timeout=1.0) as stream:
            while True:
                if args.duration and time.time() - started >= args.duration:
                    break

                payload = read_packet(stream)
                if payload is None:
                    continue

                parsed = parse_payload(payload)
                if not parsed:
                    continue

                packet_count += 1
                reading = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    **parsed,
                }
                if "eeg_power" in reading:
                    reading["normalized_eeg"] = normalize_bands(reading["eeg_power"])

                is_summary = any(
                    key in reading
                    for key in ("eeg_power", "poor_signal", "attention", "meditation", "blink_strength")
                )
                if args.raw or is_summary:
                    printed_count += 1
                    print_reading(reading, args.json)

                if writer and (args.raw or is_summary):
                    writer.writerow(flatten_reading(reading))
                    csv_file.flush()

        print(
            f"Stopped after {packet_count} decoded packets, printed {printed_count}.",
            file=sys.stderr,
            flush=True,
        )
        return 0
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr, flush=True)
        return 0
    except serial.SerialException as exc:
        print(f"Serial error: {exc}", file=sys.stderr)
        return 1
    finally:
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
