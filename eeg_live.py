#!/usr/bin/env python3
"""Live EEG serial reader for ThinkGear/NeuroSky-style packets."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import pickle
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import numpy as np
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

SENSOR_COLUMNS = [
    "GSR", "PPG",
    "Delta", "Theta",
    "LowAlpha", "HighAlpha",
    "LowBeta", "HighBeta",
    "LowGamma", "MidGamma",
]
STATE_NAMES = [
    "NORMAL", "LOW_STRESS", "MODERATE_STRESS",
    "HIGH_ANXIETY", "PANIC_STATE", "DEPRESSION",
]
MODEL_BAND_KEYS = [
    "delta", "theta",
    "low_alpha", "high_alpha",
    "low_beta", "high_beta",
    "low_gamma", "mid_gamma",
]
SEQ_LEN = 10
ATTACH_EEG_MESSAGE = "Attach EEG sensors for live values"
DETACHED_POOR_SIGNAL = 200


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


def load_sensor_model(model_dir: Path) -> dict:
    from tensorflow import keras

    model_path = model_dir / "rnn_sensor_model.keras"
    if not model_path.exists():
        model_path = model_dir / "rnn_sensor_best.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"RNN sensor model not found in {model_dir}")

    scaler_path = model_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    label_path = model_dir / "label_encoder.pkl"
    model = keras.models.load_model(str(model_path), compile=False)
    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)

    labels = STATE_NAMES
    if label_path.exists():
        with label_path.open("rb") as handle:
            label_encoder = pickle.load(handle)
        labels = list(getattr(label_encoder, "classes_", STATE_NAMES))

    return {
        "model": model,
        "scaler": scaler,
        "labels": labels,
        "path": str(model_path),
    }


def reading_to_model_row(
    reading: dict,
    *,
    gsr_default: float,
    ppg_default: float,
) -> list[float] | None:
    norm = reading.get("normalized_eeg") or {}
    if any(norm.get(name) is None for name in MODEL_BAND_KEYS):
        return None
    return [
        float(gsr_default),
        float(ppg_default),
        *[float(norm[name]) for name in MODEL_BAND_KEYS],
    ]


def predict_from_window(bundle: dict, rows: deque[list[float]]) -> dict | None:
    if len(rows) < SEQ_LEN:
        return None

    arr = np.array(list(rows)[-SEQ_LEN:], dtype=np.float32)
    scaled = bundle["scaler"].transform(arr).reshape(1, SEQ_LEN, len(SENSOR_COLUMNS))
    probs = bundle["model"].predict(scaled, verbose=0)[0]
    idx = int(np.argmax(probs))
    labels = bundle["labels"]
    label = labels[idx] if idx < len(labels) else STATE_NAMES[idx]
    return {
        "state": label,
        "state_index": idx,
        "confidence": float(probs[idx]),
        "probabilities": {
            (labels[i] if i < len(labels) else STATE_NAMES[i]): float(p)
            for i, p in enumerate(probs)
        },
        "features": SENSOR_COLUMNS,
    }


def offline_status(port: str, baud: int, *, stale: bool = False, poor_signal: int | None = None) -> dict:
    return {
        "online": False,
        "live": False,
        "attached": False,
        "stale": stale,
        "message": ATTACH_EEG_MESSAGE,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "port": port,
        "baud": baud,
        "poor_signal": poor_signal,
        "prediction": None,
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

    if reading.get("online") is False:
        print(
            f"[{reading['timestamp']}] {reading.get('message', ATTACH_EEG_MESSAGE)} "
            f"signal={reading.get('poor_signal', '-')}",
            flush=True,
        )
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
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "models", help="Directory containing rnn_sensor_model.keras and scaler.pkl")
    parser.add_argument("--no-model", action="store_true", help="Read live EEG without running the RNN model")
    parser.add_argument("--gsr-default", type=float, default=2000.0, help="GSR value used when only the EEG headset is connected")
    parser.add_argument("--ppg-default", type=float, default=72.0, help="PPG/BPM value used when only the EEG headset is connected")
    parser.add_argument("--stale-timeout", type=float, default=3.0, help="Seconds without a valid attached EEG packet before live values expire")
    args = parser.parse_args()

    if not args.port:
        print("No USB serial port found. Connect the EEG USB adapter and try again.", file=sys.stderr)
        return 2

    model_bundle = None
    prediction_window: deque[list[float]] = deque(maxlen=SEQ_LEN)
    if not args.no_model:
        try:
            model_bundle = load_sensor_model(args.model_dir)
            print(f"Loaded sensor model: {model_bundle['path']}", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"Model load failed: {exc}", file=sys.stderr)
            return 1

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
        last_live_at = 0.0
        last_offline_notice_at = 0.0

        with serial.Serial(args.port, args.baud, timeout=1.0) as stream:
            while True:
                if args.duration and time.time() - started >= args.duration:
                    break

                payload = read_packet(stream)
                if payload is None:
                    stream_age = time.time() - started
                    live_age = time.monotonic() - last_live_at if last_live_at else None
                    if (last_live_at and live_age > args.stale_timeout) or (
                        not last_live_at and stream_age > args.stale_timeout
                    ):
                        prediction_window.clear()
                        now = time.monotonic()
                        if now - last_offline_notice_at >= args.stale_timeout:
                            print_reading(
                                offline_status(args.port, args.baud, stale=True),
                                args.json,
                            )
                            printed_count += 1
                            last_offline_notice_at = now
                    continue

                parsed = parse_payload(payload)
                if not parsed:
                    continue

                packet_count += 1
                poor_signal = parsed.get("poor_signal")
                if poor_signal is not None and poor_signal >= DETACHED_POOR_SIGNAL:
                    prediction_window.clear()
                    now = time.monotonic()
                    if now - last_offline_notice_at >= 1.0:
                        print_reading(
                            offline_status(args.port, args.baud, poor_signal=poor_signal),
                            args.json,
                        )
                        printed_count += 1
                        last_offline_notice_at = now
                    continue

                reading = {
                    "online": True,
                    "live": True,
                    "attached": True,
                    "stale": False,
                    "message": "Live EEG values are fresh",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    **parsed,
                }
                if "eeg_power" in reading:
                    reading["normalized_eeg"] = normalize_bands(reading["eeg_power"])
                    last_live_at = time.monotonic()
                    last_offline_notice_at = 0.0

                    if model_bundle:
                        row = reading_to_model_row(
                            reading,
                            gsr_default=args.gsr_default,
                            ppg_default=args.ppg_default,
                        )
                        if row:
                            prediction_window.append(row)
                            reading["model_window"] = {
                                "ready": len(prediction_window) >= SEQ_LEN,
                                "samples": len(prediction_window),
                                "required": SEQ_LEN,
                                "default_channels": ["GSR", "PPG"],
                            }
                            reading["prediction"] = predict_from_window(model_bundle, prediction_window)
                            if reading["prediction"] is None:
                                reading["message"] = "Collecting live EEG window before model prediction"

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
