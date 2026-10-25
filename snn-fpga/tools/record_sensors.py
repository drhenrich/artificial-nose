#!/usr/bin/env python3
"""Nimmt Messreihen vom Wio Terminal (Firmware dieses Repos, Trainingsmodus) auf.

Die Firmware schreibt im Trainingsmodus pro Messung eine Zeile "no2,co,c2h5oh,voc" (115200 Baud).
Jede Aufnahme wird mit Zeitstempeln und Label als npz gespeichert:

    python3 record_sensors.py --port /dev/cu.usbmodem1101 --label kaffee --seconds 120 --out daten/
    python3 record_sensors.py --port ... --label luft --seconds 300 --out daten/

Ablauf je Geruch: 30 s saubere Luft, dann Probe vor den Sensor, 60-90 s halten, wegnehmen.
Mehrere Aufnahmen pro Geruch an verschiedenen Tagen machen (Drift, Temperatur, Feuchte).
Die Klasse "luft" (keine Probe) immer mit aufnehmen.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import serial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--label", required=True)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", default="daten")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t, x = [], []
    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        ser.reset_input_buffer()
        t0 = time.time()
        while time.time() - t0 < args.seconds:
            line = ser.readline().decode(errors="ignore").strip()
            parts = line.split(",")
            if len(parts) != 4:
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            t.append(time.time() - t0)
            x.append(vals)
            if len(t) % 20 == 0:
                print(f"\r{t[-1]:6.1f} s  {len(t)} Messungen  letzte {vals}", end="", flush=True)
    name = out / f"{args.label}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez_compressed(name, t=np.array(t), x=np.array(x), label=args.label)
    rate = len(t) / max(t[-1], 1e-9) if t else 0.0
    print(f"\n{len(t)} Messungen ({rate:.1f} Hz) gespeichert: {name}")


if __name__ == "__main__":
    main()
