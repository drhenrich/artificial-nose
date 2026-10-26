#!/usr/bin/env python3
"""Prueft, was die Firmware auf dem Wio Terminal ausgibt (Firmware dieses Repos, 115200 Baud).

    pip install pyserial
    python3 check_wio.py                       # Port automatisch suchen
    python3 check_wio.py --port /dev/cu.usbmodem1101 --seconds 15

Erkennt: Trainingsmodus (CSV no2,co,c2h5oh,voc - das brauchen record_sensors.py und
nose_bridge.py), Inferenzmodus (Edge-Impulse-Ausgaben), Konfigurationsmodus, keine Ausgabe.
Prueft ausserdem Messrate und ob die vier Gaskanaele sich plausibel bewegen.
"""
import argparse
import time

import serial
from serial.tools import list_ports

CHANNELS = ("NO2", "CO", "C2H5OH", "VOC")


def find_port():
    ports = list(list_ports.comports())
    for p in ports:          # Wio Terminal: Seeed / SAMD51 (USB-VID 0x2886)
        if p.vid == 0x2886 or "wio" in (p.description or "").lower():
            return p.device
    for p in ports:
        if "usbmodem" in p.device or "ttyACM" in p.device:
            return p.device
    raise SystemExit("Kein Wio Terminal gefunden. Ports: " + ", ".join(p.device for p in ports))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()
    port = args.port or find_port()
    print(f"Port: {port}, lese {args.seconds:.0f} s ...")

    csv, other = [], []
    with serial.Serial(port, 115200, timeout=1) as ser:
        ser.reset_input_buffer()
        t0 = time.time()
        while time.time() - t0 < args.seconds:
            line = ser.readline().decode(errors="replace").strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 4 and all(p.strip().lstrip("-").isdigit() for p in parts):
                csv.append([int(p) for p in parts])
            else:
                other.append(line)
    dt = time.time() - t0

    print(f"\n{len(csv)} CSV-Zeilen, {len(other)} andere Zeilen")
    if other:
        print("Andere Ausgaben (Auszug):")
        for line in list(dict.fromkeys(other))[:8]:
            print("   ", line[:100])

    if csv:
        print(f"\nTrainingsmodus: OK, {len(csv) / dt:.1f} Messungen/s")
        for c, name in enumerate(CHANNELS):
            vals = [r[c] for r in csv]
            lo, hi, last = min(vals), max(vals), vals[-1]
            note = ""
            if hi == lo:
                note = "  <- konstant (Sensor angeschlossen? an I2C/Grove-Port links?)"
            if lo >= 999:
                note = "  <- am Anschlag 999 (Sensor fehlt oder Kanal gesaettigt)"
            print(f"  {name:7s} min {lo:4d}  max {hi:4d}  letzter {last:4d}{note}")
        print("\nPasst fuer record_sensors.py und nose_bridge.py.")
    elif any("need more samples" in l.lower() or "predictions" in l.lower() or "anomaly" in l.lower()
             for l in other):
        print("\nVermutlich Inferenzmodus (Edge Impulse). Fuer Aufnahmen auf den Trainingsmodus "
              "umschalten: 5-Wege-Schalter (rechts vorn) einmal hineindruecken.")
    elif any("configuration mode" in l.lower() for l in other):
        print("\nKonfigurationsmodus (WLAN/Azure). Entsteht, wenn beim Start die Tasten A, B und C gedrueckt "
              "sind. Wio neu starten, ohne die Tasten zu halten.")
    else:
        print("\nKeine verwertbare Ausgabe. Moeglich: andere Firmware, falscher Port oder Wio im "
              "Bootloader. Firmware neu aufspielen: cd firmware && pio run -t upload")


if __name__ == "__main__":
    main()
