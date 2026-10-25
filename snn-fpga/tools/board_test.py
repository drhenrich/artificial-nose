#!/usr/bin/env python3
"""Testet die Bitstreams auf dem Tang Nano 20K ueber USB-UART gegen snn_ref.py.

    pip install pyserial numpy
    # Vollversion (rtl/top_tangnano20k_full.v, snn_full.fs): Readout auf dem FPGA
    python3 board_test.py model.safetensors connectome_mask.npz --port PORT --mode full
    # Bring-up (rtl/top_tangnano20k.v, snn_tangnano20k.fs): Spikezaehler, Readout auf dem PC
    python3 board_test.py model.safetensors connectome_mask.npz --port PORT --mode counts

full:   laedt w_out ins SDRAM (Befehl 'W', mit Pruefsumme), liest Stichproben zurueck ('R') und
        vergleicht danach Klasse und alle Logits jeder Probe bitgenau ('C').
counts: vergleicht die 800 Spikezaehler bitgenau, der PC rechnet den Readout.

Port: Linux meist /dev/ttyUSB1 (ttyUSB0 ist JTAG), macOS /dev/cu.usbserial-...1, Windows COMx.
Baudrate: full 1000000 (oder 115200 fuer snn_full_115200.fs), counts 115200.
Die Parameter (--runs, --frac) muessen zu dem Export passen, aus dem der Bitstream gebaut wurde.
"""
import argparse
import struct
import time

import numpy as np
import serial

import snn_ref

CHUNK = 1024          # Woerter pro 'W'-Frame


def read_exact(ser, n):
    data = ser.read(n)
    if len(data) != n:
        raise IOError(f"nur {len(data)} von {n} Bytes empfangen - Port, Baudrate und Bitstream pruefen")
    return data


def w_out_words(q):
    k, h = q.w_out.shape
    b = (q.w_out.astype(np.int64) & 0xFF).reshape(k, h // 4, 4)
    return (b[..., 0] | b[..., 1] << 8 | b[..., 2] << 16 | b[..., 3] << 24).ravel().astype(np.uint32)


def load_sdram(ser, words):
    t0 = time.time()
    for start in range(0, len(words), CHUNK):
        chunk = words[start:start + CHUNK]
        ser.write(bytes([0xA5]) + b"W" + struct.pack("<I", start)[:3] + struct.pack("<H", len(chunk))
                  + chunk.astype("<u4").tobytes())
        resp = read_exact(ser, 6)
        expect = int(chunk.astype(np.uint64).sum()) & 0xFFFFFFFF
        if resp[:2] != b"\x5aW" or struct.unpack("<I", resp[2:])[0] != expect:
            raise IOError(f"Pruefsumme beim Laden ab Wort {start} falsch")
        print(f"\rw_out laden: {start + len(chunk)}/{len(words)} Woerter", end="", flush=True)
    print(f"  ({time.time() - t0:.1f} s)")


def check_sdram(ser, words, n=8, seed=0):
    rng = np.random.default_rng(seed)
    errors = 0
    for start in rng.integers(0, len(words) - 64, n):
        ser.write(bytes([0xA5]) + b"R" + struct.pack("<I", int(start))[:3] + struct.pack("<H", 64))
        resp = read_exact(ser, 2 + 4 * 64)
        got = np.frombuffer(resp[2:], dtype="<u4")
        errors += int((got != words[start:start + 64]).sum())
    print(f"SDRAM-Ruecklesen: {n} x 64 Woerter, {errors} Fehler")
    return errors == 0


def set_latency(ser, lat):
    ser.write(bytes([0xA5]) + b"L" + bytes([lat]))
    resp = read_exact(ser, 3)
    return resp[:2] == b"\x5aL" and resp[2] == lat


def burst_ok(ser, words, start=230, n=200):
    ser.write(bytes([0xA5]) + b"B" + struct.pack("<I", start)[:3] + struct.pack("<H", n))
    resp = read_exact(ser, 2 + 4 * n)
    return int((np.frombuffer(resp[2:], dtype="<u4") != words[start:start + n]).sum())


def calibrate(ser, words, force=None):
    """SDRAM-Leselatenz bestimmen: Burst wie beim Readout fuer Latenz 2..7 pruefen."""
    if force:
        set_latency(ser, force)
        print(f"SDRAM-Leselatenz vorgegeben: {force}, Burst-Fehler {burst_ok(ser, words)}")
        return force
    good = []
    for lat in range(2, 8):
        set_latency(ser, lat)
        bad = burst_ok(ser, words)
        print(f"  Latenz {lat}: {bad} von 200 Burst-Woertern falsch")
        if bad == 0:
            good.append(lat)
    if not good:
        raise IOError("keine SDRAM-Leselatenz liefert korrekte Bursts")
    lat = good[len(good) // 2]
    set_latency(ser, lat)
    print(f"SDRAM-Leselatenz: {lat} (korrekt: {good})")
    return lat


def read_counts(ser, h):
    ser.write(bytes([0xA5]) + b"Z")
    resp = read_exact(ser, 2 + h + 1)
    return np.frombuffer(resp[2:2 + h], dtype=np.uint8).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("mask")
    ap.add_argument("--port", required=True)
    ap.add_argument("--mode", choices=("full", "counts"), default="full")
    ap.add_argument("--baud", type=int, help="Standard: full 1000000, counts 115200")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--frac", type=int, default=snn_ref.FRAC)
    ap.add_argument("--skip-load", action="store_true", help="w_out ist schon im SDRAM")
    ap.add_argument("--lat", type=int, help="SDRAM-Leselatenz vorgeben statt kalibrieren")
    ap.add_argument("--data", help="npz mit X (N x 52) und optional y (N,)")
    args = ap.parse_args()
    baud = args.baud or (1_000_000 if args.mode == "full" else 115_200)

    p = snn_ref.load_params(args.weights, args.mask)
    q = snn_ref.quantize(p, args.runs, args.frac)
    if args.data:
        d = np.load(args.data)
        x = d["X"][:args.n].astype(np.float32)
        y = d["y"][:args.n] if "y" in d.files else None
    else:
        x = np.random.default_rng(args.seed).standard_normal((args.n, 52)).astype(np.float32)
        y = None
    r_q = snn_ref.rate_quant(snn_ref.rate_encode(x))
    e = snn_ref.bernoulli_lfsr(r_q, args.runs, args.seed)
    logits_ref, counts_ref, _ = snn_ref.fixed_forward(q, r_q, e)
    logits_f, _ = snn_ref.float_forward(p, x, e)

    ok = hits_hw = hits_f = 0
    with serial.Serial(args.port, baud, timeout=10) as ser:
        ser.reset_input_buffer()
        if args.mode == "full" and not args.skip_load:
            words = w_out_words(q)
            load_sdram(ser, words)
            if not check_sdram(ser, words):
                return
        if args.mode == "full":
            calibrate(ser, w_out_words(q), args.lat)
        for b in range(len(x)):
            seed_b = (args.seed + b) & snn_ref.MASK32
            rates = b"".join(struct.pack("<H", int(v)) for v in r_q[b])
            t0 = time.time()
            if args.mode == "full":
                ser.write(bytes([0xA5]) + b"C" + struct.pack("<I", seed_b) + rates)
                resp = read_exact(ser, 2 + 2 + 4 + 4 * q.w_out.shape[0] + 4)
                dt = time.time() - t0
                if resp[:2] != b"\x5aC":
                    print(f"Probe {b}: falsche Antwort")
                    continue
                cls_hw = struct.unpack("<H", resp[2:4])[0]
                logits = np.frombuffer(resp[8:-4], dtype="<i4").astype(np.int64)
                if (int(logits.astype(np.uint32).astype(np.uint64).sum()) & 0xFFFFFFFF) != \
                        struct.unpack("<I", resp[-4:])[0]:
                    print(f"Probe {b}: Pruefsummenfehler")
                    continue
                diff = int((logits != logits_ref[b]).sum())
                good = diff == 0 and cls_hw == int(logits_ref[b].argmax())
                what = f"{diff} Logits verschieden"
                if not good:                   # eingrenzen: stimmen wenigstens die Spikezaehler?
                    cdiff = int((read_counts(ser, q.w_in.shape[0]) != counts_ref[b]).sum())
                    what += f", {cdiff} Spikezaehler verschieden" + (
                        " -> Kern ok, Readout/SDRAM falsch" if cdiff == 0 else " -> Kern rechnet anders")
            else:
                ser.write(bytes([0xA5]) + struct.pack("<I", seed_b) + rates)
                resp = read_exact(ser, 802)
                dt = time.time() - t0
                counts = np.frombuffer(resp[1:801], dtype=np.uint8).astype(np.int64)
                if resp[0] != 0x5A or (int(counts.sum()) & 0xFF) != resp[801]:
                    print(f"Probe {b}: falsche Antwort oder Pruefsummenfehler")
                    continue
                diff = int((counts != counts_ref[b]).sum())
                cls_hw = int((counts @ q.w_out.T + q.b_out).argmax())
                good = diff == 0
                what = f"{diff} Zaehler verschieden"
            cls_f = int(logits_f[b].argmax())
            ok += good
            line = (f"Probe {b}: {'OK ' if good else 'ABWEICHUNG'} ({what}), Klasse Board {cls_hw} / "
                    f"Float {cls_f}, {dt * 1000:.0f} ms")
            if y is not None:
                hits_hw += cls_hw == y[b]
                hits_f += cls_f == y[b]
                line += f", Label {int(y[b])}"
            print(line)
    print(f"\n{ok} von {len(x)} Proben bitgenau wie snn_ref.py")
    if y is not None:
        print(f"Genauigkeit: Board {hits_hw / len(x):.1%}, Float {hits_f / len(x):.1%}")


if __name__ == "__main__":
    main()
