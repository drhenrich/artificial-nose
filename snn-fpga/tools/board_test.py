#!/usr/bin/env python3
"""Testet den Bring-up-Bitstream (rtl/top_tangnano20k.v) auf dem Tang Nano 20K ueber USB-UART.

Schickt Eingaben an das Board, empfaengt die 800 Spikezaehler und vergleicht sie bitgenau mit
snn_ref.fixed_forward. Den Readout (692 Klassen) rechnet der PC aus den Zaehlern.

    pip install pyserial numpy
    python3 board_test.py model.safetensors connectome_mask.npz --port /dev/ttyUSB1 [--n 10]

Port: Linux meist /dev/ttyUSB1 (ttyUSB0 ist JTAG), Windows COMx, macOS /dev/cu.usbserial-*.
Die Parameter (--runs, --frac, Bitbreiten) muessen zu dem Export passen, aus dem der Bitstream
gebaut wurde (Standardwerte = Standardwerte von export_variant_e.py).
"""
import argparse
import struct
import time

import numpy as np
import serial

import snn_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("mask")
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--frac", type=int, default=snn_ref.FRAC)
    ap.add_argument("--data", help="npz mit X (N x 52) und optional y (N,)")
    args = ap.parse_args()

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
    _, counts_ref, _ = snn_ref.fixed_forward(q, r_q, e)
    logits_f, _ = snn_ref.float_forward(p, x, e)

    ok = 0
    hits_fix = hits_float = 0
    with serial.Serial(args.port, args.baud, timeout=10) as ser:
        ser.reset_input_buffer()
        for b in range(len(x)):
            seed_b = (args.seed + b) & snn_ref.MASK32
            frame = bytes([0xA5]) + struct.pack("<I", seed_b) + b"".join(
                struct.pack("<H", int(v)) for v in r_q[b])
            t0 = time.time()
            ser.write(frame)
            resp = ser.read(802)
            dt = time.time() - t0
            if len(resp) != 802 or resp[0] != 0x5A:
                print(f"Probe {b}: keine/unvollstaendige Antwort ({len(resp)} Bytes). "
                      "Port, Baudrate und geladenen Bitstream pruefen.")
                continue
            counts = np.frombuffer(resp[1:801], dtype=np.uint8).astype(np.int64)
            if (int(counts.sum()) & 0xFF) != resp[801]:
                print(f"Probe {b}: Pruefsummenfehler")
                continue
            diff = int((counts != counts_ref[b]).sum())
            logits = counts @ q.w_out.T + q.b_out
            cls_hw, cls_f = int(logits.argmax()), int(logits_f[b].argmax())
            ok += diff == 0
            line = (f"Probe {b}: {'OK ' if diff == 0 else 'ABWEICHUNG'} ({diff} Zaehler verschieden), "
                    f"Spikes {int(counts.sum())}, Klasse Board {cls_hw} / Float {cls_f}, {dt * 1000:.0f} ms")
            if y is not None:
                hits_fix += cls_hw == y[b]
                hits_float += cls_f == y[b]
                line += f", Label {int(y[b])}"
            print(line)
    print(f"\n{ok} von {len(x)} Proben bitgenau wie snn_ref.py")
    if y is not None:
        print(f"Genauigkeit: Board {hits_fix / len(x):.1%}, Float {hits_float / len(x):.1%}")


if __name__ == "__main__":
    main()
