#!/usr/bin/env python3
"""Speicherabbilder fuer Variante E (ganzes Modell auf einem Tang Nano 20K).

    python3 export_variant_e.py model.safetensors connectome_mask.npz --out build/variant_e [--runs 5]

Erzeugt (Formate siehe VARIANTE-E.md):
    ptr.hex      801 x 14 Bit   Startadresse der Synapsenliste je praesyn. Neuron (BSRAM)
    syn.hex      N   x 18 Bit   {post[17:8], w_rec[7:0]}, nach praesyn. Neuron sortiert (BSRAM)
    w_in.hex     H*26 x 18 Bit  {w_in[i,2c+1] in [16:9], w_in[i,2c] in [7:0]} (BSRAM)
    w_out.bin    K*H Byte       int8, klassenweise (SDRAM, 32-Bit-Woerter little endian)
    b_out.hex    K   x 32 Bit   Bias in Readout-Integer-Skala (SDRAM oder BSRAM)
    params.vh    Verilog-Parameter (Groessen, Shifts, Leak-Konstanten)
Am Ende werden alle Dateien zurueckgelesen und mit den quantisierten Gewichten verglichen.
"""
import argparse
from pathlib import Path

import numpy as np

import snn_ref

BLOCK_BITS = 18 * 1024   # GW2AR-18 BSRAM-Block (1K x 18 / 2K x 9 / 512 x 36)
N_BLOCKS = 46


def blocks(words, width):
    """BSRAM-Bloecke fuer words Eintraege der Breite width (Gowin-Konfigurationen)."""
    for w, depth in ((9, 2048), (18, 1024), (36, 512)):
        if width <= w:
            return -(-words // depth)
    raise ValueError(width)


def tc(v, bits):
    return int(v) & ((1 << bits) - 1)


def write_hex(path, values, bits):
    digits = -(-bits // 4)
    path.write_text("".join(f"{tc(v, bits):0{digits}x}\n" for v in values))


def read_hex(path):
    return np.array([int(line, 16) for line in path.read_text().split()], dtype=np.int64)


def sx(v, bits):
    v = v & ((1 << bits) - 1)
    return np.where(v >= 1 << (bits - 1), v - (1 << bits), v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("mask")
    ap.add_argument("--out", default="build/variant_e")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--frac", type=int, default=snn_ref.FRAC, help="wie bei snn_ref.py")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    q = snn_ref.quantize(snn_ref.load_params(args.weights, args.mask), args.runs, frac=args.frac)
    h, n_in = q.w_in.shape
    k = q.w_out.shape[0]
    assert h <= 1024 and n_in % 2 == 0

    # Rekurrent: CSR nach praesynaptischem Neuron j (Spalte von w_rec[post, pre])
    ptr, syn = [0], []
    for j in range(h):
        for i in np.nonzero(q.w_rec[:, j])[0]:
            syn.append((int(i) << 8) | tc(q.w_rec[i, j], 8))
        ptr.append(len(syn))
    n_syn = len(syn)
    nz_dropped = int((snn_ref.load_params(args.weights, args.mask)["w_rec"] != 0).sum()) - n_syn
    write_hex(out / "ptr.hex", ptr, 14)
    write_hex(out / "syn.hex", syn, 18)

    # Eingang: zwei Gewichte pro 18-Bit-Wort, neuronenweise (26 Woerter pro Neuron)
    w_in_words = [(tc(q.w_in[i, 2 * c + 1], 8) << 9) | tc(q.w_in[i, 2 * c], 8)
                  for i in range(h) for c in range(n_in // 2)]
    write_hex(out / "w_in.hex", w_in_words, 18)

    # Readout: klassenweise ins SDRAM, Bias als 32 Bit
    (out / "w_out.bin").write_bytes(q.w_out.astype(np.int8).tobytes(order="C"))
    write_hex(out / "b_out.hex", q.b_out, 32)

    (out / "params.vh").write_text(f"""// Erzeugt von export_variant_e.py - nicht von Hand aendern
localparam N_HID   = {h};
localparam N_IN    = {n_in};
localparam N_CLS   = {k};
localparam N_SYN   = {n_syn};
localparam T_STEPS = {snn_ref.T_STEPS};
localparam N_RUNS  = {q.runs};
localparam FRAC    = {q.frac};
localparam R_FRAC  = {snn_ref.R_FRAC};
localparam SH_IN   = {q.s_in + snn_ref.R_FRAC - q.frac};   // acc_in  -> Q5.10
localparam SH_REC  = {q.s_rec - q.frac};   // acc_rec -> Q5.10
localparam K_MEM   = {snn_ref.K_MEM};
localparam K_SYN   = {snn_ref.K_SYN};
localparam V_TH    = {1 << q.frac};
""")

    # Rueckleseprobe
    p_back, s_back = read_hex(out / "ptr.hex"), read_hex(out / "syn.hex")
    w_rec_back = np.zeros_like(q.w_rec)
    for j in range(h):
        for word in s_back[p_back[j]:p_back[j + 1]]:
            w_rec_back[word >> 8, j] = sx(word, 8)
    wi = read_hex(out / "w_in.hex").reshape(h, n_in // 2)
    w_in_back = np.empty_like(q.w_in)
    w_in_back[:, 0::2], w_in_back[:, 1::2] = sx(wi, 8), sx(wi >> 9, 8)
    w_out_back = np.frombuffer((out / "w_out.bin").read_bytes(), dtype=np.int8).reshape(k, h)
    assert (w_rec_back == q.w_rec).all() and (w_in_back == q.w_in).all()
    assert (w_out_back == q.w_out).all() and (sx(read_hex(out / "b_out.hex"), 32) == q.b_out).all()

    budget = [
        ("syn (rekurrente Synapsen)", blocks(n_syn, 18)),
        ("ptr (Zeigertabelle)", blocks(h + 1, 14)),
        ("w_in", blocks(len(w_in_words), 18)),
        ("acc_rec (16 Bit)", blocks(h, 16)),
        ("v (16 Bit)", blocks(h, 16)),
        ("i_syn (16 Bit)", blocks(h, 16)),
        ("counts (4 x 7 Bit pro Wort)", blocks(-(-h // 4), 28)),
    ]
    used = sum(b for _, b in budget)
    print(f"Synapsen {n_syn}" + (f" ({nz_dropped} durch INT8-Rundung zu 0 geworden)" if nz_dropped else ""))
    print("BSRAM-Plan (Bloecke a 18 Kbit):")
    for name, b in budget:
        print(f"  {name:30s} {b:3d}")
    print(f"  {'Summe':30s} {used:3d} von {N_BLOCKS}" + ("  OK" if used <= N_BLOCKS else "  ZU VIEL"))
    print(f"SDRAM: w_out {k * h / 1024:.0f} KiB + Bias {k * 4 / 1024:.1f} KiB von 8 MiB")
    print(f"Rueckleseprobe OK, Dateien in {out}/")


if __name__ == "__main__":
    main()
