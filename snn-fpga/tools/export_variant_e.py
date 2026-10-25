#!/usr/bin/env python3
"""Speicherabbilder und Testvektoren fuer Variante E (ganzes Modell auf einem Tang Nano 20K).

    python3 export_variant_e.py model.safetensors connectome_mask.npz --out build/variant_e [--tv 4]

Speicherabbilder (Formate siehe VARIANTE-E.md):
    ptr.hex      2*H+1 x 14 Bit  Listenstart je (post i, praesyn. Haelfte h), Index 2*i+h (BSRAM)
    syn.hex      N x 18 Bit      {pre_lokal[17:9], w_rec[8:0]}, pre = h*HALF + pre_lokal (BSRAM)
    w_in.hex     H*26 x 18 Bit   {w_in[i,2c+1] in [17:9], w_in[i,2c] in [8:0]} (BSRAM)
    b_out.hex    K x 32 Bit      Bias in Readout-Integer-Skala (BSRAM)
    w_out.hex    K*H/4 x 32 Bit  4 x int8 pro Wort, Byte n = w_out[c, 4j+n], klassenweise (SDRAM)
    params.vh    Verilog-Parameter
Testvektoren (--tv N, Zufallseingaben wie snn_ref.py):
    tv_r.hex (N*52 x 9 Bit), tv_seed.hex (N x 32), tv_spikes.hex (N*runs*T x H Bit, Bit i = Neuron i),
    tv_logits.hex (N*K x 32), tv_class.hex (N x 16), tv_counts.hex (N*H x 8), tv.vh (N_TV)
Am Ende werden alle Speicherabbilder zurueckgelesen und mit den quantisierten Gewichten verglichen.
"""
import argparse
from pathlib import Path

import numpy as np

import snn_ref

N_BLOCKS = 46   # GW2AR-18: 46 BSRAM-Bloecke a 18 Kbit (1K x 18 / 2K x 9 / 512 x 36)


def blocks(words, width):
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
    return np.array([int(line, 16) for line in path.read_text().split()], dtype=object)


def sx(v, bits):
    v = int(v) & ((1 << bits) - 1)
    return v - (1 << bits) if v >= 1 << (bits - 1) else v


def export_memories(q, out, bits_in, bits_rec):
    h, n_in = q.w_in.shape
    k = q.w_out.shape[0]
    half = h // 2
    assert h % 2 == 0 and half <= 512 and n_in % 2 == 0 and h % 4 == 0
    assert bits_in <= 9 and bits_rec <= 9

    # Rekurrent: pro Ziel-Neuron i zwei Listen (Quellen 0..HALF-1, HALF..H-1), nur Gewichte != 0
    ptr, syn = [0], []
    for i in range(h):
        for hh in range(2):
            for j in np.nonzero(q.w_rec[i, hh * half:(hh + 1) * half])[0]:
                syn.append((int(j) << 9) | tc(q.w_rec[i, hh * half + j], 9))
            ptr.append(len(syn))
    write_hex(out / "ptr.hex", ptr, 14)
    write_hex(out / "syn.hex", syn, 18)
    w_in_words = [(tc(q.w_in[i, 2 * c + 1], 9) << 9) | tc(q.w_in[i, 2 * c], 9)
                  for i in range(h) for c in range(n_in // 2)]
    write_hex(out / "w_in.hex", w_in_words, 18)
    write_hex(out / "b_out.hex", q.b_out, 32)
    w_out_words = [sum(tc(q.w_out[c, 4 * j + n], 8) << (8 * n) for n in range(4))
                   for c in range(k) for j in range(h // 4)]
    write_hex(out / "w_out.hex", w_out_words, 32)

    (out / "params.vh").write_text(f"""// Erzeugt von export_variant_e.py - nicht von Hand aendern
localparam N_HID   = {h};
localparam HALF    = {half};
localparam N_IN    = {n_in};
localparam N_CLS   = {k};
localparam N_SYN   = {len(syn)};
localparam T_STEPS = {snn_ref.T_STEPS};
localparam N_RUNS  = {q.runs};
localparam FRAC    = {q.frac};
localparam SH_IN   = {q.s_in + snn_ref.R_FRAC - q.frac};   // acc_in  -> Zustandsformat
localparam SH_REC  = {q.s_rec - q.frac};   // acc_rec -> Zustandsformat (negativ = Links-Shift)
localparam K_MEM   = {snn_ref.K_MEM};
localparam K_SYN   = {snn_ref.K_SYN};
localparam V_TH    = {1 << q.frac};
""")

    # Rueckleseprobe
    p_back, s_back = read_hex(out / "ptr.hex"), read_hex(out / "syn.hex")
    w_rec_back = np.zeros_like(q.w_rec)
    for i in range(h):
        for hh in range(2):
            for word in s_back[p_back[2 * i + hh]:p_back[2 * i + hh + 1]]:
                w_rec_back[i, hh * half + (word >> 9)] = sx(word, 9)
    wi = read_hex(out / "w_in.hex")
    w_in_back = np.array([[sx(wi[i * (n_in // 2) + kk // 2] >> (9 * (kk % 2)), 9) for kk in range(n_in)]
                          for i in range(h)])
    wo = read_hex(out / "w_out.hex")
    w_out_back = np.array([[sx(wo[c * (h // 4) + ii // 4] >> (8 * (ii % 4)), 8) for ii in range(h)]
                           for c in range(k)])
    b_back = np.array([sx(v, 32) for v in read_hex(out / "b_out.hex")])
    assert (w_rec_back == q.w_rec).all() and (w_in_back == q.w_in).all()
    assert (w_out_back == q.w_out).all() and (b_back == q.b_out).all()

    budget = [
        ("syn (rekurrente Synapsen)", blocks(len(syn), 18)),
        ("ptr (Listenanfaenge)", blocks(len(ptr), 14)),
        ("w_in", blocks(len(w_in_words), 18)),
        ("v (16 Bit)", blocks(h, 16)),
        ("i_syn (16 Bit)", blocks(h, 16)),
        ("counts (4 x 7 Bit pro Wort)", blocks(h // 4, 28)),
        ("b_out (32 Bit)", blocks(k, 32)),
    ]
    print(f"Synapsen {len(syn)} (von {int((q.w_rec != 0).sum())} != 0 nach Quantisierung)")
    print("BSRAM-Plan (Bloecke a 18 Kbit):")
    for name, b in budget:
        print(f"  {name:30s} {b:3d}")
    used = sum(b for _, b in budget)
    print(f"  {'Summe':30s} {used:3d} von {N_BLOCKS}" + ("  OK" if used <= N_BLOCKS else "  ZU VIEL"))
    print(f"SDRAM: w_out {k * h / 1024:.0f} KiB von 8 MiB")
    print("Rueckleseprobe OK")


def export_testvectors(q, out, n, seed):
    h, n_in = q.w_in.shape
    x = np.random.default_rng(seed).standard_normal((n, n_in)).astype(np.float32)
    r_q = snn_ref.rate_quant(snn_ref.rate_encode(x))
    e = snn_ref.bernoulli_lfsr(r_q, q.runs, seed)
    logits, counts, spikes = snn_ref.fixed_forward(q, r_q, e)
    write_hex(out / "tv_counts.hex", counts.ravel(), 8)
    write_hex(out / "tv_r.hex", r_q.ravel(), 9)
    write_hex(out / "tv_seed.hex", [(seed + b) & snn_ref.MASK32 for b in range(n)], 32)
    lines = []
    for b in range(n):
        for m in range(q.runs):
            for t in range(snn_ref.T_STEPS):
                v = sum(1 << int(i) for i in np.nonzero(spikes[m, t, b])[0])
                lines.append(f"{v:0{h // 4}x}\n")
    (out / "tv_spikes.hex").write_text("".join(lines))
    write_hex(out / "tv_logits.hex", logits.ravel(), 32)
    write_hex(out / "tv_class.hex", logits.argmax(1), 16)
    (out / "tv.vh").write_text(f"localparam N_TV = {n};\n")
    print(f"Testvektoren: {n} Proben, Klassen {list(logits.argmax(1))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("mask")
    ap.add_argument("--out", default="build/variant_e")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--frac", type=int, default=snn_ref.FRAC)
    ap.add_argument("--bits-in", type=int, default=snn_ref.BITS_IN)
    ap.add_argument("--bits-rec", type=int, default=snn_ref.BITS_REC)
    ap.add_argument("--tv", type=int, default=4, help="Anzahl Testvektor-Proben (0 = keine)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    q = snn_ref.quantize(snn_ref.load_params(args.weights, args.mask), args.runs, args.frac,
                         args.bits_in, args.bits_rec, 8)
    export_memories(q, out, args.bits_in, args.bits_rec)
    if args.tv:
        export_testvectors(q, out, args.tv, args.seed)
    print(f"Dateien in {out}/")


if __name__ == "__main__":
    main()
