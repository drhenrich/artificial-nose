#!/usr/bin/env python3
"""Referenzmodell fuer MaskedRecurrentLIFSNN: Float (wie modeling_snn.py) und bitgenauer Fixpunkt.

Das Fixpunkt-Modell ist die Spezifikation fuer die HDL (siehe KONZEPT.md, Abschnitt 4.3).
Jede Operation hier entspricht genau einer Hardware-Operation (Integer, Shift, Saettigung).

    python3 snn_ref.py model.safetensors connectome_mask.npz [--data daten.npz] [--n 200]
                       [--runs 5] [--dump testvektoren.npz]

--data: npz mit X (N x 52, standardisierte Rezeptorwerte) und optional y (N,) Klassenindizes.
Ohne --data werden Zufallseingaben (randn wie im Model-Card-Beispiel) verwendet.
"""
import argparse
from dataclasses import dataclass

import numpy as np

from inspect_model import load_mask, load_safetensors

T_STEPS = 20

# Norse LIFParameters-Standardwerte (modeling_snn.py setzt nur method und alpha), dt = 1 ms
DT, TAU_MEM_INV, TAU_SYN_INV = 1e-3, 100.0, 200.0
V_LEAK, V_TH, V_RESET = 0.0, 1.0, 0.0

# Fixpunkt-Format
FRAC = 10                      # v, i in Q5.10 (int16), V_TH = 1.0 = 1024
R_FRAC = 8                     # Eingangsrate r in Q0.8, 0..256 (256 = 1.0)
K_MEM = round(DT * TAU_MEM_INV * 2 ** 16)   # 6554  (0.1 in Q0.16)
K_SYN = round(DT * TAU_SYN_INV * 2 ** 16)   # 13107 (0.2 in Q0.16)
INT16_MIN, INT16_MAX = -2 ** 15, 2 ** 15 - 1
MASK32 = 0xFFFFFFFF


# ---------------------------------------------------------------- Parameter

def load_params(weights_path, mask_path):
    t = load_safetensors(weights_path)
    adj = load_mask(mask_path).astype(np.float32)
    mask, sign = (adj != 0).astype(np.float32), np.sign(adj).astype(np.float32)
    return {
        "w_in": t["input_proj.weight"].astype(np.float32),                 # (H, 52)
        "w_rec": (t["recurrent"] * mask * sign).astype(np.float32),        # (H, H), [post, pre]
        "w_out": t["readout.weight"].astype(np.float32),                   # (K, H)
        "b_out": t["readout.bias"].astype(np.float32),                     # (K,)
    }


def pow2_frac(w, bits=8):
    """Groesstes s, sodass max|w| * 2^s noch in INTbits passt."""
    m = float(np.abs(w).max())
    return 0 if m == 0 else int(np.floor(np.log2((2 ** (bits - 1) - 1) / m)))


def quant(w, s, bits=8):
    return np.clip(np.round(w * 2.0 ** s), -2 ** (bits - 1), 2 ** (bits - 1) - 1).astype(np.int64)


@dataclass
class QParams:
    w_in: np.ndarray    # int8-Werte, Skala 2^-s_in
    w_rec: np.ndarray   # int8-Werte, Skala 2^-s_rec (0 = keine Synapse)
    w_out: np.ndarray   # int8-Werte, Skala 2^-s_out
    b_out: np.ndarray   # int32, Skala des Readout-Akkumulators (haengt von runs ab)
    s_in: int
    s_rec: int
    s_out: int
    runs: int


def quantize(p, runs, bits=8):
    s_in, s_rec, s_out = (pow2_frac(p[k], bits) for k in ("w_in", "w_rec", "w_out"))
    # Readout: logit = (counts @ w_out_q.T) * 2^-s_out / (T*runs) + b
    #   -> b in derselben Integer-Skala: b_q = b * 2^s_out * T * runs
    b_q = np.round(p["b_out"].astype(np.float64) * 2.0 ** s_out * T_STEPS * runs).astype(np.int64)
    return QParams(quant(p["w_in"], s_in, bits), quant(p["w_rec"], s_rec, bits),
                   quant(p["w_out"], s_out, bits), b_q, s_in, s_rec, s_out, runs)


# ---------------------------------------------------------------- Eingang

def rate_encode(x):
    """Wie MaskedRecurrentLIFSNN._rate_encode: Min-Max pro Probe."""
    x = x.astype(np.float32)
    lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
    return np.clip((x - lo) / np.maximum(hi - lo, np.float32(1e-6)), 0.0, 1.0)


def rate_quant(r):
    return np.round(r.astype(np.float64) * 2 ** R_FRAC).astype(np.int64)  # 0..256


def xorshift32(x):
    x ^= (x << 13) & MASK32
    x ^= x >> 17
    x ^= (x << 5) & MASK32
    return x


def bernoulli_lfsr(r_q, runs, seed):
    """Eingangsspikes wie auf B0: ein xorshift32 pro Inferenz, Reihenfolge (Lauf, Schritt, Kanal).

    e = (zufall[31:24] < r_q), also P(e=1) = r_q / 256 exakt.
    Seed pro Lauf m: seed + m * 0x9E3779B9 (0 wird zu 1). Pro Probe b wird seed + b verwendet.
    Rueckgabe: bool-Array (runs, T, B, n_in).
    """
    b, n_in = r_q.shape
    e = np.zeros((runs, T_STEPS, b, n_in), dtype=bool)
    for m in range(runs):
        x = (np.arange(b, dtype=np.uint64) + seed + m * 0x9E3779B9) & MASK32
        x[x == 0] = 1
        for t in range(T_STEPS):
            for k in range(n_in):
                x = xorshift32(x)
                e[m, t, :, k] = (x >> 24) < r_q[:, k].astype(np.uint64)
    return e


# ---------------------------------------------------------------- Float-Modell

def float_forward(p, x, e):
    """Bildet modeling_snn.py nach; e ersetzt torch.bernoulli. Rueckgabe: Logits (B, K), Spikes."""
    r = rate_encode(x)
    runs, steps, b, _ = e.shape
    h = p["w_rec"].shape[0]
    logits = np.zeros((b, p["w_out"].shape[0]), dtype=np.float64)
    spikes = np.zeros((runs, steps, b, h), dtype=bool)
    for m in range(runs):
        v = np.full((b, h), V_LEAK, np.float32)
        i = np.zeros((b, h), np.float32)
        spk = np.zeros((b, h), np.float32)
        acc = np.zeros((b, h), np.float32)
        for t in range(steps):
            cur = (e[m, t] * r) @ p["w_in"].T + spk @ p["w_rec"].T
            # Norse lif_feed_forward_step
            v_dec = v + np.float32(DT * TAU_MEM_INV) * ((V_LEAK - v) + i)
            i_dec = i + np.float32(-DT * TAU_SYN_INV) * i
            z = (v_dec - V_TH > 0).astype(np.float32)
            v = (1 - z) * v_dec + z * V_RESET
            i = i_dec + cur
            spk = z
            acc += z
            spikes[m, t] = z > 0
        logits += (acc / np.float32(steps)) @ p["w_out"].T + p["b_out"]
    return logits / runs, spikes


# ---------------------------------------------------------------- Fixpunkt-Modell

def shift_round(x, s):
    """Arithmetischer Rechts-Shift mit Rundung (Halbe aufwaerts); s < 0 = Links-Shift."""
    if s > 0:
        return (x + (1 << (s - 1))) >> s
    return x << (-s)


def mul_q16(k, x):
    return (k * x + (1 << 15)) >> 16


def sat16(x, stats=None, name=None):
    if stats is not None:
        stats[name] = stats.get(name, 0) + int(((x < INT16_MIN) | (x > INT16_MAX)).sum())
    return np.clip(x, INT16_MIN, INT16_MAX)


def fixed_forward(q, r_q, e, stats=None):
    """Bitgenaues Hardware-Modell. Alle Groessen int64-Arrays mit Integer-Werten.

    Rueckgabe: logits_int (B, K), counts (B, H), spikes (runs, T, B, H).
    """
    runs, steps, b, _ = e.shape
    h = q.w_rec.shape[0]
    counts = np.zeros((b, h), np.int64)
    spikes = np.zeros((runs, steps, b, h), dtype=bool)
    sh_in = q.s_in + R_FRAC - FRAC
    sh_rec = q.s_rec - FRAC
    for m in range(runs):
        v = np.zeros((b, h), np.int64)
        i = np.zeros((b, h), np.int64)
        spk = np.zeros((b, h), np.int64)
        for t in range(steps):
            acc_in = (e[m, t] * r_q) @ q.w_in.T          # Summe ueber Eingangsspikes: w_in * r
            acc_rec = spk @ q.w_rec.T                    # Summe ueber praesyn. Spikes: w_rec
            cur = shift_round(acc_in, sh_in) + shift_round(acc_rec, sh_rec)
            v_dec = sat16(v + mul_q16(K_MEM, i - v), stats, "sat_v")     # V_LEAK = 0
            i_dec = i - mul_q16(K_SYN, i)
            z = v_dec > (1 << FRAC)
            v = np.where(z, 0, v_dec)                    # V_RESET = 0
            i = sat16(i_dec + cur, stats, "sat_i")
            spk = z.astype(np.int64)
            counts += spk
            spikes[m, t] = z
            if stats is not None:
                for name, val in (("max_acc_in", acc_in), ("max_acc_rec", acc_rec), ("max_cur", cur)):
                    stats[name] = max(stats.get(name, 0), int(np.abs(val).max()))
    logits_int = counts @ q.w_out.T + q.b_out
    if stats is not None:
        stats["max_logit_acc"] = int(np.abs(logits_int).max())
    return logits_int, counts, spikes


def logits_int_to_float(q, logits_int):
    return logits_int / (2.0 ** q.s_out * T_STEPS * q.runs)


# ---------------------------------------------------------------- Auswertung

def bits_needed(v):
    return int(np.ceil(np.log2(v + 1))) + 1 if v else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("mask")
    ap.add_argument("--data", help="npz mit X (N x 52) und optional y (N,)")
    ap.add_argument("--n", type=int, default=200, help="Anzahl Proben (Zufall oder erste n aus --data)")
    ap.add_argument("--runs", type=int, default=5, help="Monte-Carlo-Laeufe pro Probe")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dump", help="Testvektoren fuer die HDL-Simulation als npz schreiben")
    args = ap.parse_args()

    p = load_params(args.weights, args.mask)
    q = quantize(p, args.runs)
    print(f"Skalen: w_in 2^-{q.s_in}, w_rec 2^-{q.s_rec}, w_out 2^-{q.s_out} | "
          f"Shifts in Q{15 - FRAC}.{FRAC}: Eingang {q.s_in + R_FRAC - FRAC}, rekurrent {q.s_rec - FRAC}")

    y = None
    if args.data:
        d = np.load(args.data)
        x = d["X"][:args.n].astype(np.float32)
        y = d["y"][:args.n] if "y" in d.files else None
    else:
        x = np.random.default_rng(args.seed).standard_normal((args.n, p["w_in"].shape[1])).astype(np.float32)

    r_q = rate_quant(rate_encode(x))
    e = bernoulli_lfsr(r_q, args.runs, args.seed)
    lf, sf = float_forward(p, x, e)
    stats = {}
    li, counts, si = fixed_forward(q, r_q, e, stats)
    lq = logits_int_to_float(q, li)

    pf, pq = lf.argmax(1), li.argmax(1)
    top5 = np.argsort(-lq, axis=1)[:, :5]
    print(f"\nProben {len(x)}, Laeufe {args.runs}, gleiche Eingangsspikes fuer Float und Fixpunkt")
    print(f"Feuerrate  Float {sf.mean():.4f}  Fixpunkt {si.mean():.4f}  (Spikes/Neuron/Schritt)")
    print(f"Stille Neuronen (Fixpunkt): {(counts == 0).mean():.1%}")
    print(f"Spike-Uebereinstimmung: {(sf == si).mean():.4%}")
    print(f"Argmax gleich: {(pf == pq).mean():.2%}  | Float-Argmax in Fixpunkt-Top-5: "
          f"{(top5 == pf[:, None]).any(1).mean():.2%}")
    print(f"Logit-Abweichung: mittel {np.abs(lq - lf).mean():.4g}, Korrelation "
          f"{np.corrcoef(lq.ravel(), lf.ravel())[0, 1]:.5f}")
    if y is not None:
        print(f"Genauigkeit: Float {(pf == y).mean():.2%}  Fixpunkt {(pq == y).mean():.2%}")

    print("\nWertebereiche fuer die HDL (Bitbreite inkl. Vorzeichen):")
    for k in ("max_acc_in", "max_acc_rec", "max_cur", "max_logit_acc"):
        print(f"  {k:14s} {stats[k]:>12d}  -> {bits_needed(stats[k])} Bit")
    print(f"  Saettigungen: v {stats['sat_v']}, i {stats['sat_i']}")

    if args.dump:
        nd = min(8, len(x))
        np.savez_compressed(
            args.dump, r_q=r_q[:nd], e=e[:, :, :nd], spikes=si[:, :, :nd], counts=counts[:nd],
            logits_int=li[:nd], w_in=q.w_in, w_rec=q.w_rec, w_out=q.w_out, b_out=q.b_out,
            shifts=np.array([q.s_in, q.s_rec, q.s_out, FRAC, R_FRAC, K_MEM, K_SYN, q.runs]))
        print(f"\nTestvektoren ({nd} Proben) geschrieben: {args.dump}")


if __name__ == "__main__":
    main()
