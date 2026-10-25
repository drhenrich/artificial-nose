#!/usr/bin/env python3
"""Analysiert das FlyWire-Olfactory-SNN fuer die Partitionierung auf N Tang Nano 20K.

Nur numpy wird benoetigt (safetensors und scipy-npz werden direkt geparst).

    python3 inspect_model.py model.safetensors connectome_mask.npz --boards 8 --wbits 8

Gibt Tensor-Formen, Sparsity der Konnektivitaet, Fan-out-Statistik und eine
BSRAM-Schaetzung pro Board aus (GW2AR-18: 46 x 18 Kbit = 828 Kbit BSRAM).
"""
import argparse
import json
import struct

import numpy as np

BSRAM_KBIT = 828
DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": None, "F64": np.float64,
          "I64": np.int64, "I32": np.int32, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        blob = f.read()
    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        raw = blob[a:b]
        if meta["dtype"] == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            arr = u16.view(np.float32)
        else:
            arr = np.frombuffer(raw, dtype=DTYPES[meta["dtype"]])
        tensors[name] = arr.reshape(meta["shape"])
    return tensors


def load_mask(path):
    z = np.load(path, allow_pickle=False)
    keys = set(z.files)
    if {"data", "indices", "indptr", "shape"} <= keys:  # scipy.sparse.save_npz (CSR/CSC)
        shape = tuple(z["shape"])
        fmt = z["format"].item() if "format" in keys else b"csr"
        fmt = fmt.decode() if isinstance(fmt, bytes) else str(fmt)
        dense = np.zeros(shape, dtype=np.float32)
        indptr, indices, data = z["indptr"], z["indices"], z["data"]
        for i in range(len(indptr) - 1):
            sl = slice(indptr[i], indptr[i + 1])
            if fmt == "csr":
                dense[i, indices[sl]] = data[sl]
            else:
                dense[indices[sl], i] = data[sl]
        return dense
    if {"row", "col", "shape"} <= keys:  # COO
        dense = np.zeros(tuple(z["shape"]), dtype=np.float32)
        dense[z["row"], z["col"]] = z["data"] if "data" in keys else 1
        return dense
    return z[z.files[0]]


def quant_error(w, bits):
    scale = np.abs(w).max() / (2 ** (bits - 1) - 1) or 1.0
    q = np.round(w / scale)
    return scale, float(np.abs(q * scale - w).mean() / (np.abs(w).mean() or 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("mask")
    ap.add_argument("--boards", type=int, default=8, help="Boards fuer das Hidden Layer")
    ap.add_argument("--wbits", type=int, default=8, help="Gewichtsbreite in Bit")
    args = ap.parse_args()

    tensors = load_safetensors(args.weights)
    print("== Tensoren ==")
    for name, t in tensors.items():
        line = f"{name:40s} {str(t.shape):16s} {t.dtype}"
        if t.dtype.kind == "f" and t.size > 1:
            _, err = quant_error(t.astype(np.float32), args.wbits)
            line += f"  rel. INT{args.wbits}-Fehler {err:.3%}"
        print(line)

    raw = load_mask(args.mask)
    mask = raw != 0
    n = mask.shape[0]
    nnz = int(mask.sum())
    print(f"\n== Konnektom-Maske == {mask.shape}, nnz={nnz}, Dichte={nnz / mask.size:.2%}")
    # Modell rechnet spk @ W.T, also W[post, pre]: Spalten = praesynaptisch
    fan_out, fan_in = mask.sum(axis=0), mask.sum(axis=1)
    print(f"Fan-out (Spalten) min/mean/max: {fan_out.min()}/{fan_out.mean():.1f}/{fan_out.max()}")
    print(f"Fan-in  (Zeilen)  min/mean/max: {fan_in.min()}/{fan_in.mean():.1f}/{fan_in.max()}")

    exc, inh = int((raw > 0).sum()), int((raw < 0).sum())
    if inh:
        print(f"Vorzeichen: {exc} erregend, {inh} hemmend ({inh / nnz:.1%})")
        mixed = sum(1 for j in range(n) if (raw[:, j] > 0).any() and (raw[:, j] < 0).any())
        print(f"Dale-Prinzip: {n - mixed}/{n} praesyn. Neuronen mit einheitlichem Vorzeichen"
              + (" -> Vorzeichen pro Neuron speichern" if mixed == 0 else " -> Vorzeichen pro Synapse speichern"))
    else:
        print("Keine negativen Eintraege: Vorzeichen liegen nicht in der Maske, Datei-Schluessel pruefen")

    b = args.boards
    per = -(-n // b)
    idx_bits = int(np.ceil(np.log2(per)))
    print(f"\n== Speicher pro Board ({b} Boards, {per} Neuronen/Board, {args.wbits}-bit) ==")
    shapes = {name: t.shape for name, t in tensors.items()}
    n_in = next((s[1] for s in shapes.values() if len(s) == 2 and s[0] == n and s[1] != n), 0)
    n_out = next((s[0] for s in shapes.values() if len(s) == 2 and s[1] == n and s[0] != n), 0)
    for k in range(b):
        rows = slice(k * per, min((k + 1) * per, n))
        syn = int(mask[rows].sum())
        rec = syn * (args.wbits + idx_bits) + n * 16       # CSR nach praesyn. Neuron + Zeigertabelle
        w_in = (rows.stop - rows.start) * n_in * args.wbits
        w_out = (rows.stop - rows.start) * n_out * args.wbits  # verteilte Teilsummen der Readout-Schicht
        state = (rows.stop - rows.start) * 32
        tot = (rec + w_in + w_out + state) / 1024
        flag = "OK" if tot < 0.85 * BSRAM_KBIT else "-> Readout ins SDRAM auslagern"
        print(f"Board {k + 1}: Synapsen {syn:6d} | rec {rec / 1024:6.1f} | in {w_in / 1024:6.1f} | "
              f"out {w_out / 1024:6.1f} | Summe {tot:6.1f} Kbit  {flag}")

    cross = sum(int(mask[k * per:(k + 1) * per].sum()) -
                int(mask[k * per:(k + 1) * per, k * per:(k + 1) * per].sum()) for k in range(b))
    print(f"\nSynapsen ueber Boardgrenzen: {cross} ({cross / max(nnz, 1):.1%}) "
          f"- irrelevant fuer den Spike-Vektor-Ring, relevant nur fuer AER-Varianten.")


if __name__ == "__main__":
    main()
