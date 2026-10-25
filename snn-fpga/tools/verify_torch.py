#!/usr/bin/env python3
"""Prueft, dass snn_ref.float_forward das Original (modeling_snn.py + Norse) exakt nachbildet.

Benoetigt torch, norse, scipy, safetensors. torch.bernoulli wird durch dieselben
Eingangsspikes ersetzt, die auch das Referenzmodell bekommt.

    python3 verify_torch.py pfad/zu/modelordner [--n 32 --runs 2]

Der Modellordner enthaelt modeling_snn.py, model.safetensors, connectome_mask.npz, config.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from safetensors.torch import load_file

import snn_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    d = Path(args.model_dir)

    sys.path.insert(0, str(d))
    from modeling_snn import MaskedRecurrentLIFSNN

    cfg = json.loads((d / "config.json").read_text())
    model = MaskedRecurrentLIFSNN(cfg["input_dim"], cfg["hidden_dim"], cfg["num_classes"],
                                  sp.load_npz(d / "connectome_mask.npz"),
                                  steps=cfg["snn_steps"], alpha=cfg["snn_alpha"])
    model.load_state_dict(load_file(str(d / "model.safetensors")))
    model.eval()

    x = np.random.default_rng(args.seed).standard_normal((args.n, cfg["input_dim"])).astype(np.float32)
    e = snn_ref.bernoulli_lfsr(snn_ref.rate_quant(snn_ref.rate_encode(x)), args.runs, args.seed)

    draws = iter(torch.from_numpy(e.reshape(-1, *e.shape[2:]).astype(np.float32)))
    orig_bernoulli = torch.bernoulli
    torch.bernoulli = lambda rates, *a, **k: next(draws)
    try:
        with torch.no_grad():
            lt = sum(model(torch.from_numpy(x))[0].double() for _ in range(args.runs)) / args.runs
    finally:
        torch.bernoulli = orig_bernoulli

    p = snn_ref.load_params(d / "model.safetensors", d / "connectome_mask.npz")
    lf, _ = snn_ref.float_forward(p, x, e)
    diff = np.abs(lt.numpy() - lf)
    print(f"Torch vs. snn_ref.float_forward: max |dLogit| = {diff.max():.3e}, "
          f"Argmax gleich {(lt.numpy().argmax(1) == lf.argmax(1)).mean():.2%}")
    ok = diff.max() < 1e-3
    print("OK" if ok else "ABWEICHUNG")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
