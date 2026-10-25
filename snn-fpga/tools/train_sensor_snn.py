#!/usr/bin/env python3
"""Neutraining des FlyWire-SNN auf eigene Gassensor-Aufnahmen (record_sensors.py).

Das Konnektom (recurrent, Maske, Vorzeichen) bleibt eingefroren. Trainiert werden die
Eingangsschicht (Start: Originalgewichte) und eine neue Ausgangsschicht fuer die eigenen Klassen.
Ergebnis ist ein Modellordner im selben Format wie das Hugging-Face-Modell, den
export_variant_e.py, snn_ref.py und board_test.py direkt verwenden:

    pip install torch norse scipy safetensors numpy
    python3 train_sensor_snn.py flywire-snn daten/ --out sensor-model
    python3 train_sensor_snn.py flywire-snn --synthetic synth-daten --out sensor-model   # Test ohne Sensor

Ausgabe: sensor-model/{model.safetensors, connectome_mask.npz, config.json, modeling_snn.py,
sensor_meta.json (Klassen, Normierung, Gate-Schwelle)}.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import sensor_features as sf
import snn_ref

GATE_LABEL = "luft"


# ------------------------------------------------------------------ Daten

def synth_sessions(out, n_per=4, seed=0):
    """Kuenstliche Aufnahmen: Grundlinie, Probe ab 30 s fuer 60 s, Abklingen; Drift und Rauschen."""
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    gains = {"luft": [0, 0, 0, 0], "kaffee": [0.2, 0.5, 0.9, 1.2], "ethanol": [0.1, 0.4, 1.8, 0.8],
             "essig": [0.6, 0.2, 0.5, 1.5], "rauch": [1.2, 1.4, 0.3, 0.9]}
    for label, g in gains.items():
        for s in range(n_per):
            t = np.cumsum(rng.uniform(0.08, 0.14, 1500))            # ~ 8 Hz, unregelmaessig
            base = np.array([80, 120, 60, 90]) * rng.uniform(0.85, 1.15, 4)
            prof = np.where(t < 30, 0, np.where(t < 90, 1 - np.exp(-(t - 30) / 8),
                                                (1 - np.exp(-60 / 8)) * np.exp(-(t - 90) / 15)))
            gain = np.array(g) * rng.uniform(0.8, 1.2, 4)
            x = base * np.exp(np.outer(prof, gain)) * (1 + 0.02 * rng.standard_normal((len(t), 4)))
            x *= 1 + 0.03 * np.linspace(0, 1, len(t))[:, None] * rng.standard_normal(4)   # Drift
            np.savez_compressed(out / f"{label}_synth{s}.npz", t=t, x=np.clip(np.round(x), 0, 999),
                                label=label)
    return out


def session_windows(path, stride_s=1.0):
    """Merkmale, Intensitaet fuer alle Fenster einer Aufnahme (Grundlinie wie im Live-Betrieb)."""
    d = np.load(path)
    t, x, label = d["t"], d["x"].astype(np.float64), str(d["label"])
    grid = np.arange(t[0], t[-1], 1.0 / sf.RATE_HZ)
    xr = np.stack([np.interp(grid, t, x[:, c]) for c in range(x.shape[1])], axis=1)
    feats, inten, base = [], [], None
    step = int(stride_s * sf.RATE_HZ)
    for n in range(len(grid)):
        base = sf.update_baseline(base, xr[n])
        if n + 1 >= sf.N_POINTS and (n + 1 - sf.N_POINTS) % step == 0:
            win = xr[n + 1 - sf.N_POINTS:n + 1]
            feats.append(sf.window_features(win, base))
            inten.append(sf.intensity(win, base))
    return np.array(feats), np.array(inten), label


def build_dataset(data_dir, gate):
    files = [f for f in sorted(Path(data_dir).glob("*.npz"))
             if {"t", "x", "label"} <= set(np.load(f).files)]      # nur Aufnahmen von record_sensors.py
    if not files:
        raise SystemExit(f"keine Aufnahmen in {data_dir}")
    per = [session_windows(f) for f in files]
    luft = np.concatenate([i for _, i, lab in per if lab == GATE_LABEL]) if any(
        lab == GATE_LABEL for _, _, lab in per) else None
    if gate is None:
        gate = float(np.percentile(luft, 99) * 1.5) if luft is not None else 0.3
    labels = sorted({lab for _, _, lab in per})
    X, y, sess = [], [], []
    for si, (f, i, lab) in enumerate(per):
        keep = (i >= gate) if lab != GATE_LABEL else np.ones(len(i), bool)
        X.append(f[keep]); y += [labels.index(lab)] * int(keep.sum()); sess += [si] * int(keep.sum())
    return np.concatenate(X), np.array(y), np.array(sess), labels, gate, [str(f) for f in files]


def split(y, sess, rng):
    """Je Klasse eine Aufnahme zurueckhalten, wenn moeglich; sonst 20 % zufaellig."""
    val = np.zeros(len(y), bool)
    for c in np.unique(y):
        s = np.unique(sess[y == c])
        if len(s) >= 2:
            val |= sess == s[-1]
        else:
            idx = np.nonzero(y == c)[0]
            val[rng.choice(idx, max(1, len(idx) // 5), replace=False)] = True
    return ~val, val


# ------------------------------------------------------------------ Training

def train(model_dir, X, y, k, epochs, lr, seed):
    import scipy.sparse as sp
    import torch
    from safetensors.torch import load_file
    sys.path.insert(0, str(model_dir))
    from modeling_snn import MaskedRecurrentLIFSNN

    torch.manual_seed(seed)
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    adj = sp.load_npz(Path(model_dir) / "connectome_mask.npz")
    sd = load_file(str(Path(model_dir) / "model.safetensors"))
    model = MaskedRecurrentLIFSNN(cfg["input_dim"], cfg["hidden_dim"], k, adj,
                                  steps=cfg["snn_steps"], alpha=cfg["snn_alpha"])
    with torch.no_grad():
        model.recurrent.copy_(sd["recurrent"])
        model.input_proj.weight.copy_(sd["input_proj.weight"])
    model.recurrent.requires_grad_(False)                     # Konnektom eingefroren
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-5)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).long()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot, hit = 0.0, 0
        for i in range(0, len(Xt), 32):
            b = perm[i:i + 32]
            logits, _ = model(Xt[b])
            loss = torch.nn.functional.cross_entropy(logits, yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b); hit += int((logits.argmax(1) == yt[b]).sum())
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"Epoche {ep:3d}: Verlust {tot / len(Xt):.3f}, Trainingsgenauigkeit {hit / len(Xt):.1%}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="Ordner des Originalmodells (model.safetensors, connectome_mask.npz, "
                                      "config.json, modeling_snn.py)")
    ap.add_argument("data_dir", nargs="?", help="Aufnahmen von record_sensors.py")
    ap.add_argument("--synthetic", help="kuenstliche Aufnahmen in diesem Ordner erzeugen und verwenden")
    ap.add_argument("--out", default="sensor-model")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gate", type=float, help="Intensitaets-Schwelle (Standard: aus Klasse 'luft')")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_dir = synth_sessions(args.synthetic) if args.synthetic else args.data_dir
    if data_dir is None:
        raise SystemExit("data_dir oder --synthetic angeben")
    X, y, sess, labels, gate, files = build_dataset(data_dir, args.gate)
    rng = np.random.default_rng(args.seed)
    tr, va = split(y, sess, rng)
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = ((X - mean) / std).astype(np.float32)
    print(f"{len(files)} Aufnahmen, {len(y)} Fenster, Klassen {labels}, Gate {gate:.3f}, "
          f"Training {int(tr.sum())} / Validierung {int(va.sum())}")

    model = train(args.model_dir, Xn[tr], y[tr], len(labels), args.epochs, args.lr, args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file({n: t.detach().contiguous() for n, t in model.state_dict().items()}, str(out / "model.safetensors"))
    for f in ("connectome_mask.npz", "modeling_snn.py"):
        shutil.copy(Path(args.model_dir) / f, out / f)
    cfg = json.loads((Path(args.model_dir) / "config.json").read_text())
    cfg.update(num_classes=len(labels), test_acc=None, best_val_acc=None, fold=None)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    meta = dict(labels=labels, mean=mean.tolist(), std=std.tolist(), gate=gate, gate_label=GATE_LABEL,
                window_s=sf.WINDOW_S, rate_hz=sf.RATE_HZ, features=sf.FEATURE_NAMES, recordings=files)

    # Bewertung auf der Validierung: Float (wie Torch) und bitgenauer Fixpunkt (wie FPGA)
    p = snn_ref.load_params(out / "model.safetensors", out / "connectome_mask.npz")
    q = snn_ref.quantize(p, 5)
    r_q = snn_ref.rate_quant(snn_ref.rate_encode(Xn[va]))
    e = snn_ref.bernoulli_lfsr(r_q, 5, 1)
    lf, _ = snn_ref.float_forward(p, Xn[va], e)
    li, _, _ = snn_ref.fixed_forward(q, r_q, e)
    acc_f, acc_q = float((lf.argmax(1) == y[va]).mean()), float((li.argmax(1) == y[va]).mean())
    print(f"Validierung: Float {acc_f:.1%}, Fixpunkt (FPGA) {acc_q:.1%}")
    meta.update(val_acc_float=acc_f, val_acc_fixed=acc_q)
    (out / "sensor_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Modell gespeichert in {out}/ - weiter mit export_variant_e.py und build.sh")


if __name__ == "__main__":
    main()
