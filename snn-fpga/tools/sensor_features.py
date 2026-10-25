"""Merkmale aus dem Grove-Multichannel-Gassensor (NO2, CO, C2H5OH, VOC) fuer das SNN.

Ein Fenster von WINDOW_S Sekunden wird auf RATE_HZ neu abgetastet. Pro Kanal werden die Werte
logarithmiert und um eine Grundlinie (saubere Luft) verschoben; daraus entstehen 13 Merkmale,
zusammen 4 x 13 = 52 = Eingangsbreite des Modells. Dieselben Funktionen nutzen Training
(train_sensor_snn.py) und Live-Betrieb (nose_bridge.py).
"""
import numpy as np

CHANNELS = ("no2", "co", "c2h5oh", "voc")
WINDOW_S = 10.0
RATE_HZ = 10.0
N_POINTS = int(WINDOW_S * RATE_HZ)
FEATURES_PER_CHANNEL = 13
N_FEATURES = len(CHANNELS) * FEATURES_PER_CHANNEL      # 52
FEATURE_NAMES = [f"{c}_{n}" for c in CHANNELS for n in (
    "last", "mean", "std", "min", "max", "slope", "first_half", "second_half",
    "max_absdiff", "mean_diff", "range", "delta", "peak_pos")]


def resample(t, x, t_end=None):
    """Linear auf N_POINTS Punkte im Fenster [t_end - WINDOW_S, t_end] interpolieren.

    t: (n,) Sekunden, x: (n, 4) Rohwerte. Rueckgabe (N_POINTS, 4).
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    t_end = t[-1] if t_end is None else t_end
    grid = np.linspace(t_end - WINDOW_S, t_end, N_POINTS)
    return np.stack([np.interp(grid, t, x[:, c]) for c in range(x.shape[1])], axis=1)


def to_log(x, baseline):
    """log1p-Skala relativ zur Grundlinie (MOX-Sensoren antworten etwa logarithmisch)."""
    return np.log1p(np.maximum(x, 0.0)) - np.log1p(np.maximum(baseline, 0.0))


def window_features(win, baseline):
    """win: (N_POINTS, 4) neu abgetastete Rohwerte, baseline: (4,). Rueckgabe (52,) float32."""
    z = to_log(win, baseline)
    n = z.shape[0]
    tt = np.linspace(-1.0, 1.0, n)
    feats = []
    for c in range(z.shape[1]):
        v = z[:, c]
        d = np.diff(v)
        feats += [
            v[-1], v.mean(), v.std(), v.min(), v.max(),
            float(np.dot(tt, v - v.mean()) / np.dot(tt, tt)),       # Steigung
            v[: n // 2].mean(), v[n // 2:].mean(),
            np.abs(d).max(), d.mean(), v.max() - v.min(), v[-1] - v[0],
            np.argmax(np.abs(v)) / (n - 1),
        ]
    return np.asarray(feats, dtype=np.float32)


def intensity(win, baseline):
    """Geruchsintensitaet: Summe der |log-Abweichung| der letzten Probe ueber alle Kanaele.

    Das Modell normiert jede Probe per Min-Max (Konzentration geht verloren), deshalb kommt
    die Intensitaet fuer Gate und Quellensuche von hier.
    """
    return float(np.abs(to_log(win[-1], baseline)).sum())


def update_baseline(baseline, sample, alpha=0.002):
    """Grundlinie langsam nachfuehren, nur nach unten schnell (saubere Luft = Minimum)."""
    sample = np.asarray(sample, dtype=np.float64)
    if baseline is None:
        return sample.copy()
    return np.where(sample < baseline, sample, baseline + alpha * (sample - baseline))
