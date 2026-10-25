#!/usr/bin/env python3
"""Live-Betrieb: Gassensor (Wio Terminal) -> Merkmale -> FPGA (Tang Nano 20K) -> erkannter Geruch.

    python3 nose_bridge.py sensor-model --sensor-port /dev/cu.usbmodem1101 \
            --fpga-port /dev/cu.usbserial-20250303171 [--ros2]
    python3 nose_bridge.py sensor-model --replay daten/kaffee_....npz --no-fpga    # Test ohne Hardware

Einmal pro Sekunde wird ein 10-s-Fenster ausgewertet (wie im Training, sensor_features.py).
Liegt die Intensitaet unter der Gate-Schwelle, ist das Ergebnis "luft" ohne Klassifikation.
Sonst gehen die Raten an den FPGA (Befehl 'C' von top_tangnano20k_full); beim Start werden die
Ausgangsgewichte ins SDRAM geladen. --no-fpga rechnet bitgleich mit snn_ref.py auf dem Host.

Ausgabe: eine JSON-Zeile pro Sekunde {t, label, confidence, intensity}; mit --ros2 zusaetzlich
/olfaction/odor (std_msgs/String, JSON) und /olfaction/intensity (std_msgs/Float32).
"""
import argparse
import json
import struct
import time
from collections import deque
from pathlib import Path

import numpy as np

import board_test
import sensor_features as sf
import snn_ref


class Classifier:
    def __init__(self, model_dir, fpga_port, baud, skip_load, runs=5):
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "sensor_meta.json").read_text())
        self.mean = np.array(self.meta["mean"], dtype=np.float32)
        self.std = np.array(self.meta["std"], dtype=np.float32)
        p = snn_ref.load_params(model_dir / "model.safetensors", model_dir / "connectome_mask.npz")
        self.q = snn_ref.quantize(p, runs)
        self.scale = 2.0 ** self.q.s_out * snn_ref.T_STEPS * runs
        self.seed = 1
        self.ser = None
        if fpga_port:
            import serial
            self.ser = serial.Serial(fpga_port, baud, timeout=10)
            self.ser.reset_input_buffer()
            if not skip_load:
                words = board_test.w_out_words(self.q)
                board_test.load_sdram(self.ser, words)
                board_test.check_sdram(self.ser, words)
            board_test.calibrate(self.ser, board_test.w_out_words(self.q))

    def __call__(self, feats):
        x = ((feats - self.mean) / self.std).astype(np.float32)[None, :]
        r_q = snn_ref.rate_quant(snn_ref.rate_encode(x))
        self.seed = (self.seed + 1) & snn_ref.MASK32
        if self.ser:
            k = self.q.w_out.shape[0]
            self.ser.write(bytes([0xA5]) + b"C" + struct.pack("<I", self.seed)
                           + b"".join(struct.pack("<H", int(v)) for v in r_q[0]))
            resp = board_test.read_exact(self.ser, 2 + 2 + 4 + 4 * k + 4)
            logits = np.frombuffer(resp[8:-4], dtype="<i4").astype(np.int64)
        else:
            e = snn_ref.bernoulli_lfsr(r_q, self.q.runs, self.seed)
            logits = snn_ref.fixed_forward(self.q, r_q, e)[0][0]
        z = logits / self.scale
        prob = np.exp(z - z.max())
        prob /= prob.sum()
        c = int(logits.argmax())
        return self.meta["labels"][c], float(prob[c])


def sensor_stream(port, baud):
    import serial
    with serial.Serial(port, baud, timeout=1) as ser:
        ser.reset_input_buffer()
        t0 = time.time()
        while True:
            parts = ser.readline().decode(errors="ignore").strip().split(",")
            if len(parts) == 4:
                try:
                    yield time.time() - t0, np.array([float(v) for v in parts])
                except ValueError:
                    pass


def replay_stream(path):
    d = np.load(path)
    for t, x in zip(d["t"], d["x"]):
        yield float(t), x.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="Ausgabe von train_sensor_snn.py")
    ap.add_argument("--sensor-port")
    ap.add_argument("--sensor-baud", type=int, default=115200)
    ap.add_argument("--replay", help="Aufnahme (record_sensors.py) statt Live-Sensor abspielen")
    ap.add_argument("--fpga-port")
    ap.add_argument("--fpga-baud", type=int, default=1_000_000)
    ap.add_argument("--no-fpga", action="store_true", help="auf dem Host rechnen (snn_ref, bitgleich)")
    ap.add_argument("--skip-load", action="store_true", help="w_out ist schon im SDRAM")
    ap.add_argument("--ros2", action="store_true")
    args = ap.parse_args()
    if not args.no_fpga and not args.fpga_port:
        raise SystemExit("--fpga-port oder --no-fpga angeben")

    clf = Classifier(args.model_dir, None if args.no_fpga else args.fpga_port, args.fpga_baud, args.skip_load)
    gate = clf.meta["gate"]
    pub = None
    if args.ros2:
        import rclpy
        from std_msgs.msg import Float32, String
        rclpy.init()
        node = rclpy.create_node("olfaction_bridge")
        pub = (node.create_publisher(String, "/olfaction/odor", 10),
               node.create_publisher(Float32, "/olfaction/intensity", 10), String, Float32)

    stream = replay_stream(args.replay) if args.replay else sensor_stream(args.sensor_port, args.sensor_baud)
    win = deque(maxlen=sf.N_POINTS)
    base = None
    t_grid = None
    last_t, last_x = None, None
    step = int(sf.RATE_HZ)                      # 1 Auswertung pro Sekunde
    n_new = 0
    for t, x in stream:
        if t_grid is None:
            t_grid, last_t, last_x = t, t, x
        while t_grid <= t:                      # auf 10 Hz neu abtasten (wie im Training)
            a = 0.0 if t == last_t else (t_grid - last_t) / (t - last_t)
            xg = last_x + a * (x - last_x)
            base = sf.update_baseline(base, xg)
            win.append(xg)
            n_new += 1
            t_grid += 1.0 / sf.RATE_HZ
            if len(win) == sf.N_POINTS and n_new >= step:
                n_new = 0
                w = np.array(win)
                inten = sf.intensity(w, base)
                if inten < gate:
                    label, conf = clf.meta["gate_label"], 1.0
                else:
                    label, conf = clf(sf.window_features(w, base))
                msg = {"t": round(t_grid, 1), "label": label, "confidence": round(conf, 3),
                       "intensity": round(inten, 3)}
                print(json.dumps(msg), flush=True)
                if pub:
                    pub[0].publish(pub[2](data=json.dumps(msg)))
                    pub[1].publish(pub[3](data=float(inten)))
        last_t, last_x = t, x


if __name__ == "__main__":
    main()
