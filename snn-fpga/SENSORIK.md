# Eigene Gerüche: Sensor → Neutraining → FPGA → Lite3

Das veröffentlichte Modell kennt 692 DoOR-Duftstoffe, gemessen als Antworten von Fliegenrezeptoren.
Technische Gassensoren liefern andere Signale. Deshalb wird das Netz auf die eigenen Sensoren und
Gerüche neu trainiert:
- **Eingefroren:** das Konnektom, also Topologie und gelernte rekurrente Gewichte
- **Neu trainiert:** die Eingangsschicht und eine neue Ausgangsschicht mit eigenen Klassen

```
Wio Terminal (Firmware dieses Repos, Trainingsmodus)  ── CSV no2,co,c2h5oh,voc ──►  Host (PC / Lite3-SBC)
                                                                                     │ 10-s-Fenster, 1/s
                                                                                     │ 52 Merkmale, Gate
Tang Nano 20K (snn_full.fs)  ◄── 'C' Raten ── nose_bridge.py ── JSON / ROS 2 ──────► Lite3-Verhalten
```

## 1. Merkmale ([`tools/sensor_features.py`](tools/sensor_features.py))

- Fenster von 10 s, neu abgetastet auf 10 Hz (100 Punkte); jede Sekunde eine Auswertung.
- Pro Kanal: `log1p(Rohwert) − log1p(Grundlinie)`, die Grundlinie läuft langsam nach (Minimum = saubere Luft).
- 13 Merkmale pro Kanal ergeben 4 × 13 = 52 = Eingangsbreite des Modells: letzter Wert, Mittel, Streuung, Min, Max, Steigung, Mittel erste/zweite Hälfte, größte Änderung, mittlere Änderung, Spannweite, Gesamtänderung, Lage des Maximums.
- **Intensität** = Summe der |log-Abweichung| der letzten Probe. Das Modell normiert jede Probe per Min-Max und verliert dabei die Konzentration. Die Intensität dient deshalb als Gate (unter der Schwelle → „luft“) und als Signal für die Quellensuche.

## 2. Aufnehmen ([`tools/record_sensors.py`](tools/record_sensors.py))

```bash
python3 snn-fpga/tools/record_sensors.py --port /dev/cu.usbmodemXXXX --label luft   --seconds 300 --out daten/
python3 snn-fpga/tools/record_sensors.py --port /dev/cu.usbmodemXXXX --label kaffee --seconds 120 --out daten/
```

Pro Aufnahme so vorgehen:
1. 30 s saubere Luft
2. Probe vor den Sensor halten, 60–90 s
3. Probe wegnehmen

Pro Geruch mindestens 3–4 Aufnahmen an verschiedenen Tagen machen, denn Drift, Temperatur und
Feuchte schwanken. Die Klasse `luft` muss dabei sein, aus ihr wird die Gate-Schwelle berechnet.

## 3. Neu trainieren ([`tools/train_sensor_snn.py`](tools/train_sensor_snn.py))

```bash
pip install torch norse scipy safetensors numpy
python3 snn-fpga/tools/train_sensor_snn.py flywire-snn daten/ --out sensor-model
```

- **Validierung:** Pro Klasse wird die letzte Aufnahme zurückgehalten, wenn mindestens 2 vorhanden sind. Das Skript meldet die Genauigkeit für Float und für Fixpunkt (= FPGA).
- **Ausgabe:** `sensor-model/` im selben Format wie das Hugging-Face-Modell, dazu `sensor_meta.json` mit Klassen, Normierung und Gate-Schwelle.
- **Ohne Sensor testen:** `--synthetic synth-daten` erzeugt künstliche Aufnahmen für 5 Klassen.

## 4. Auf den FPGA bringen

```bash
python3 snn-fpga/tools/verify_torch.py sensor-model                 # Float-Referenz == PyTorch
python3 snn-fpga/tools/export_variant_e.py sensor-model/model.safetensors sensor-model/connectome_mask.npz --out build/sensor
snn-fpga/rtl/sim.sh   build/sensor                                  # PASS erwartet
snn-fpga/rtl/build.sh build/sensor full                             # -> build/sensor/snn_full.fs
openFPGALoader -b tangnano20k build/sensor/snn_full.fs
```

Neu gebaut werden muss nach jedem Neutraining: `w_in`, Klassenzahl und Bias stecken im Bitstream.
`w_out` lädt `nose_bridge.py` beim Start ins SDRAM.

## 5. Live-Betrieb ([`tools/nose_bridge.py`](tools/nose_bridge.py))

```bash
python3 snn-fpga/tools/nose_bridge.py sensor-model --sensor-port /dev/cu.usbmodemXXXX \
        --fpga-port /dev/cu.usbserial-XXXX1            # optional: --ros2
# Test ohne Hardware: Aufnahme abspielen, bitgleich auf dem Host rechnen
python3 snn-fpga/tools/nose_bridge.py sensor-model --replay daten/kaffee_....npz --no-fpga
```

Ausgabe: eine JSON-Zeile pro Sekunde, z. B. `{"t": 40.1, "label": "kaffee", "confidence": 0.999, "intensity": 1.944}`.
Mit `--ros2` erscheinen die Werte zusätzlich als `/olfaction/odor` (`std_msgs/String`, JSON) und
`/olfaction/intensity` (`std_msgs/Float32`). Auf diese Topics setzt das Cast-and-Surge-Verhalten
aus [KONZEPT.md](KONZEPT.md), Abschnitt 7.2, auf.

## Geprüft (synthetische Daten, 5 Klassen)

| Schritt | Ergebnis |
|---|---|
| Training (30 Epochen) | Validierung: Float 95,5 %, Fixpunkt 95,8 % |
| `verify_torch.py` | Logit-Abweichung 1·10⁻⁶ |
| Kern-Simulation (600 Zeitschritte) und Gesamtsystem mit SDRAM über UART | bitgenau, PASS |
| `nose_bridge.py --replay --no-fpga` | Luft → `luft` (Gate), Probe → richtige Klasse, danach wieder `luft` |

Synthetische Daten sind einfacher als echte. Wie gut die Fliegen-Verschaltung mit dem
Grove-Sensor und seinen 4 breitbandigen MOX-Kanälen trennt, zeigen erst eigene Aufnahmen. Mehr
Kanäle helfen, z. B. BME688 mit Heizprofilen (KONZEPT.md, Abschnitt 3).
