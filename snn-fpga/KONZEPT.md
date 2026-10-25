# Konzept: FlyWire-Olfaktorik-SNN auf 10× Tang Nano 20K für den DeepRobotics Lite3

Ziel: Der Lite3 bekommt eine „Fliegennase“. Eine Gassensor-Batterie liefert Messwerte, ein
Cluster aus 10 Tang Nano 20K führt das biologisch beschränkte Spiking Neural Network
[`Vick-MIRE/flywire-olfactory-snn`](https://huggingface.co/Vick-MIRE/flywire-olfactory-snn)
in Echtzeit aus, und der Roboter reagiert auf das Ergebnis (Geruch melden oder zur Quelle laufen).

> **Stand der Annahmen:** Die Modelldaten (52 Eingänge, 800 LIF-Neuronen, dünnbesetzte
> 800×800-Konnektom-Maske, 500 Klassen, Norse/PyTorch, `.safetensors` + `connectome_mask.npz`)
> stammen aus der Modellbeschreibung. Vor dem HDL-Entwurf mit
> [`tools/inspect_model.py`](tools/inspect_model.py) vermessen (Abschnitt 9).

---

## 1. Korrekturen zur ursprünglichen KI-Antwort

Die Roadmap aus der KI-Antwort stimmt in der Richtung (Quantisierung, Partitionierung,
Spike-Kommunikation), unterschätzt aber den Chip und setzt den Engpass an die falsche Stelle:

| Punkt | KI-Antwort | Richtig |
|---|---|---|
| Speicher GW2AR-18 | „~828 Kb BRAM“ | 828 Kbit BSRAM (46 × 18 Kbit) **plus 64 Mbit SDRAM im selben Gehäuse** (GW2AR = mit eingebettetem SDRAM) |
| Engpass | rekurrente 800×800-Matrix | Die **Readout-Schicht** 800×500 (≈3,2 Mbit bei INT8) ist der größte Block. Die rekurrente Matrix ist bei ~5–15 % Dichte nur ~0,3–1 Mbit |
| Rechenleistung | Cluster nötig, weil ein Chip zu klein ist | Rechnerisch reichen 1–3 Boards. Das Netz ist ereignisgetrieben, ein Zeitschritt kostet einige tausend Additionen |
| Partitionierung mit METIS | Kanten zwischen FPGAs minimieren | Mit einem **Spike-Vektor-Broadcast** (Abschnitt 4) bekommt jedes Board ohnehin alle 800 Spikes. Wie viele Kanten Boardgrenzen kreuzen, spielt dann keine Rolle; METIS ist unnötig |
| Eingang | 52 Kanäle direkt einspeisen | Die 52 Eingänge sind Rezeptor-/Glomerulus-Kanäle der Fliege. **Kein technischer Gassensor liefert diese Signale.** Eine Adapterschicht muss neu trainiert werden (Abschnitt 3) |
| Ausgang | 500 Klassen | Die 500 Geruchsklassen kommen aus dem Trainingsdatensatz und haben nichts mit deinen Sensoren zu tun. Die Readout-Schicht wird für eigene Klassen neu trainiert |

**Konsequenz:** Der Cluster ist sinnvoll, aber aus anderen Gründen als in der KI-Antwort.
Er verteilt den Speicher, sodass alles im BSRAM bleibt und kein SDRAM-Controller nötig ist.
Er gibt Reserve für größere Konnektom-Ausschnitte und ermöglicht Varianten wie Stereo-Riechen
oder Redundanz (Abschnitt 8).

---

## 2. Systemüberblick

```
            ┌──────────────────────── Lite3 ─────────────────────────┐
 Luft ──►   │  Sensorkopf (vorn, tief)                               │
            │  5× BME688 + SGP41 + T/rH                              │
            │        │ I²C                                           │
            │        ▼                                               │
            │  Sensor-MCU (Wio Terminal aus diesem Repo / RP2040)    │
            │   - Heizprofile, Normierung, Driftkompensation         │
            │   - 52-Byte-Merkmalsvektor @ 1–10 Hz                   │
            │        │ UART 3 Mbit/s                                 │
            │        ▼                                               │
            │  ┌──────────── FPGA-Cluster (Backplane) ─────────────┐ │
            │  │ B0 IO/Encoder ─► B1 ─► B2 ─► … ─► B8 ─► B9 Readout│ │
            │  │   ▲                                         │     │ │
            │  │   └──────────────── Spike-Ring ◄────────────┘     │ │
            │  └───────────────────────────────────────────────────┘ │
            │        │ UART/USB (Klasse, Konfidenz, Spikeraten)      │
            │        ▼                                               │
            │  Wahrnehmungsrechner / Payload-SBC (ROS 2)             │
            │   - /olfaction/odor, /olfaction/concentration          │
            │   - Verhalten: Cast-and-Surge-Quellensuche             │
            │        │ Ethernet / UDP (DeepRobotics SDK)             │
            │        ▼                                               │
            │  Bewegungssteuerung (Geschwindigkeitsbefehle)          │
            └────────────────────────────────────────────────────────┘
```

Sinnvolle Arbeitsteilung: Die FPGAs rechnen nur das SNN. Sensorlogik (I²C, Heizprofile,
Kalibrierung) liegt auf dem Mikrocontroller, Verhalten und Robotik in ROS 2. So bleibt jede
Komponente einzeln testbar.

---

## 3. Sensorik und Eingangsadapter

### 3.1 Hardware

| Option | Kanäle | Bemerkung |
|---|---|---|
| **5× Bosch BME688** mit je 10 Heizstufen | 50 + T + rH = **52** | Passt genau auf 52 Eingänge. Die Heizprofile erzeugen „virtuelle Rezeptoren“ mit unterschiedlicher Selektivität, ähnlich einer Rezeptorfamilie |
| Grove Multichannel Gas v2 (aus diesem Repo) | 4 (NO2, CO, C2H5OH, VOC) | Für den Prototyp ausreichend: 4 Kanäle × 13 Zeitmerkmale (Wert, Ableitung, Fenster) = 52 |
| Sensirion SGP41 zusätzlich | 2 (VOC/NOx-Rohsignal) | Robust, gut als Referenzkanal |

Der Sensorkopf sitzt vorne unten am Rumpf, weg von Motor- und Lüfterabluft, möglichst mit
leichter Ansaugung (kleiner Lüfter, konstanter Volumenstrom). Der Gang des Lite3 verwirbelt
die Luft, daher hilft ein Prallblech.

### 3.2 Merkmalsvektor (MCU)
1. Rohwerte → Baseline-Abzug (gleitendes Minimum über Minuten, gegen Drift)
2. log-Skalierung (MOX-Sensoren antworten etwa logarithmisch auf Konzentration)
3. Z-Normierung mit gespeicherten Statistiken → Abbildung auf `uint8` (0–255)
4. Senden als Frame `0xA5 | seq | 52×u8 | crc8` an Board 0

### 3.3 Neu trainieren statt direkt einspeisen

Das Konnektom (`W_rec` + Maske) ist der biologische Teil und bleibt **eingefroren**.
Neu trainiert werden:

- `W_in` (52 → 800) auf deine Sensormerkmale (oder ein vorgeschalteter 52×52-Adapter, der Sensorkanäle auf „Fliegenrezeptoren“ abbildet, wenn die originale `W_in` erhalten bleiben soll)
- `W_out` (800 → K) für deine eigenen K Klassen (z. B. Luft, Ethanol, Kaffee, Rauch, Lösungsmittel, Unbekannt)

Training in Norse mit Surrogate Gradients und **quantisierungsbewusstem Training** (Gewichte
im Forward-Pass auf INT8 runden, Membranpotential auf 16 Bit begrenzen). Dann stimmt das
FPGA-Ergebnis bitgenau mit dem Python-Referenzmodell überein.

Datensatz: auf dem Roboter aufnehmen (stehend und laufend). Pro Klasse mehrere Sessions an
verschiedenen Tagen, wegen Sensordrift, Temperatur und Feuchte.

---

## 4. FPGA-Cluster-Architektur

### 4.1 Ressourcen Tang Nano 20K (GW2AR-LV18QN88C8/I7)
- 20.736 LUT4, 15.552 FF, 48 Multiplizierer (18×18)
- 828 Kbit BSRAM + 64 Mbit eingebettetes SDRAM
- 27-MHz-Oszillator, PLL bis > 100 MHz
- BL616 als USB-JTAG/UART-Brücke
- 3,3-V-I/O über Stiftleisten

### 4.2 Rollenverteilung

| Board | Aufgabe | Speicher (Schätzung, INT8, 8 % Dichte) |
|---|---|---|
| **B0** | UART zur MCU, Eingangspuffer, Poisson-/Rate-Encoder (LFSR), Takt- und Zeitschritt-Master, Ring-Master | < 50 Kbit |
| **B1–B8** | je 100 LIF-Neuronen: Eingangsstrom `W_in·x`, rekurrente Synapsen, Membrandynamik, Spike-Zähler, Teilsummen der Readout-Schicht | ≈ 540 Kbit (siehe Skriptausgabe) |
| **B9** | Sammelt die Readout-Teilsummen, Bias, Argmax/Softmax-Näherung, Konfidenz, UART/USB zum Roboter, Debug-Spike-Monitor | < 50 Kbit |

800 / 8 = 100 Neuronen pro Board ergibt glatte Adressen (7-Bit-Index). B0 und B9 als reine
IO-Knoten zu bauen hält die acht Neuronen-Boards **identisch**: gleicher Bitstream, nur die
Speicherinitialisierung und die `BOARD_ID` unterscheiden sich.

### 4.3 Neuronenkern (pro Board, zeitmultiplexed)

Pro Board gibt es **einen** Rechenkern, der 100 Neuronen nacheinander bearbeitet. 100
parallele Neuronen wären Verschwendung, denn die Taktrate ist um Größenordnungen höher als die
nötige Zeitschrittrate.

Ablauf pro Zeitschritt *t*:

```
1. Empfange Spike-Vektor s[t-1] (800 Bit) vom Ring
2. Für jedes j mit s[j]=1:               // ereignisgetrieben
       für jede Synapse (j → i) auf diesem Board:   // CSR, nach präsyn. j sortiert
           I[i] += W_rec[j,i]
3. Für jeden Eingangskanal k mit Spike e[k]=1 (von B0 mitgesendet):
           I[i] += W_in[k,i]   (für alle i dieses Boards)
4. Für i = 0..99:                        // LIF-Update, Norse-kompatibel
       v[i] += (I[i] - v[i]) >>> SHIFT_MEM   // Leak als Shift: dt/tau = 2^-SHIFT
       I[i] -= I[i] >>> SHIFT_SYN
       if v[i] >= V_TH: spike[i]=1; v[i]=V_RESET; count[i]++
5. Lege eigene 100 Spike-Bits in den Ring-Frame
```

- `tau_mem` und `tau_syn` werden beim Training auf Werte festgelegt, bei denen `dt/tau` eine Zweierpotenz ist. Das spart Multiplizierer.
- Gewichte INT8, Ströme und Potentiale INT16 mit Sättigung
- Speicher in BSRAM:
  - `ptr[800]`: Startadresse der Synapsenliste je präsynaptischem Neuron
  - `syn[]`: {Ziel-Index 7 Bit, Gewicht 8 Bit}
  - `w_in[52][100]`
  - `w_out[100][K]`
  - Zustände `v`, `I`, `count`

### 4.4 Readout verteilt berechnen

Am Ende eines Inferenzfensters (T Zeitschritte) berechnet jedes Board die Teilsumme über
seine eigenen Neuronen:

```
logit_partial[c] = Σ_i count[i] · W_out[i,c]       (100·K MACs, Multiplizierer)
```

Die Teilsummen laufen über den Ring nach B9 und werden dort addiert. Damit braucht kein Board
die ganze 800×K-Matrix. Bei K = 500 sind das ≈ 390 Kbit pro Board und passen noch ins BSRAM.
Bei K ≈ 10 eigenen Klassen ist die Readout-Schicht vernachlässigbar.

Vorher prüfen, ob das Originalmodell auf Spike-Zählern ausliest oder auf dem Membranpotential
einer nicht feuernden LI-Ausgangszelle. Die zweite Variante läuft genauso, nur werden die
Spikes dann bei jedem Zeitschritt akkumuliert statt am Fensterende.

---

## 5. Inter-FPGA-Kommunikation: Spike-Vektor-Ring

Statt AER-Paketen (Adresse pro Spike, variable Last) zirkuliert **pro Zeitschritt ein fester
Frame**. Er enthält alle 800 Spike-Bits plus die 52 Eingangs-Spikes. Das Ergebnis ist
deterministisch, braucht keine Arbitrierung und hat konstante Latenz.

```
Frame (TDM, 4-Bit-Bus):  SYNC | t[16] | CMD[8] | e[52] | s[800] | CRC8
                         └── 4 + 4 + 2 + 13 + 200 + 2 ≈ 225 Takte
```

- **Physik:** quellsynchron, 4 Datenleitungen + Takt + Frame-Valid = 6 Leitungen raus und 6 rein pro Board, dazu GND zwischen den Signalen. 3,3 V LVCMOS, kurze Wege auf einer Backplane
- **Takt:** 25–50 MHz Ringtakt. Bei 25 MHz dauert ein Frame ≈ 9 µs, plus etwa 2 Takte pro Hop
- **Ablauf:** B0 startet den Frame. Jedes Board übernimmt den Frame (Eingang → asynchrones FIFO → eigene Taktdomäne), ersetzt seinen 100-Bit-Slot durch die neuen Spikes und gibt ihn weiter. B9 schließt den Ring zu B0
- **Zeitschritt-Semantik:** Neuronen rechnen in Schritt *t* mit `s[t-1]`, genau wie die diskrete Simulation in Norse. Das macht das System bitgenau zur Referenz
- **Steuerframes** (`CMD`): Reset, Fenster-Start/-Ende, Readout-Teilsummen sammeln (dann trägt der Frame K × 24 Bit), Gewichte nachladen (optional), Status/Heartbeat

### Zeitbudget (Schätzung)
| Schritt | Takte | bei 54 MHz Kerntakt |
|---|---|---|
| Ring-Frame | ≈ 250 | 5 µs |
| Synapsen: 80 Spikes × ≈ 8 lokale Ziele | ≈ 650 | 12 µs |
| LIF-Update 100 Neuronen | 100 | 2 µs |
| **Summe pro Zeitschritt** | | **≈ 20 µs → ~50.000 Schritte/s** |

Eine Inferenz mit T = 100 Schritten dauert etwa 2 ms. Die MOX-Sensoren reagieren im Bereich
von Sekunden. **Echtzeit ist kein Problem**, die Reserve deckt auch mehrere Inferenzen pro
Messung ab (Ensemble über verschiedene Poisson-Seeds).

---

## 6. Toolchain und Build

```
Hugging Face  ─►  export.py (Python)                     ─►  build/board_<n>/*.hex
(.safetensors,    - Laden, Maske anwenden                     (ptr, syn, w_in, w_out, params)
 mask.npz)        - QAT-Gewichte → INT8, Partitionierung
                  - bitgenaues Fixpunkt-Referenzmodell (numpy)
                                │
                                ▼
rtl/ (Verilog, parametrisch: BOARD_ID, N_LOCAL, K)       ─►  Verilator-Testbench
  lif_core.v  syn_engine.v  ring_node.v  uart.v               vergleicht Spikes Zeitschritt für
  encoder.v (B0)  readout_sum.v (B9)  top_*.v                 Zeitschritt mit der Referenz
                                │
                                ▼
Yosys + nextpnr-himbaechel (Gowin) + Apycula    oder  Gowin EDA (Education)
                                │
                                ▼
openFPGALoader -b tangnano20k  (10 Boards über USB-Hub, per Seriennummer adressiert)
```

- Speicherinitialisierung über `$readmemh` in den BSRAM-Inferenz-Templates. Pro Board wird nur die Init-Datei getauscht. Der Bitstream wird zwar pro Board neu gepackt, die Synthese aber nur einmal für die Neuronen-Boards durchgeführt
- Alternative ohne Neusynthese: Gewichte zur Laufzeit über den Ring von B0 laden (`CMD = LOAD`). Dann sind alle acht Neuronen-Boards bitstreamgleich und neue Trainings gehen ohne FPGA-Build
- CI: Referenzmodell plus Verilator-Simulation als GitHub-Action (analog zum bestehenden PlatformIO-Workflow)

---

## 7. Integration in den Lite3

### 7.1 Mechanik und Strom
- **Backplane-Platine** mit 10 Steckplätzen für die Tang Nano 20K. Die Ring-Leitungen laufen als Leiterbahnen, keine Dupont-Kabel, denn die Vibrationen beim Laufen lösen Steckbrücken. Boards zusätzlich verschrauben oder verklemmen
- Versorgung: 10 × ≈ 0,3–0,5 W ≈ 3–5 W → DC/DC 5 V / 3 A aus der Payload-Versorgung des Lite3. Spannung und Belastbarkeit des Payload-Anschlusses im Lite3-Handbuch prüfen
- Gehäuse auf der Payload-Schiene, Spritzwasser- und Staubschutz, passive Kühlung reicht

### 7.2 Software auf dem Roboter
- ROS-2-Knoten `olfaction_bridge` auf dem Wahrnehmungsrechner oder einem eigenen Payload-SBC (z. B. Raspberry Pi 5 per Ethernet am Roboter-Switch):
  - liest UART-Frames von B9: `seq | class | confidence | K Logits | Fenster-Spikerate`
  - publiziert `/olfaction/odor` (Klasse + Konfidenz), `/olfaction/intensity` (Gesamtspikerate als Konzentrationsmaß), `/olfaction/raw` (Debug)
- Verhaltensknoten, inspiriert von Insekten-Chemotaxis (**Cast-and-Surge**):
  - Geruch erkannt und Intensität steigt → geradeaus gegen den Wind/Gradienten („surge“)
  - Geruch verloren → Zickzack quer mit wachsender Amplitude („cast“)
  - Intensität über Schwelle → Stop, Meldung
  - Geschwindigkeitsbefehle über das DeepRobotics-SDK bzw. die ROS-2-Schnittstelle des Lite3. Ports und Nachrichtenformate aus der SDK-Doku der jeweiligen Firmware-Version übernehmen
- Sicherheit: Das Verhalten gibt nur Sollgeschwindigkeiten mit Begrenzung vor. Not-Aus und Hinderniserkennung des Roboters haben immer Vorrang. Fällt der Heartbeat von B9 aus, schaltet der Knoten auf „Olfaktorik inaktiv“

---

## 8. Varianten für die 10 Boards

| Variante | Aufteilung | Nutzen |
|---|---|---|
| **A – Referenz (empfohlen)** | B0 IO, B1–B8 je 100 Neuronen, B9 Readout | Einfach, alles im BSRAM, identische Neuronen-Boards |
| B – Stereo-Riechen | 2 Sensorköpfe links/rechts, 2 Netzinstanzen à 4 Boards (200 Neuronen/Board), B0 und B9 gemeinsam | Fliegen vergleichen beide Antennen und erkennen so die Richtung. Das Links/Rechts-Signal steuert direkt die Drehrate |
| C – Redundanz | 3 Instanzen à 3 Boards + 1 Voter | Mehrheitsentscheid gegen Störungen (Vibration, EMV), Ensemble über verschiedene Seeds |
| D – Größeres Konnektom | alle 10 Boards als Neuronen-Boards, bis ~3.000 Neuronen mit SDRAM | Wenn später ein größerer FlyWire-Ausschnitt (z. B. mit Pilzkörper-Kenyonzellen) genutzt werden soll |

Start mit **A**. Ist A lauffähig, ist der Wechsel zu **B** reine Konfigurationssache, denn
Kern und Ring sind gleich.

---

## 9. Umsetzungsplan

| Phase | Inhalt | Ergebnis / Abnahme |
|---|---|---|
| **0 – Analyse** (1 Woche) | Modell laden, `inspect_model.py` ausführen, Norse-Code lesen (τ, Schwelle, dt, T, Encoding, Readout-Art), Lizenz prüfen | Tabelle der echten Parameter, Speicherbudget bestätigt |
| **1 – Referenz** (2 Wochen) | Fixpunkt-Referenzmodell in numpy, Vergleich mit Norse-Float auf dem Originaldatensatz | Genauigkeitsverlust durch INT8/Shift-Leak < 2 % |
| **2 – Ein Board** (3 Wochen) | `lif_core` + `syn_engine` für 100 Neuronen, Verilator-Vergleich, dann auf ein Tang Nano 20K; Spikes per UART auslesen | Bitgenaue Spikes gegenüber Referenz |
| **3 – Ring** (3 Wochen) | `ring_node`, Backplane-Prototyp mit 3 Boards, dann 10; Fehlerzähler (CRC), Langzeittest | > 10⁹ Frames ohne CRC-Fehler; volle 800-Neuronen-Inferenz bitgenau |
| **4 – Sensorik** (parallel, 3 Wochen) | Sensorkopf, MCU-Firmware (Wio Terminal aus diesem Repo erweitern), Datensatz aufnehmen | Datensatz ≥ 6 Klassen, mehrere Tage |
| **5 – Training** (2 Wochen) | `W_in`/`W_out` mit QAT neu trainieren, Konnektom fix | Offline-Genauigkeit auf Testtagen; Export → FPGA |
| **6 – Roboter** (3 Wochen) | Strom/Mechanik, ROS-2-Bridge, Klassifikation im Stand und beim Laufen, dann Cast-and-Surge | Lite3 findet eine Ethanolquelle im Raum |

### Erster Schritt (heute)
```bash
pip install huggingface_hub numpy
huggingface-cli download Vick-MIRE/flywire-olfactory-snn --local-dir flywire-snn
python3 snn-fpga/tools/inspect_model.py flywire-snn/<gewichte>.safetensors \
        flywire-snn/connectome_mask.npz --boards 8 --wbits 8
```
Das Skript gibt Tensorformen, INT8-Quantisierungsfehler, Dichte und Fan-in/-out der Maske
und den BSRAM-Bedarf pro Board aus. Damit lassen sich die Schätzungen oben bestätigen oder
korrigieren. Liegt ein Board über ≈ 700 Kbit, wandert `w_out` ins eingebettete SDRAM, oder
die Klassenzahl wird verkleinert.

---

## 10. Risiken

| Risiko | Gegenmaßnahme |
|---|---|
| Biologischer Mehrwert geht durch Neutraining verloren | Konnektom fest lassen; Vergleich gegen ein gleich großes Zufallsnetz als Kontrolle. Das zeigt, ob die Fliegenstruktur hilft |
| Sensordrift / Temperatur / Feuchte | Baseline-Tracking auf der MCU, T/rH als Eingänge, Daten über viele Tage |
| Luftströmung durch Gang und Lüfter | Ansaugung mit konstantem Volumenstrom, Sensorkopf vorn unten, Klassifikation beim Laufen separat validieren |
| Signalintegrität im Ring | Backplane statt Kabel, moderater Takt (25 MHz), CRC und Fehlerzähler, Takt-Neuabgleich je Frame |
| Unklare Norse-Details (Reset-Art, Readout, Encoding) | Phase 0/1: bitgenaue Referenz vor jeder HDL-Zeile |
| Lizenz der Modellgewichte / FlyWire-Daten | Lizenz im Modell-Repository und die FlyWire-Nutzungsbedingungen vor einer Veröffentlichung prüfen |
