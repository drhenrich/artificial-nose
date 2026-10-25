# Konzept: FlyWire-Olfaktorik-SNN auf 10× Tang Nano 20K für den DeepRobotics Lite3

Ziel: Der Lite3 bekommt eine „Fliegennase“. Eine Gassensor-Batterie liefert Messwerte, ein
Cluster aus 10 Tang Nano 20K führt das biologisch beschränkte Spiking Neural Network
[`Vick-MIRE/flywire-olfactory-snn`](https://huggingface.co/Vick-MIRE/flywire-olfactory-snn)
in Echtzeit aus, und der Roboter reagiert auf das Ergebnis (Geruch melden oder zur Quelle laufen).

> **Stand:** Architektur aus Model Card und `modeling_snn.py` übernommen (Abschnitt 0).
> Offen sind nur noch die Zahlen der echten Gewichtsdateien (Dichte, Anteil gekippter
> Vorzeichen, Wertebereiche). Sie liefert
> [`tools/inspect_model.py`](tools/inspect_model.py) (Abschnitt 9).

---

## 0. Modell laut Model Card und `modeling_snn.py`

```
x (52 DoOR-Rezeptorwerte, standardisiert)
  → r = (x − min x) / (max x − min x)          Min-Max pro Messung → [0, 1]
  für t = 1..20:
     e_t   = Bernoulli(r)                       Poisson-Spikes
     I_t   = W_in · (e_t ⊙ r)                   Linear(52 → 800, ohne Bias): Spike × Rate!
           + W_eff · spk_{t-1}                  W_eff = recurrent ⊙ mask ⊙ sign
     spk_t = LIFCell(I_t)
  → rate = Σ spk_t / 20
  → Linear(800 → 692, mit Bias)                 = Logits
```

| Merkmal | Wert | Folge für die Hardware |
|---|---|---|
| Hidden-Neuronen | 800 (Projektionsneuronen des Antennallobus + Kenyonzellen des Pilzkörpers, FlyWire-IDs in `connectome_meta.json`) | Kenyonzellen feuern biologisch sehr spärlich, das passt gut zur ereignisgetriebenen Verarbeitung |
| Synapsen | **14.998** gerichtete Kanten → Dichte **2,34 %**, im Mittel 18,7 Ein- und Ausgänge pro Neuron | Die ganze rekurrente Schicht braucht nur ≈ 260 Kbit (INT8 + 7-Bit-Index + Zeigertabelle) und passt sogar auf ein einzelnes Board |
| Zeitschritte | **20** | Eine Inferenz dauert ≈ 0,5 ms (Abschnitt 5) |
| Neuronenmodell | Norse `LIFCell`, nur `method="super"` und α = 100 gesetzt, sonst **Norse-Standardwerte**: `tau_mem_inv=100`, `tau_syn_inv=200`, `v_leak=0`, `v_th=1`, `v_reset=0`, `dt=1 ms`, Startzustand v = i = 0 | Feste Konstanten K_MEM = 0,1, K_SYN = 0,2. Surrogate und α wirken nur im Training, in der Inferenz gilt die harte Schwelle `v > 1` |
| Rekurrente Gewichte | `recurrent ⊙ mask ⊙ sign`. `recurrent` ist ein freier Parameter **ohne `abs()`**, initialisiert symmetrisch um 0 (Kaiming) | Die Model Card sagt „nur die Beträge sind gelernt“, der Code erzwingt das aber nicht. Effektive Vorzeichen können gegen das Konnektom kippen. Biologisch fixiert ist also sicher nur die **Topologie**. Hardware: effektive Gewichte offline berechnen und **vorzeichenbehaftet pro Synapse** speichern (INT8). Das Skript misst, wie viele Vorzeichen gekippt sind |
| Konvention | `spk_prev @ W_effᵀ` → `W_eff[post, pre]` | Zeilen = Ziel, Spalten = Quelle. Die CSR-Liste wird nach Spalten (präsynaptisch) sortiert |
| Eingang | Pro Messung Min-Max auf [0, 1]; Projektion von `bernoulli(r) * r` | Der stärkste Kanal ist immer 1, der schwächste immer 0. **Absolute Konzentration geht verloren** (Folgen in 3.2 und 7.2). Eingangsstrom = `W_in[i,k] · r_k` pro Eingangsspike, also Multiplikationen |
| Erster Zeitschritt | Eingang landet erst am Ende eines Schritts im Synapsenstrom | In Schritt 1 kann kein Neuron feuern, effektiv tragen 19 Schritte Information |
| Readout | `spikes_acc / 20` → `Linear(800 → 692)` **mit Bias** | Die Hardware zählt Spikes (5 Bit). Der Faktor 1/20 wird in `W_out` eingerechnet, den Bias addiert B9 |
| Auswertung | 5× Monte-Carlo-Mittel über die Poisson-Kodierung | Auf dem FPGA 5 Durchläufe mit verschiedenen LFSR-Seeds und die Spikezähler summieren, Readout nur einmal (≈ 3 ms gesamt) |
| Daten | DoOR 2.0 (CC BY-SA 4.0), **692 Odorant-Klassen** (laut `config.json`, die Model Card nennt veraltet 500) | Siehe Lizenz-Risiko in Abschnitt 10 |
| Genauigkeit | Test 47,6 %, Validierung 47,7 % (Checkpoint fold 0, seed 1010, Abbruch nach Epoche 28). Zufall wäre 0,14 % | Referenzwert für Phase 1: Die Fixpunkt-Version soll auf demselben Split ≤ 2 Prozentpunkte darunter liegen |

---

## 1. Korrekturen zur ursprünglichen KI-Antwort

Die Roadmap aus der KI-Antwort stimmt in der Richtung (Quantisierung, Partitionierung,
Spike-Kommunikation), unterschätzt aber den Chip und setzt den Engpass an die falsche Stelle:

| Punkt | KI-Antwort | Richtig |
|---|---|---|
| Speicher GW2AR-18 | „~828 Kb BRAM“ | 828 Kbit BSRAM (46 × 18 Kbit) **plus 64 Mbit SDRAM im selben Gehäuse** (GW2AR = mit eingebettetem SDRAM) |
| Engpass | rekurrente 800×800-Matrix | Die **Readout-Schicht** 800×692 (≈ 4,4 Mbit bei INT8) ist mit Abstand der größte Block. Die rekurrente Matrix hat nur 14.998 Synapsen (2,34 %) und braucht ≈ 260 Kbit |
| Rechenleistung | Cluster nötig, weil ein Chip zu klein ist | Rechnerisch reichen 1–3 Boards. Das Netz ist ereignisgetrieben, ein Zeitschritt kostet einige tausend Additionen |
| Partitionierung mit METIS | Kanten zwischen FPGAs minimieren | Mit einem **Spike-Vektor-Broadcast** (Abschnitt 4) bekommt jedes Board ohnehin alle 800 Spikes. Wie viele Kanten Boardgrenzen kreuzen, spielt dann keine Rolle; METIS ist unnötig |
| Eingang | 52 Kanäle direkt einspeisen | Die 52 Eingänge sind die **DoOR-Antworten von Fliegen-Duftrezeptoren** (Or/Ir-Gene). **Kein technischer Gassensor liefert diese Signale.** Eine Adapterschicht muss neu trainiert werden (Abschnitt 3) |
| Ausgang | 500 Klassen | Es sind tatsächlich 692 DoOR-Odorants. Das Netz wurde auf Rezeptorprofile trainiert, nicht auf Sensordaten. Die Readout-Schicht wird für eigene Klassen neu trainiert |

**Konsequenz:** Der Cluster ist sinnvoll, aber aus anderen Gründen als in der KI-Antwort.
Rein technisch läuft das Originalmodell auch auf **einem** Board (Variante E, Abschnitt 8), dann
mit der Readout-Schicht im eingebetteten SDRAM. Der Cluster verteilt den Speicher, sodass alles
im BSRAM bleibt und kein SDRAM-Controller nötig ist.
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
3. Z-Normierung mit gespeicherten Statistiken (entspricht den „standardisierten“ DoOR-Werten)
4. **Intensität** `A = Σ |z_k|` berechnen, *bevor* normiert wird
5. **Gate:** Liegt `A` unter einer Schwelle (saubere Luft), wird nicht klassifiziert, Ergebnis „Luft“. Sonst bläst die Min-Max-Normierung des Modells reines Sensorrauschen auf den vollen Bereich [0, 1] auf und erzeugt zufällige Klassen
6. Min-Max wie `_rate_encode` → `r_k` als `uint8` (0–255, 255 ≙ 1,0). Die Division einmal pro Messung auf der MCU ist billiger als im FPGA
7. Senden als Frame `0xA5 | seq | A[16] | 52×u8 | crc8` an Board 0

Bei eigenem Neutraining (3.3) lohnt es sich, `_rate_encode` durch eine **feste, globale
Normierung** zu ersetzen. Dann trägt `r` die Konzentration wieder mit, und das Gate wird zur
reinen Sicherheitsmaßnahme.

### 3.3 Neu trainieren statt direkt einspeisen

Das Konnektom (Maske + Vorzeichen, idealerweise auch die gelernten Beträge von `W_rec`) ist der
biologische Teil und bleibt **eingefroren**. Neu trainiert werden:

- `W_in` (52 → 800) auf deine Sensormerkmale (oder ein vorgeschalteter 52×52-Adapter, der Sensorkanäle auf „Fliegenrezeptoren“ abbildet, wenn die originale `W_in` erhalten bleiben soll)
- `W_out` (800 → K) für deine eigenen K Klassen (z. B. Luft, Ethanol, Kaffee, Rauch, Lösungsmittel, Unbekannt)

Training mit dem vorhandenen `MaskedRecurrentLIFSNN` (Adam, CrossEntropy, 20 Schritte wie im
Original), ergänzt um **quantisierungsbewusstes Training**: Gewichte im Forward-Pass auf INT8
runden, Membranpotential auf 16 Bit begrenzen. Dann stimmt das FPGA-Ergebnis bitgenau mit dem
Python-Referenzmodell überein.

Gut gelöst ist das als **Zwei-Stufen-Adapter**: Die Sensormerkmale werden auf 52
„Pseudo-Rezeptoren“ abgebildet, das originale `W_in` bleibt dahinter erhalten. Dann wird zuerst
nur der Adapter (+ `W_out`) trainiert. Erst wenn das nicht reicht, wird `W_in` freigegeben.

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

| Board | Aufgabe | Speicher (INT8, 2,34 % Dichte, K = 692) |
|---|---|---|
| **B0** | UART zur MCU, Eingangspuffer `r[52]`, Bernoulli-Encoder (LFSR), Takt- und Zeitschritt-Master, Ring-Master | < 50 Kbit |
| **B1–B8** | je 100 LIF-Neuronen: Eingangsstrom `W_in·(e⊙r)`, rekurrente Synapsen, Membrandynamik, Spike-Zähler, Teilsummen der Readout-Schicht | ≈ 625 Kbit von 828: rekurrent ≈ 40, `w_in` ≈ 41, `w_out` ≈ 541 |
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
           I[i] += W_eff[i,j]                       // vorzeichenbehaftet, INT8
3. Für jeden Eingangskanal k mit Spike e[k]=1 (von B0 mitgesendet):
           I[i] += W_in[i,k] · r[k]   (für alle i dieses Boards, 4 Multiplizierer parallel)
4. Für i = 0..99:                        // LIF-Update in Norse-Reihenfolge (LIFCell)
       v_dec = v[i] + K_MEM·(V_LEAK − v[i] + i_syn[i])   // K_MEM = dt·tau_mem_inv (Std: 0,1)
       i_dec = i_syn[i] − K_SYN·i_syn[i]                 // K_SYN = dt·tau_syn_inv (Std: 0,2)
       z     = (v_dec > V_TH)                            // harte Schwelle in der Inferenz
       v[i]  = z ? V_RESET : v_dec
       i_syn[i] = i_dec + I[i];  I[i] = 0                // Eingang erst NACH dem v-Update
       if z: spike[i]=1; count[i]++
5. Lege eigene 100 Spike-Bits in den Ring-Frame
```

- **Wichtig für Bitgenauigkeit:** Norse rechnet `v` mit dem *alten* Synapsenstrom und addiert den neuen Eingang erst danach auf den Strom. Ein naiver LIF-Kern liegt deshalb um einen Zeitschritt daneben.
- Die Norse-Standardwerte 0,1 und 0,2 (im Code bestätigt) sind keine Zweierpotenzen. Statt das Netz für Shift-Leaks neu zu trainieren, multipliziert der Kern mit Q0.16-Konstanten. Ein zeitmultiplexter Kern braucht dafür 2 der 48 Multiplizierer, so bleiben die vortrainierten Gewichte direkt nutzbar.
- Fixpunkt: Potentiale und Ströme INT16 mit Sättigung, `v_th = 1,0` ≙ 2¹⁰ (Q5.10). Gewichte INT8 mit einer Zweierpotenz-Skala pro Tensor, die das Skript vorschlägt (`2^-frac`). Die Umrechnung nach Q5.10 ist dann ein Shift.
- Vorzeichen: Weil `recurrent` nicht auf ≥ 0 beschränkt ist, wird das effektive Vorzeichen **pro Synapse** gespeichert. Zeigt das Skript bei den echten Gewichten 0 % gekippte Vorzeichen und Dale-Konformität, reicht ein Vorzeichen pro präsynaptischem Neuron, und die Synapsen bekommen eine Bitstufe mehr Auflösung.
- **Poisson-Eingang:** Ein LFSR pro Kanal auf B0 setzt pro Zeitschritt `e[k] = (rand9 < r_k + r_k/128)` (9-Bit-Vergleich, damit `r = 255` sicher immer feuert). Für die 5 Monte-Carlo-Durchläufe startet das LFSR mit verschiedenen Seeds. `r[52]` geht einmal pro Inferenz per Steuerframe an alle Boards.
- **Verifikation:** `torch.bernoulli` lässt sich in Hardware nicht bitgleich nachbauen. Für den Bitvergleich gibt es deshalb einen Testmodus, in dem B0 eine aus Python exportierte Folge `e_t` abspielt. Mit LFSR wird statistisch verglichen (Genauigkeit über viele Proben).
- Speicher in BSRAM:
  - `ptr[800]`: Startadresse der Synapsenliste je präsynaptischem Neuron
  - `syn[]`: {Ziel-Index 7 Bit, Gewicht 8 Bit}
  - `w_in[52][100]`
  - `w_out[100][K]`
  - Zustände `v`, `I`, `count`

### 4.4 Readout verteilt berechnen

Am Ende eines Inferenzfensters (T = 20 Zeitschritte) berechnet jedes Board die Teilsumme über
seine eigenen Neuronen:

```
logit_partial[c] = Σ_i count[i] · W_out'[i,c]      (100·K MACs, Multiplizierer)
W_out' = W_out / (20·M)  (Mittel über 20 Schritte und M = 5 Monte-Carlo-Läufe, offline eingerechnet; Bias addiert B9)
```

Die Teilsummen laufen über den Ring nach B9 und werden dort addiert. Damit braucht kein Board
die ganze 800×K-Matrix. Bei K = 692 sind das ≈ 541 Kbit pro Board (≈ 31 der 46 BSRAM-Blöcke).
Das passt, lässt aber wenig Reserve.
Bei K ≈ 10 eigenen Klassen ist die Readout-Schicht vernachlässigbar.

Laut Model Card liest das Modell auf zeitgemittelten Spikeraten aus, genau passend zu den
Zählern. `count` braucht 7 Bit (0–100 bei 5 Monte-Carlo-Durchläufen; weil der Readout linear ist, ist die Summe der Zähler exakt gleichwertig zum Mittel der Logits, mit `W_out' = W_out / 100`). Die 100·692 MACs laufen mit 4 Multiplizierern bei
54 MHz in ≈ 0,3 ms. Die 692 Teilsummen à 24 Bit brauchen einen Ringumlauf (≈ 4.150 Takte, 0,17 ms
bei 25 MHz), jedes Board addiert seinen Anteil im Durchlauf. Bei eigenen 10 Klassen ist beides
vernachlässigbar.

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
- **Steuerframes** (`CMD`): Reset, Fenster-Start mit `r[52]` (52 × 8 Bit) und Seed, Fenster-Ende, Readout-Teilsummen sammeln (dann trägt der Frame K × 24 Bit), Gewichte nachladen (optional), Status/Heartbeat

### Zeitbudget (Schätzung)
| Schritt | Takte | bei 54 MHz Kerntakt |
|---|---|---|
| Ring-Frame | ≈ 250 | 5 µs |
| Eingang: ≈ 26 aktive Kanäle × 100 Neuronen, 4 MAC parallel | ≈ 650 | 12 µs |
| Synapsen: 80 Spikes × ≈ 2,3 lokale Ziele (Fan-out 18,7 / 8 Boards) | ≈ 190 | 4 µs |
| LIF-Update 100 Neuronen | 100 | 2 µs |
| **Summe pro Zeitschritt** | | **≈ 25 µs → ~40.000 Schritte/s** |

Eine Inferenz mit T = 20 Schritten dauert etwa 0,5 ms, die 5-fache Monte-Carlo-Mittelung wie
im Original etwa 2,5 ms plus Readout (≈ 0,5 ms). Die MOX-Sensoren reagieren im Bereich von Sekunden.
**Echtzeit ist kein Problem.** Die Reserve reicht auch für mehr Monte-Carlo-Durchläufe, wenn
das die Konfidenz stabilisiert.

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
  - publiziert `/olfaction/odor` (Klasse + Konfidenz), `/olfaction/intensity` (Intensität `A` von der MCU. Die Spikerate des Netzes taugt wegen der Min-Max-Normierung **nicht** als Konzentrationsmaß), `/olfaction/raw` (Debug)
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
| D – Größeres Konnektom | alle 10 Boards als Neuronen-Boards, bis ~3.000 Neuronen mit SDRAM | Wenn später ein größerer FlyWire-Ausschnitt (z. B. alle ≈ 2.000 Kenyonzellen) genutzt werden soll |
| E – Ein Board | alle 800 Neuronen auf einem Board (≈ 560 Kbit BSRAM für rekurrent + `w_in`), `w_out` 4,4 Mbit im eingebetteten SDRAM | Schnellster Weg zum ersten lauffähigen Gesamtmodell in Phase 2. Ein Zeitschritt dauert dann ≈ 8× länger (≈ 0,2 ms), immer noch weit schneller als die Sensoren |

Phase 2 mit **E** (ein Board, volles Modell) beginnen, dann auf **A** verteilen. Ist A lauffähig, ist der Wechsel zu **B** reine Konfigurationssache, denn
Kern und Ring sind gleich.

---

## 9. Umsetzungsplan

| Phase | Inhalt | Ergebnis / Abnahme |
|---|---|---|
| **0 – Analyse** (wenige Tage) | `inspect_model.py` auf die echten Dateien, `connectome_meta.json` lesen, Lizenz prüfen (Modellcode ist bereits ausgewertet) | Speicherbudget, Gewichtsbereiche und Vorzeichen-Statistik bestätigt |
| **1 – Referenz** (2 Wochen) | Fixpunkt-Referenzmodell in numpy, Vergleich mit dem Norse-Float-Modell auf DoOR (gleiche 5-fold-Splits, 5× Monte-Carlo) | Genauigkeitsverlust durch INT8/INT16 < 2 Prozentpunkte |
| **2 – Ein Board** (3 Wochen) | `lif_core` + `syn_engine` für 100 Neuronen, Verilator-Vergleich, dann auf ein Tang Nano 20K; Spikes per UART auslesen | Bitgenaue Spikes gegenüber Referenz |
| **3 – Ring** (3 Wochen) | `ring_node`, Backplane-Prototyp mit 3 Boards, dann 10; Fehlerzähler (CRC), Langzeittest | > 10⁹ Frames ohne CRC-Fehler; volle 800-Neuronen-Inferenz bitgenau |
| **4 – Sensorik** (parallel, 3 Wochen) | Sensorkopf, MCU-Firmware (Wio Terminal aus diesem Repo erweitern), Datensatz aufnehmen | Datensatz ≥ 6 Klassen, mehrere Tage |
| **5 – Training** (2 Wochen) | `W_in`/`W_out` mit QAT neu trainieren, Konnektom fix | Offline-Genauigkeit auf Testtagen; Export → FPGA |
| **6 – Roboter** (3 Wochen) | Strom/Mechanik, ROS-2-Bridge, Klassifikation im Stand und beim Laufen, dann Cast-and-Surge | Lite3 findet eine Ethanolquelle im Raum |

### Erster Schritt (heute)
```bash
pip install huggingface_hub numpy
huggingface-cli download Vick-MIRE/flywire-olfactory-snn --local-dir flywire-snn
python3 snn-fpga/tools/inspect_model.py flywire-snn/model.safetensors \
        flywire-snn/connectome_mask.npz --boards 8 --wbits 8
cat flywire-snn/config.json flywire-snn/connectome_meta.json
```
Das Skript gibt Tensorformen, INT8-Fehler mit Zweierpotenz-Skala, die effektiven rekurrenten
Gewichte (`recurrent·mask·sign`) samt Anteil gegen das Konnektom gekippter Vorzeichen, Dichte
und Fan-in/-out, Dale-Konformität und den BSRAM-Bedarf pro Board aus. Damit lassen sich die Schätzungen oben bestätigen oder
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
| Biologische Vorzeichen im Originalmodell nicht erzwungen | Beim eigenen Neutraining `recurrent` durch `abs(recurrent)` oder `softplus(recurrent)` ersetzen. Dann gilt das Konnektom-Vorzeichen wirklich (bei Dale-konformer Maske auch Dale), das Konnektom wirkt stärker als Prior, und die Hardware spart ein Bit pro Synapse |
| Min-Max-Normierung verwirft die Konzentration und verstärkt Rauschen in sauberer Luft | Intensität und Gate auf der MCU (3.2); bei Neutraining feste Normierung |
| Abweichung Norse-Float ↔ Fixpunkt | Phase 1: bitgenaue Referenz vor jeder HDL-Zeile, Testmodus mit vorgegebenen Eingangsspikes |
| Lizenz | Die Model Card nennt nur die Datenlizenzen: DoOR ist **CC BY-SA 4.0** (Share-Alike; ob das auf trainierte Gewichte durchschlägt, ist rechtlich offen). Für FlyWire gelten eigene Nutzungsbedingungen. Für den Eigengebrauch unkritisch; vor einer Veröffentlichung abgeleiteter Gewichte die Lizenz des Modell-Repos klären und Münch & Galizia (2016) sowie Dorkenwald et al. (2024) zitieren |
| Domänenlücke DoOR → Gassensoren | Die Fliegenrezeptoren sind hochselektiv, MOX-Sensoren breitbandig. Der Adapter muss diese Lücke schließen. Ehrliche Erwartung: Das Konnektom wirkt als strukturierter Merkmalsraum, nicht als fertiger „Fliegengeruchssinn“. Die Kontrolle mit einem Zufallsnetz (erste Zeile) zeigt den tatsächlichen Nutzen |
