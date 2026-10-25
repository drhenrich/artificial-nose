# Variante E: das ganze Modell auf einem Tang Nano 20K

Das veröffentlichte Modell (800 Neuronen, 14.998 Synapsen, 692 Klassen) passt vollständig auf
**ein** GW2AR-18. Rekurrente Schicht und Eingangsgewichte liegen im BSRAM, die Ausgangsschicht
(541 KiB) im eingebetteten 64-Mbit-SDRAM.

**Stand:** Verilog-Kern fertig und bitgenau gegen `snn_ref.py` verifiziert (Verilator, 20 Proben,
2.000 Zeitschritte, alle Spikes und 13.840 Logits identisch). Bring-up-Bitstream für den Tang Nano
20K gebaut (Open-Source-Toolchain, Timing erfüllt). Readout über SDRAM in der Simulation
verifiziert, SDRAM-Controller fehlt noch: Beim Bring-up rechnet der PC den Readout.

| Datei | Inhalt |
|---|---|
| [`rtl/snn_core.v`](rtl/snn_core.v) | SNN-Kern (Eingangsspikes, Neuronenschleife, LIF, Readout bzw. Zählerausgabe) |
| [`rtl/uart.v`](rtl/uart.v), [`rtl/top_tangnano20k.v`](rtl/top_tangnano20k.v), [`rtl/tangnano20k.cst`](rtl/tangnano20k.cst) | Bring-up-Top mit UART und LEDs |
| [`rtl/tb/tb_snn_core.v`](rtl/tb/tb_snn_core.v), [`rtl/tb/tb_top.v`](rtl/tb/tb_top.v) | Testbenches (Kern mit SDRAM-Modell; Top über UART) |
| [`rtl/sim.sh`](rtl/sim.sh), [`rtl/build.sh`](rtl/build.sh) | Simulation, Bitstream |
| [`tools/export_variant_e.py`](tools/export_variant_e.py) | Speicherabbilder, `params.vh`, Testvektoren |
| [`tools/board_test.py`](tools/board_test.py) | Test auf dem Board über USB-UART |

Die Rechenvorschrift ist dieselbe wie in allen Varianten: `fixed_forward()` in
[`tools/snn_ref.py`](tools/snn_ref.py).

---

## 1. Zahlenformate (echtes Modell)

| Größe | Format |
|---|---|
| `w_in`, `w_rec` | **9 Bit** signed, Skala 2⁻⁵ bzw. 2⁻⁶ |
| `w_out` | 8 Bit signed, Skala 2⁻⁵ |
| `v`, `i_syn` | int16, Q8.7 (`V_TH` = 128) |
| Shifts | `SH_IN = 6`, `SH_REC = −1` (Links-Shift) |
| Bitbreiten | `acc_in` 20 Bit (Register 24), `acc_rec` 12 Bit (20), Strom 14 Bit, Readout-Akkumulator 18 Bit (32) |

9 statt 8 Bit: Die Übereinstimmung mit dem Float-Modell bei gleichen Eingangsspikes steigt von 92
auf 96 % (Argmax). Synapsen gehen dadurch kaum mehr verloren: 4.773 der 14.998 Gewichte sind
schon im trainierten Float-Modell praktisch 0 (|w| < 10⁻³⁰). Nach der Quantisierung bleiben
9.731 Synapsen.

## 2. Speicherplan (gemessen: 37 BSRAM-Blöcke im Bring-up, 38 mit Readout)

| Speicher | Inhalt | Format | Blöcke |
|---|---|---|---|
| `syn` | 9.731 Synapsen, sortiert nach Ziel-Neuron *i* und Hälfte *h* der Quellneuronen | 18 Bit `{pre_lokal[17:9], w[8:0]}`, Quelle = *h*·400 + `pre_lokal` | 10 |
| `ptr` | 1.601 Listenanfänge, Index 2*i*+*h* | 14 Bit | 2 |
| `w_in` | 800 × 52 Gewichte | 18 Bit = 2 × 9 Bit, 26 Wörter pro Neuron | 21 |
| `v`, `i_syn` | Zustände | je 16 Bit | 2 |
| `counts` | Spikezähler, 4 pro Wort | 4 × 7 Bit (0–100) | 1 |
| `b_out` | Bias | 32 Bit | 2 |
| **Summe** | | | **38 von 46** |
| SDRAM | `w_out` klassenweise, 200 Wörter à 4 Gewichte pro Klasse | 4 × int8 pro 32-Bit-Wort | 541 KiB von 8 MiB |

Warum die Aufteilung nach Hälften: 10 Bit Neuronenindex + 9 Bit Gewicht wären 19 Bit und passen
nicht ins 18-Bit-BSRAM-Wort. Mit getrennten Listen für Quellen 0–399 und 400–799 reichen 9 Bit
Index, und jedes Wort ist voll genutzt.

## 3. Ablauf (sammelnd)

```
pro Inferenz:  N_RUNS = 5 Läufe × T = 20 Schritte, dann Readout

S_GEN     52 Takte    e[k] = xorshift32()[31:24] < r[k]      (Seed pro Lauf: seed + m·0x9E3779B9)
pro Neuron i (≈ 38 Takte):
  S_N0..3             ptr[2i], ptr[2i+1], ptr[2i+2] lesen
  S_MAC   parallel:   acc_in  += e[k]·r[k]·w_in[i,k]          2 Gewichte/Takt, 26 Takte
                      acc_rec += s_prev[pre] ? w : 0          1 Synapse/Takt, Fan-in von i
  S_LIF1              i_new = sat16(i_syn + rnd(acc_in, SH_IN) + rnd(acc_rec, SH_REC))
  S_LIF2              v_dec = sat16(v + mulq16(K_MEM, i_new − v));  i_syn = i_new − mulq16(K_SYN, i_new)
                      z = v_dec > V_TH;  v = z ? 0 : v_dec;  counts[i] += z;  s_cur[i] = z
S_STEP              s_prev ← s_cur
```

Der Kern **sammelt** die Eingänge pro Ziel-Neuron (gather), statt Spikes zu verteilen
(scatter). Das braucht keinen `acc_rec`-Speicher und keine Read-Modify-Write-Konflikte, und die
Laufzeit hängt nicht von der Feuerrate ab. Im ersten Schritt eines Laufs werden `v` und `i_syn`
als 0 gelesen, im ersten Schritt der Inferenz die Zähler, sodass keine Löschschleife nötig ist.

## 4. Readout

```
für c = 0..691:   Burst von 200 Wörtern aus dem SDRAM (wout_req / wout_addr, Daten mit wout_valid)
    acc = b_out[c] + Σ_j Σ_n counts[4j+n] · w_out[c, 4j+n]      4 MAC/Takt (Zählerwort j passt zu w_out-Wort j)
    Argmax laufend (erstes Maximum wie numpy)
```

Mit Parameter `READOUT = 0` (Bring-up) gibt der Kern stattdessen die 200 Zählerwörter aus.

## 5. Zeit (gemessen in der Simulation)

| | Takte | bei 27 MHz | bei 54 MHz |
|---|---|---|---|
| Inferenz (5 Läufe × 20 Schritte × 800 Neuronen) | ≈ 2.890.000 | 107 ms | 54 ms |
| Readout (692 × ≈ 210) | ≈ 146.000 | 5 ms | 3 ms |
| **Klassifikation** | **≈ 3.036.000** | **112 ms (≈ 9/s)** | **56 ms (≈ 18/s)** |

Die Gassensoren liefern 1–10 Messungen pro Sekunde. Der Bring-up läuft mit 27 MHz direkt vom
Quarz, nextpnr meldet maximal 58 MHz. Beschleunigungen für später:
- Listenanfänge des nächsten Neurons vorab lesen (spart 4 Takte pro Neuron)
- `w_in` mit beiden BSRAM-Ports lesen (4 Gewichte/Takt, 13 statt 26 Takte)
- 54 MHz über die PLL

Zusammen ergibt das etwa 20 ms pro Klassifikation.

## 6. Ressourcen (nextpnr, Bring-up-Top mit UART)

| Ressource | belegt | verfügbar |
|---|---|---|
| LUT4 | 4.542 (21 %) | 20.736 |
| Flipflops | 2.539 (16 %) | 15.552 |
| ALU | 612 (3 %) | 15.552 |
| BSRAM | 37 (80 %) | 46 |
| Fmax | 58 MHz | (27 MHz genutzt) |

## 7. Inbetriebnahme auf dem Tang Nano 20K

**Bitstream bauen** (oder den fertigen `snn_tangnano20k.fs` verwenden):
```bash
python3 snn-fpga/tools/export_variant_e.py model.safetensors connectome_mask.npz --out build/e
snn-fpga/rtl/build.sh build/e          # yosys + nextpnr-himbaechel + gowin_pack (OSS CAD Suite)
```

**Laden und testen:**
```bash
openFPGALoader -b tangnano20k build/e/snn_tangnano20k.fs     # ins SRAM (weg nach Stromausfall)
pip install pyserial numpy
python3 snn-fpga/tools/board_test.py model.safetensors connectome_mask.npz --port /dev/ttyUSB1 --n 10
```

Erwartet: `10 von 10 Proben bitgenau wie snn_ref.py`, etwa 180 ms pro Probe. Davon entfallen
112 ms aufs Rechnen und 70 ms auf die UART mit 115.200 Baud.

**LEDs** (aktiv low): LED0 leuchtet beim Rechnen, LED1 blinkt als Herzschlag (zeigt, dass der Bitstream
läuft), LED2 wechselt bei jedem empfangenen Frame, LED3 bei jeder gesendeten Antwort.

**Wenn nichts zurückkommt:** Blinkt LED1 nicht, ist der Bitstream nicht geladen. Sonst den
richtigen Port prüfen: unter Linux meist `ttyUSB1`, weil `ttyUSB0` der JTAG ist. Unter Windows
im Gerätemanager nachsehen. Die Pinbelegung (Takt 4, UART TX 69 / RX 70, LEDs 15–20) folgt den
Sipeed-Beispielen; bei abweichender Board-Revision `rtl/tangnano20k.cst` anpassen.

## 8. Nächste Schritte

1. Bring-up auf dem Board bestätigen (`board_test.py`)
2. SDRAM-Controller für das eingebettete SDRAM (GW2AR: 64 Mbit, 32 Bit breit) und Laden von `w_out` per UART oder aus dem SPI-Flash. Dann läuft der Readout auf dem FPGA (`READOUT = 1`, in der Simulation bereits verifiziert)
3. Geschwindigkeit (Abschnitt 5) und UART-Protokoll für die Sensor-MCU

## 9. Vom einen Board zum Cluster: Variante F („E-Cluster“)

Wenn ein Board das ganze Modell trägt, nutzt man die übrigen neun am einfachsten als
**unabhängige Kopien**:

- **Parallele Monte-Carlo-Läufe:** 5 Boards rechnen je einen Lauf mit eigenem Seed. Die Zähler werden auf dem Host addiert, das ist exakt gleichwertig, weil der Readout linear ist. Die Inferenzzeit sinkt auf ein Fünftel
- **Stereo:** je ein Board pro Sensorkopf links und rechts
- **Redundanz:** 3 Boards mit gleichem Seed, Mehrheitsentscheid

Verbindung: ein gemeinsamer Bus vom Host (SPI oder je ein UART) zu den Boards. Ring,
Taktsynchronisation und unterschiedliche Speicherabbilder pro Board entfallen.

**Empfehlung:** Für das veröffentlichte Modell ist **E bzw. F** die bessere Zielarchitektur als
der Ring aus Variante A. A lohnt sich erst, wenn das Netz nicht mehr auf einen Chip passt,
z. B. mit einem größeren FlyWire-Ausschnitt (Variante D).
