# Variante E: das ganze Modell auf einem Tang Nano 20K

Das veröffentlichte Modell (800 Neuronen, 14.998 Synapsen, 692 Klassen) passt vollständig auf
**ein** GW2AR-18. Rekurrente Schicht und Eingangsgewichte liegen im BSRAM, die Ausgangsschicht
(541 KiB) im eingebetteten 64-Mbit-SDRAM.

## Stand

| Version | Bitstream | Status |
|---|---|---|
| Bring-up (27 MHz, Zählerausgabe, Readout auf dem PC) | `snn_bringup.fs` (früher `snn_tangnano20k.fs`) | **Auf dem Board bestätigt (25.10.2026):** 10/10 Proben, alle 8.000 Spikezähler bitgenau, 187 ms pro Probe |
| **Vollversion** (54 MHz, SDRAM-Readout und Argmax auf dem FPGA, 1 Mbaud) | `snn_full.fs`, Ausweichvariante `snn_full_115200.fs` | **Auf dem Board bestätigt (25.10.2026, v3):** 10/10 Proben, alle 6.920 Logits und Klassen bitgenau, 66 ms pro Probe inkl. UART-Antwort, Laden von `w_out` 6,0 s. SDRAM-Abtastung nach 2,5 Takten (fallende Flanke) |

| Datei | Inhalt |
|---|---|
| [`rtl/snn_core.v`](rtl/snn_core.v) | SNN-Kern (Eingangsspikes, Neuronenschleife, LIF, Readout bzw. Zählerausgabe) |
| [`rtl/sdram_ctrl.v`](rtl/sdram_ctrl.v) | Controller für das eingebettete SDRAM (Init, Refresh, Lese-Bursts, Schreiben) |
| [`rtl/top_tangnano20k_full.v`](rtl/top_tangnano20k_full.v) | Vollversion: PLL 54 MHz, UART-Befehle W/R/C, SDRAM, Kern |
| [`rtl/top_tangnano20k.v`](rtl/top_tangnano20k.v) | Bring-up-Top (Zählerausgabe) |
| [`rtl/uart.v`](rtl/uart.v), [`rtl/tangnano20k.cst`](rtl/tangnano20k.cst) | UART, Pinbelegung |
| [`rtl/tb/`](rtl/tb) | Testbenches: Kern, SDRAM-Controller mit SDRAM-Modell, Bring-up-Top, Vollversion |
| [`rtl/sim.sh`](rtl/sim.sh), [`rtl/build.sh`](rtl/build.sh) | Simulation, Bitstream (`full` oder `bringup`, Baudrate per `BAUD=`) |
| [`tools/export_variant_e.py`](tools/export_variant_e.py) | Speicherabbilder, `params.vh`, Testvektoren |
| [`tools/board_test.py`](tools/board_test.py) | Test auf dem Board (`--mode full` oder `counts`) |

Die Rechenvorschrift ist dieselbe wie in allen Varianten: `fixed_forward()` in
[`tools/snn_ref.py`](tools/snn_ref.py).

---

## 1. Zahlenformate (veröffentlichtes Modell)

| Größe | Format |
|---|---|
| `w_in`, `w_rec` | 9 Bit signed, Skala 2⁻⁵ bzw. 2⁻⁶ |
| `w_out` | 8 Bit signed, Skala 2⁻⁵ |
| `v`, `i_syn` | int16, Q8.7 (`V_TH` = 128) |
| Shifts | `SH_IN = 6`, `SH_REC = −1` (Links-Shift) |
| Bitbreiten | `acc_in` 20 Bit (Register 24), `acc_rec` 12 Bit (20), Strom 14 Bit, Readout-Akkumulator 18 Bit (32) |

9 statt 8 Bit: Die Übereinstimmung mit dem Float-Modell bei gleichen Eingangsspikes steigt von 92
auf 96 % (Argmax). Synapsen gehen dadurch kaum mehr verloren: 4.773 der 14.998 Gewichte sind
schon im trainierten Float-Modell praktisch 0 (|w| < 10⁻³⁰). Nach der Quantisierung bleiben
9.731 Synapsen.

## 2. Speicherplan (Vollversion: 42 BSRAM-Blöcke laut nextpnr)

| Speicher | Inhalt | Format | Blöcke |
|---|---|---|---|
| `syn0`, `syn1` | 4.120 + 5.611 Synapsen, getrennt nach Quellhälfte (0–399 / 400–799), sortiert nach Ziel-Neuron | 18 Bit `{pre_lokal[17:9], w[8:0]}` | 11 |
| `pp` | 801 Listenanfänge `{Start syn1, Start syn0}` | 28 Bit | 2 |
| `win0`, `win1` | 800 × 52 Gewichte, Takt *c*: `win0` = Gewichte 4*c*, 4*c*+1; `win1` = 4*c*+2, 4*c*+3 | 18 Bit = 2 × 9 Bit | 22 |
| `v`, `i_syn` | Zustände | je 16 Bit | 2 |
| `counts` | Spikezähler, 4 pro Wort | 4 × 7 Bit (0–100) | 1 |
| `b_out` | Bias | 32 Bit | 2 |
| Kern gesamt | | | **40** |
| `lbuf` (Top) | Logits für die UART-Antwort | 32 Bit | 2 |
| SDRAM | `w_out` klassenweise, 200 Wörter à 4 Gewichte pro Klasse | 4 × int8 pro 32-Bit-Wort | 541 KiB von 8 MiB |

10 Bit Neuronenindex + 9 Bit Gewicht passen nicht ins 18-Bit-Wort. Mit getrennten Speichern pro
Quellhälfte reichen 9 Bit Index, und beide Hälften laufen parallel.

## 3. Ablauf (sammelnd, Pipeline)

```
pro Inferenz:  N_RUNS = 5 Läufe × T = 20 Schritte, dann Readout

S_GEN     52 Takte   e[k] = xorshift32()[31:24] < r[k]       (Seed pro Lauf: seed + m·0x9E3779B9)
pro Neuron i: max(13, Fan-in Hälfte 0, Fan-in Hälfte 1) + 5 Takte, im Mittel ≈ 19,5
  S_MAC   parallel, 2-stufige Pipeline (Produkte | Akkumulieren):
            acc_in  += e[k]·r[k]·w_in[i,k]       4 Gewichte/Takt aus win0 + win1, 13 Takte
            acc_rec += s_prev[pre]·w             je 1 Synapse/Takt aus syn0 und syn1
          währenddessen: Listenende des nächsten Neurons aus pp lesen
  S_LIF1  i_new = sat16(i_syn + rnd(acc_in, SH_IN) + rnd(acc_rec, SH_REC))
  S_LIF2  v_dec = sat16(v + mulq16(K_MEM, i_new − v));  i_syn = i_new − mulq16(K_SYN, i_new)
          z = v_dec > V_TH;  v = z ? 0 : v_dec;  counts[i] += z;  s_cur[i] = z
          -> nächstes Neuron startet sofort
S_STEP    s_prev ← s_cur
```

Der Kern **sammelt** die Eingänge pro Ziel-Neuron (gather). Es gibt keinen `acc_rec`-Speicher
und keine Read-Modify-Write-Konflikte. Im ersten Schritt eines Laufs werden `v` und `i_syn` als
0 gelesen, im ersten Schritt der Inferenz die Zähler, sodass keine Löschschleife nötig ist.

## 4. Readout und SDRAM

```
für c = 0..N_CLS-1:  Lese-Burst von 200 Wörtern ab c·200 (wout_req / wout_addr, Daten mit wout_valid)
    acc = b_out[c] + Σ_j Σ_n counts[4j+n] · w_out[c, 4j+n]     4 MAC/Takt, 2-stufige Pipeline
    Argmax laufend (erstes Maximum wie numpy)
```

`sdram_ctrl.v`:
- Wortadresse `{bank[1:0], row[10:0], col[7:0]}`, CL 2, Burstlänge 1
- ein READ pro Takt, über Zeilengrenzen mit PRECHARGE/ACTIVATE
- Schreiben mit Auto-Precharge, Refresh alle 600 Takte (Vorgabe 843)
- SDRAM-Takt = invertierter Systemtakt, Lesedaten `rd_lat` Takte nach dem READ übernommen

In der Simulation prüft ein SDRAM-Modell tRCD, tRP, tRFC, offene Zeilen und den Refresh-Abstand
(gemessen höchstens 779 Takte).

**Leselatenz auf echter Hardware:** Beim ersten Board-Test waren Einzelzugriffe korrekt, die Bursts
des Readouts aber falsch. Einzelzugriffe verbergen einen Abtastfehler, weil der Datenbus den Wert
noch hält. Der zweite Test (Latenz 2…7, nur steigende Flanke) ergab:
- Abtastung nach 2 Takten: 27 % der Burst-Wörter richtig
- Abtastung nach 3 und mehr Takten: praktisch keine

Der Abtastzeitpunkt lag also auf dem Datenwechsel. Das gültige Fenster liegt etwa einen halben
Takt früher als im Modell angenommen.

Deshalb gibt es jetzt:
- **Einstellbarkeit zur Laufzeit:** Latenz 2…7 plus Abtastung auf der **fallenden Flanke**, also einen halben Takt früher (Befehl `L`)
- **Kalibrierung:** `board_test.py` und `nose_bridge.py` prüfen nach dem Laden alle Zeitpunkte von 1,5 bis 7,0 Takten in halben Takten mit echten Bursts (Befehl `B`) und nehmen die Mitte des längsten fehlerfreien Bereichs. Gemessen (v3):

| Abtastung nach | 1,5 | 2,0 | **2,5 (fallend)** | 3,0 … 7,0 |
|---|---|---|---|---|
| falsche Burst-Wörter von 200 | 200 | 161 | **0** | 198 |

Das fehlerfreie Fenster ist nur einen halben Takt breit, nach links und rechts gibt es also wenig
Reserve. Die Kalibrierung beim Start fängt Temperatur- und Exemplarstreuung ab. Ab dem nächsten
Build ist 2,5 Takte der Startwert. Mehr Reserve brächte eine feinere Phasenverschiebung des
SDRAM-Takts über den zweiten PLL-Ausgang

Diagnosebefehle der Vollversion:

| Befehl | Antwort | Zweck |
|---|---|---|
| `0xA5 'L' code` | `0x5A 'L' code` | SDRAM-Abtastung: Bits 2:0 Latenz 2..7, Bit 3 fallende Flanke (½ Takt früher) |
| `0xA5 'B' addr[3] n[2]` | `0x5A 'B' daten[4n]` | n ≤ 256 Wörter als *ein* Burst lesen, wie beim Readout |
| `0xA5 'Z'` | `0x5A 'Z' zähler[800] summe` | Spikezähler der letzten Klassifikation, trennt Kern- von Readout-Fehlern |

## 5. Zeit (Simulation)

| | Takte | bei 54 MHz |
|---|---|---|
| Inferenz (5 Läufe × 20 Schritte × 800 Neuronen) | ≈ 1.570.000 | 29 ms |
| Readout (692 Klassen, SDRAM) | ≈ 147.000 | 2,7 ms |
| **Klassifikation** | **1.717.700** | **31,8 ms (≈ 31/s)** |
| zum Vergleich Bring-up (27 MHz, alter Kern) | 3.036.000 | 112 ms |

Dazu kommt die UART-Antwort mit allen 692 Logits (2.780 Bytes): 28 ms bei 1 Mbaud bzw. 241 ms
bei 115.200 Baud. Für den Betrieb reichen Klasse und Logit-Maximum. Mit einem eigenen
Modell (z. B. 5 Klassen) entfällt das praktisch.

## 6. Ressourcen (nextpnr, Vollversion)

| Ressource | belegt | verfügbar |
|---|---|---|
| LUT4 | 7.528 (36 %) | 20.736 |
| Flipflops | 3.319 (21 %) | 15.552 |
| BSRAM | 42 (91 %) | 46 |
| Multiplizierer | 6 × 18×18, 4 × 9×9 | 48 / 96 |
| rPLL | 1 | 2 |
| Fmax | 61,8 MHz (1 Mbaud), 58,4 MHz (115.200 Baud) | 54 MHz genutzt |

## 7. Inbetriebnahme

**Bauen** (OSS CAD Suite) oder die fertigen Bitstreams verwenden:
```bash
python3 snn-fpga/tools/export_variant_e.py flywire-snn/model.safetensors flywire-snn/connectome_mask.npz --out build/e
snn-fpga/rtl/build.sh build/e full                 # -> build/e/snn_full.fs (1 Mbaud)
BAUD=115200 snn-fpga/rtl/build.sh build/e full     # -> build/e/snn_full_115200.fs
snn-fpga/rtl/build.sh build/e bringup              # -> build/e/snn_bringup.fs
```

**Vollversion testen:**
```bash
openFPGALoader -b tangnano20k snn_full.fs
python3 snn-fpga/tools/board_test.py flywire-snn/model.safetensors flywire-snn/connectome_mask.npz \
        --port /dev/cu.usbserial-XXXX1 --mode full --n 10
```
Ablauf:
1. Das Skript lädt die 138.400 Wörter von `w_out` ins SDRAM, etwa 6 s bei 1 Mbaud, mit Prüfsumme je 1.024 Wörter.
2. Es liest 8 × 64 Wörter einzeln zurück.
3. Es kalibriert die SDRAM-Leselatenz mit Bursts.
4. Danach vergleicht es Klasse und alle 692 Logits jeder Probe bitgenau. Bei Abweichung prüft es zusätzlich die Spikezähler.

Das SDRAM behält die Gewichte, bis der Strom weg ist. Danach `--skip-load` weglassen.

**Wenn es nicht klappt:**

| Symptom | Ursache / Abhilfe |
|---|---|
| Keine Antwort, LED1 blinkt | Baudrate: BL616 schafft evtl. kein 1 Mbaud → `snn_full_115200.fs` laden und `--baud 115200` |
| LED2 bleibt aus | SDRAM nicht initialisiert (PLL/Takt prüfen) |
| Prüfsumme beim Laden falsch | UART-Übertragung gestört → niedrigere Baudrate |
| Rücklesefehler im SDRAM oder keine Latenz liefert korrekte Bursts | Lese-Timing außerhalb 2..7 oder Takt/Phase: bitte melden |
| Logits falsch | `board_test.py` meldet zusätzlich, ob die Spikezähler stimmen (`Kern ok, Readout/SDRAM falsch` oder `Kern rechnet anders`) |

**LEDs** (aktiv low):

| LED | Bring-up | Vollversion |
|---|---|---|
| LED0 | rechnet | rechnet |
| LED1 | Herzschlag | Herzschlag |
| LED2 | wechselt je empfangenem Frame | SDRAM bereit |
| LED3 | wechselt je gesendeter Antwort | wechselt je Befehl |

## 8. Nächste Schritte

1. ~~Bring-up auf dem Board bestätigen~~ erledigt: 10/10 Proben bitgenau
2. ~~Vollversion auf dem Board testen~~ erledigt: 10/10 Proben bitgenau, 66 ms
3. Eigene Gerüche: Aufnahme, Neutraining, Live-Betrieb, siehe [SENSORIK.md](SENSORIK.md)
4. `w_out` beim Einschalten aus dem SPI-Flash statt per UART laden

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
