# Variante E: das ganze Modell auf einem Tang Nano 20K

Das veröffentlichte Modell (800 Neuronen, 14.998 Synapsen, 692 Klassen) passt vollständig auf
**ein** GW2AR-18. Rekurrente Schicht und Eingangsgewichte liegen im BSRAM, die Ausgangsschicht
(541 KiB) im eingebetteten 64-Mbit-SDRAM. Damit läuft das Gesamtmodell ohne Ring und ohne
Synchronisation zwischen Boards.

Zahlen aus dem echten Modell (`export_variant_e.py` auf `model.safetensors`): Zustandsformat
Q8.7, Shifts `SH_IN = 5`, `SH_REC = −2`, Bitbreiten `acc_in` 19, `acc_rec` 11, Strom 14,
Readout-Akkumulator 18 Bit.

Die Rechenvorschrift ist dieselbe wie in allen Varianten: `fixed_forward()` in
[`tools/snn_ref.py`](tools/snn_ref.py), Formate in [KONZEPT.md, Abschnitt 4.5](KONZEPT.md).
Variante E legt nur fest, *wie* ein einzelner Chip das abarbeitet.

---

## 1. Blockschaltbild

```
           UART (3 Mbit/s)                                   SDRAM 64 Mbit (im Gehäuse)
 MCU ◄──────────────────► uart_if ──► ctrl_fsm               ┌──────────────────────┐
  r[52] u8, Seed, Start     │            │                    │ w_out  692 × 800 i8  │
  ◄── Klasse, Logits        │            ▼                    │ b_out  692 × i32     │
                            │   ┌──── step_engine ────┐       └──────────▲───────────┘
                            │   │ A rec_scatter (CSR)  │                  │ 32 Bit Burst
                            │   │ B neuron_loop        │        readout_engine
                            │   │   in_mac ×4 → LIF    │        4 MAC/Takt → Argmax
                            │   └──────────────────────┘                  │
                            │     BSRAM: syn ptr w_in acc_rec v i counts  │
                            └──────────── xorshift32 (Eingangsspikes) ────┘
```

## 2. Speicherplan

Ausgabe von [`tools/export_variant_e.py`](tools/export_variant_e.py) für die Dimensionen des
echten Modells (Gowin-BSRAM: 46 Blöcke à 18 Kbit, Konfiguration 1K×18 / 2K×9 / 512×36):

| Speicher | Inhalt | Format | Blöcke |
|---|---|---|---|
| `syn` | 9.534 Synapsen ≠ 0 nach INT8-Rundung (von 14.998), nach präsynaptischem Neuron sortiert | 18 Bit `{post[17:8], w[7:0]}`, passt genau in 1K×18 | 10 (15 bei allen Kanten) |
| `ptr` | 801 Startadressen | 14 Bit | 1 |
| `w_in` | 800 × 52 Gewichte | 18 Bit = 2 × int8 (Bits 16:9 und 7:0), 26 Wörter pro Neuron | 21 |
| `acc_rec` | rekurrenter Akkumulator je Neuron | 16 Bit | 1 |
| `v`, `i_syn` | Zustände | je 16 Bit | 2 |
| `counts` | Spikezähler, 4 pro Wort | 4 × 7 Bit (0–100) | 1 |
| **Summe** | | | **36 von 46** (41 bei allen Kanten) |
| SDRAM | `w_out` klassenweise (800 Byte pro Klasse) + `b_out` | int8 / int32 | 544 KiB von 8 MiB |

Spike-Vektoren `s_prev`/`s_curr` (je 800 Bit) und die 52 Eingangsspikes liegen in Registern oder
Shadow-SRAM (LUT-RAM).

**Reserve:** 5 Blöcke. Reicht es später nicht (z. B. 9-Bit-Gewichte oder mehr Neuronen), wandern
zuerst `counts` und `acc_rec` ins Shadow-SRAM.

## 3. Ablauf pro Zeitschritt

Phase A und B setzen `fixed_forward()` exakt um. Die Integer-Summen sind von der Reihenfolge
unabhängig, deshalb darf die Hardware anders sortieren als numpy.

```
Phase 0  Eingangsspikes      52 Takte    e[k] = xorshift32()[31:24] < r[k]

Phase A  rec_scatter         ≈ 19 Takte pro Spike aus dem letzten Schritt
         für jedes j mit s_prev[j] = 1:
             für a in ptr[j] .. ptr[j+1]-1:
                 {post, w} = syn[a]
                 acc_rec[post] += w            // Read-Modify-Write, 1/Takt mit Forwarding

Phase B  neuron_loop         13 Takte pro Neuron
         für i = 0..799:
             acc_in = Σ_k e[k]·r[k]·w_in[i,k]   // 4 Gewichte/Takt: Dual-Port × 2 pro Wort
             cur    = rnd(acc_in, SH_IN) + rnd(acc_rec[i], SH_REC);  acc_rec[i] = 0
             i_new  = sat16(i_syn[i] + cur)                    // Norse 1.1.0: Eingang zuerst
             v_dec  = sat16(v[i] + mulq16(K_MEM, i_new − v[i]))
             i_syn[i] = i_new − mulq16(K_SYN, i_new)
             z      = v_dec > V_TH
             v[i]   = z ? 0 : v_dec
             s_curr[i] = z;  counts[i] += z
         s_prev ← s_curr
```

Phase B ist eine Pipeline: Während Neuron *i* seine 13 MAC-Takte läuft, wird der LIF-Update von
Neuron *i−1* zurückgeschrieben. Multiplizierer: 4 für `r·w`, 2 für die Leak-Konstanten, 4 für den
Readout, also 10 von 48.

## 4. Readout

Nach `N_RUNS × 20` Schritten (Zähler laufen über alle Monte-Carlo-Läufe weiter):

```
für c = 0..691:                          // SDRAM-Burst, 200 Wörter à 4 Gewichte
    acc = b_out[c] + Σ_i counts[i] · w_out[c,i]     // 4 MAC/Takt
    Argmax laufend mitführen; optional acc per UART-FIFO ausgeben
```

Klassenweise Ordnung bedeutet: Pro Klasse gibt es einen einzigen Akkumulator, die Logits müssen
nicht zwischengespeichert werden. Argmax läuft im Strom mit. Wer alle Logits braucht (Softmax,
Top-k im Host), bekommt sie direkt beim Rechnen über einen FIFO an den UART.

## 5. Zeitbudget (54 MHz, Annahme 5 % Feuerrate)

| Teil | Takte | Zeit |
|---|---|---|
| Phase 0 | 52 | 1 µs |
| Phase A: 40 Spikes × 18,7 Synapsen | ≈ 750 | 14 µs |
| Phase B: 800 × 13 | 10.400 | 193 µs |
| **Zeitschritt** | ≈ 11.200 | **≈ 210 µs** |
| Inferenz: 20 Schritte × 5 Läufe | | 21 ms |
| Readout: 553.600 Byte / 4 pro Takt, +20 % SDRAM-Overhead | ≈ 166.000 | 3 ms |
| **Klassifikation gesamt** | | **≈ 24 ms → ~40 pro Sekunde** |

Die Gassensoren liefern 1–10 Messungen pro Sekunde, die Reserve ist also groß. Wird es doch eng,
hilft zuerst eine höhere Taktrate (PLL), dann mehr Eingangs-MACs pro Takt.

## 6. Ressourcen (Schätzung)

| Block | LUT4 | FF | DSP (Multiplizierer) |
|---|---|---|---|
| step_engine (A + B, Pipeline) | ≈ 2.500 | ≈ 1.500 | 6 |
| readout_engine + Argmax | ≈ 600 | ≈ 400 | 4 |
| SDRAM-Controller | ≈ 1.000 | ≈ 600 | – |
| UART, xorshift, Steuerung, Spike-Register | ≈ 1.500 | ≈ 1.900 | – |
| **Summe** | **≈ 5.600 von 20.736** | **≈ 4.400 von 15.552** | **10 von 48** |

## 7. Laden der Gewichte

| Stufe | `syn`, `ptr`, `w_in` (BSRAM) | `w_out`, `b_out` (SDRAM) |
|---|---|---|
| **2a – Prototyp** | `$readmemh` beim Synthetisieren, Gewichte stecken im Bitstream | per UART beim Start (544 KiB bei 3 Mbit/s ≈ 1,8 s) |
| **2b – Feld** | ebenfalls per UART ladbar, der Bitstream ist dann gewichtsunabhängig | aus dem SPI-Flash des Boards (64 Mbit, Bitstream belegt nur einen Teil) |

## 8. UART-Protokoll (Vorschlag)

| Richtung | Frame |
|---|---|
| MCU → FPGA | `0xA5 'C' seq seed[32] r[52] crc8`: klassifizieren |
| MCU → FPGA | `0xA5 'L' region addr[24] len[16] daten… crc8`: Gewichte laden |
| FPGA → MCU | `0x5A 'R' seq klasse[16] logit_max[32] logit_2[32] spikes[16] crc8`: Ergebnis |
| FPGA → MCU | `0x5A 'D' …`: Debug, Spike-Vektor pro Schritt (für den Vergleich mit `snn_ref.py`) |

## 9. Verifikation

1. `verify_torch.py`: Das Float-Modell in `snn_ref.py` entspricht `modeling_snn.py` + Norse.
2. `snn_ref.py --dump tv.npz`: erzeugt Testvektoren (Raten, Eingangsspikes, Spikes pro Schritt, Zähler, Logits).
3. Verilator-Testbench lädt die Hex-Dateien aus `export_variant_e.py`, spielt `tv.npz` ab und vergleicht jeden Spike in jedem Schritt.
4. Auf dem Board: Debug-Frames gegen dieselben Testvektoren.

## 10. Vom einen Board zum Cluster: Variante F („E-Cluster“)

Wenn ein Board das ganze Modell trägt, nutzt man die übrigen neun am einfachsten als
**unabhängige Kopien**:

- **Parallele Monte-Carlo-Läufe:** 5 Boards rechnen je einen Lauf mit eigenem Seed. Die Zähler oder Logits werden auf dem Host addiert. Das ist exakt gleichwertig, weil der Readout linear ist. Eine Klassifikation dauert dann etwa 5 ms statt 24 ms
- **Stereo:** je ein Board pro Sensorkopf links und rechts
- **Redundanz:** 3 Boards mit gleichem Seed, Mehrheitsentscheid

Verbindung: ein gemeinsamer SPI-Bus vom Host (oder von der MCU) mit einem Chip-Select pro Board.
Ring, Taktsynchronisation und unterschiedliche Speicherabbilder pro Board entfallen.

**Empfehlung:** Für das veröffentlichte Modell ist **E bzw. F** die bessere Zielarchitektur als
der Ring aus Variante A. A lohnt sich erst, wenn das Netz nicht mehr auf einen Chip passt,
z. B. mit einem größeren FlyWire-Ausschnitt (Variante D).
