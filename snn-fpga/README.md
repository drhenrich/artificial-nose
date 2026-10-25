# FlyWire-Olfaktorik-SNN auf dem Tang Nano 20K

Eine „Fliegennase“ für den DeepRobotics Lite3. Das konnektom-beschränkte Spiking Neural Network
[`Vick-MIRE/flywire-olfactory-snn`](https://huggingface.co/Vick-MIRE/flywire-olfactory-snn)
läuft bitgenau auf einem Gowin-FPGA (Tang Nano 20K, GW2AR-18). Die Verschaltung der 800 Neuronen
stammt aus dem FlyWire-Konnektom der Fruchtfliege: Projektionsneuronen des Antennallobus und
Kenyonzellen.

## Stand (25.10.2026)

| Baustein | Status |
|---|---|
| Konzept, Architekturvarianten, Lite3-Integration | [KONZEPT.md](KONZEPT.md) |
| Float-Referenz in numpy | gegen PyTorch/Norse 1.1.0 geprüft (Logit-Abweichung < 10⁻⁵) |
| Bitgenaues Fixpunkt-Modell | 9-Bit-Gewichte, Q8.7-Zustände; Argmax 96 % gleich wie Float bei gleichen Eingangsspikes |
| Verilog-Kern (Variante E: ganzes Modell auf einem Board) | Verilator: 2.000 Zeitschritte, alle Spikes und Logits bitgenau |
| Bitstream (yosys + nextpnr-himbaechel + apycula) | 21 % LUT, 37 von 46 BSRAM, Fmax 58 MHz |
| **Test auf dem Tang Nano 20K** | **10 von 10 Proben bitgenau (8.000 Spikezähler), 187 ms pro Probe** |
| Readout (692 Klassen) auf dem FPGA mit SDRAM | in Simulation verifiziert, SDRAM-Controller in Arbeit |
| Beschleunigung (≈ 112 → 20–30 ms), Anbindung der Gassensoren, Neutraining | in Arbeit |

Details zu Variante E, Speicherplan, Zeitbudget und Inbetriebnahme: [VARIANTE-E.md](VARIANTE-E.md).

## Schnellstart

```bash
# Modell holen (Hugging Face)
hf download Vick-MIRE/flywire-olfactory-snn --local-dir flywire-snn

# Modell untersuchen und Fixpunkt-Version gegen Float vergleichen
python3 snn-fpga/tools/inspect_model.py flywire-snn/model.safetensors flywire-snn/connectome_mask.npz
python3 snn-fpga/tools/snn_ref.py       flywire-snn/model.safetensors flywire-snn/connectome_mask.npz

# Speicherabbilder + Testvektoren, Simulation, Bitstream
python3 snn-fpga/tools/export_variant_e.py flywire-snn/model.safetensors flywire-snn/connectome_mask.npz --out build/e
snn-fpga/rtl/sim.sh   build/e            # Verilator: PASS erwartet
snn-fpga/rtl/build.sh build/e            # -> build/e/snn_tangnano20k.fs

# Auf dem Board testen
openFPGALoader -b tangnano20k build/e/snn_tangnano20k.fs
pip install pyserial numpy
python3 snn-fpga/tools/board_test.py flywire-snn/model.safetensors flywire-snn/connectome_mask.npz \
        --port /dev/cu.usbserial-XXXX1 --n 10      # Linux: /dev/ttyUSB1, Windows: COMx
```

## Aufbau

```
snn-fpga/
├── KONZEPT.md               Gesamtkonzept: Modell, Messungen, Varianten A–F, Sensorik, Lite3
├── VARIANTE-E.md            Ein-Board-Variante: Formate, Speicherplan, Ablauf, Inbetriebnahme
├── tools/
│   ├── inspect_model.py     Tensoren, Sparsity, Vorzeichen, BSRAM-Budget
│   ├── snn_ref.py           Float- und bitgenaues Fixpunkt-Modell (Spezifikation für die HDL)
│   ├── verify_torch.py      Float-Referenz gegen modeling_snn.py + Norse
│   ├── export_variant_e.py  Speicherabbilder (*.hex), params.vh, Testvektoren
│   └── board_test.py        Test auf dem Board über USB-UART
└── rtl/
    ├── snn_core.v           SNN-Kern (LIF wie Norse 1.1.0, Readout bzw. Zählerausgabe)
    ├── top_tangnano20k.v    Bring-up-Top: UART, LEDs
    ├── uart.v, tangnano20k.cst
    ├── tb/                  Testbenches (Kern mit SDRAM-Modell, Top über UART)
    ├── sim.sh               Simulation (Verilator oder iverilog)
    └── build.sh             Bitstream (OSS CAD Suite)
```

## Wichtige Befunde

- Das Modell hat **692 Klassen** (laut `config.json`, die Model Card nennt 500) und **14.998 Synapsen** (2,34 % Dichte). Das ganze Netz passt deshalb auf einen einzelnen GW2AR-18.
- Norse 1.1.0 addiert den Eingang **vor** dem Membran-Update. Ein Kern mit der älteren Reihenfolge liegt einen Zeitschritt daneben.
- 46 % der rekurrenten Vorzeichen sind nach dem Training gegen das Konnektom gekippt (`recurrent` ist nicht auf ≥ 0 beschränkt). Biologisch fixiert ist nur die Topologie.
- 32 % der Synapsengewichte sind im trainierten Modell praktisch 0.
- Die Streuung durch die Poisson-Kodierung ist größer als der Quantisierungsfehler. Mehr Monte-Carlo-Läufe bringen mehr als mehr Bits.

## Lizenzen und Zitate

Die Modellgewichte und Speicherabbilder sind nicht im Repository enthalten. Sie werden lokal aus
dem Hugging-Face-Modell erzeugt. Datengrundlage:

- DoOR 2.0, CC BY-SA 4.0: Münch & Galizia (2016), *Scientific Reports* 6, 21841. https://doi.org/10.1038/srep21841
- FlyWire: Dorkenwald et al. (2024), *Nature* 634, 124–138. https://doi.org/10.1038/s41586-024-07558-y
