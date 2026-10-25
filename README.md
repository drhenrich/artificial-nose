![Featured on MAKE:](_assets/FeaturedonMakebadge_32x32.png) [![PlatformIO CI](https://github.com/kartben/artificial-nose/actions/workflows/platformio.yml/badge.svg)](https://github.com/kartben/artificial-nose/actions/workflows/platformio.yml)

# Artificial Nose

This repository contains all the instructions, source code, and resources for building your very own DIY artificial nose, powered by TinyML.

## Introduction

Some time back in May 2020, like I'm guessing many other people, I spent quite some time trying to perfect my bread recipe, including trying to determine when my sourdough starter would be in the ideal condition to bake perfect baguettes.

Fast-forward to a few weeks later, I had assembled a full-blown (pun intended!) artificial nose. It can be used for a wide variety of applications, from helping folks suffering from anosmia to spot the smell of burning food or spoiled milk, to monitoring the cleanliness of office buildings, etc.

I am sharing in this repository all the instructions, source code, bill of materials, etc. for you to build and train your very own artificial nose. Enjoy! 👃

## Repository contents

* 💰 [Bill of materials](./bom/README.md) ;
* ⚡ [Schematics & Wiring diagram](./schematics/README.md) ;
* 👩‍💻 [Firmware source code](./firmware) ;
* 🧊 [3D enclosure model files](./enclosure/README.md) ;
* 🪰 [FlyWire olfactory SNN on a Tang Nano 20K FPGA](./snn-fpga/README.md) (German).

## 🪰 FlyWire-SNN auf FPGA (Erweiterung)

Dieser Fork erweitert die künstliche Nase um ein biologisch beschränktes Spiking Neural Network.
Das Modell [`Vick-MIRE/flywire-olfactory-snn`](https://huggingface.co/Vick-MIRE/flywire-olfactory-snn)
hat 800 LIF-Neuronen, deren Verschaltung aus dem FlyWire-Konnektom der Fruchtfliege stammt.
Es läuft bitgenau auf einem Tang Nano 20K (Gowin GW2AR-18) und ist für den Einsatz auf einem
DeepRobotics Lite3 gedacht.

- **Stand:** Verilog-Kern auf dem Board bestätigt, alle Spikezähler in 10 von 10 Proben bitgenau wie das Python-Referenzmodell. Vollversion mit SDRAM-Readout (31,8 ms pro Klassifikation) und Sensor-Neutraining in der Simulation verifiziert
- **Einstieg:** [snn-fpga/README.md](./snn-fpga/README.md), Konzept in [snn-fpga/KONZEPT.md](./snn-fpga/KONZEPT.md), Hardware in [snn-fpga/VARIANTE-E.md](./snn-fpga/VARIANTE-E.md)

## Author <!-- omit in toc -->

👤 **Benjamin Cabé**

- Website: [https://blog.benjamin-cabe.com](https://blog.benjamin-cabe.com)
- Twitter: [@kartben](https://twitter.com/kartben)
- Github: [@kartben](https://github.com/kartben)
- LinkedIn: [@benjamincabe](https://linkedin.com/in/benjamincabe)

## 🤝 Contributing <!-- omit in toc -->

Contributions, issues and feature requests are welcome!

Feel free to check [issues page](https://github.com/kartben/artificial-nose/issues).

## Show your support <!-- omit in toc -->

Give a ⭐️ if this project helped you!


## 📝 License <!-- omit in toc -->

Copyright &copy; 2020-2021 [Benjamin Cabé](https://github.com/kartben).

This project is [MIT](/LICENSE) licensed.

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_