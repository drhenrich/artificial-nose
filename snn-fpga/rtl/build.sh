#!/bin/sh
# Bitstream fuer den Tang Nano 20K mit der Open-Source-Toolchain
#   ./build.sh <build-verzeichnis> [full|bringup]
#     full     (Standard) top_tangnano20k_full: 54 MHz, SDRAM-Readout, 1 Mbaud
#     bringup  top_tangnano20k: 27 MHz, Zaehlerausgabe, 115200 Baud
# Das Build-Verzeichnis enthaelt params.vh und die *.hex aus export_variant_e.py.
# Benoetigt yosys, nextpnr-himbaechel (mit Gowin) und gowin_pack (apycula), z. B. aus der
# OSS CAD Suite. Ergebnis: <build-verzeichnis>/snn_<variante>.fs
set -e
BUILD=$(cd "$1" && pwd)
VARIANT=${2:-full}
RTL=$(cd "$(dirname "$0")" && pwd)
YOSYS=${YOSYS:-yosys}
NEXTPNR=${NEXTPNR:-nextpnr-himbaechel}
PACK=${PACK:-gowin_pack}
cd "$BUILD"

if [ "$VARIANT" = bringup ]; then
    TOP=top_tangnano20k; SRC="$RTL/top_tangnano20k.v"; FREQ=27
else
    TOP=top_tangnano20k_full; SRC="$RTL/top_tangnano20k_full.v $RTL/sdram_ctrl.v"; FREQ=54
fi

# Optional andere Baudrate: BAUD=115200 ./build.sh ...
CHPARAM=""
SUFFIX=""
if [ -n "$BAUD" ]; then CHPARAM="chparam -set BAUD $BAUD $TOP;"; SUFFIX="_$BAUD"; fi

$YOSYS -q -l yosys_$VARIANT$SUFFIX.log -p "read_verilog -I. $SRC $RTL/snn_core.v $RTL/uart.v; \
    $CHPARAM synth_gowin -family gw2a -top $TOP -json $TOP$SUFFIX.json"

$NEXTPNR --json $TOP$SUFFIX.json --write pnr_$VARIANT$SUFFIX.json --device GW2AR-LV18QN88C8/I7 \
    --vopt family=GW2A-18C --vopt cst="$RTL/tangnano20k.cst" \
    --freq $FREQ --log nextpnr_$VARIANT$SUFFIX.log

$PACK -d GW2A-18C -o snn_$VARIANT$SUFFIX.fs pnr_$VARIANT$SUFFIX.json
echo "Fertig: $BUILD/snn_$VARIANT$SUFFIX.fs"
echo "Laden (fluechtig):   openFPGALoader -b tangnano20k $BUILD/snn_$VARIANT$SUFFIX.fs"
echo "Flashen (dauerhaft): openFPGALoader -b tangnano20k -f $BUILD/snn_$VARIANT$SUFFIX.fs"
