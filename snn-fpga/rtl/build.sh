#!/bin/sh
# Bitstream fuer den Tang Nano 20K (Bring-up, top_tangnano20k) mit der Open-Source-Toolchain
#   ./build.sh <build-verzeichnis>
# Das Build-Verzeichnis enthaelt params.vh und die *.hex aus export_variant_e.py.
# Benoetigt yosys, nextpnr-himbaechel (mit Gowin) und gowin_pack (apycula), z. B. aus der
# OSS CAD Suite. Ergebnis: <build-verzeichnis>/snn_tangnano20k.fs
set -e
BUILD=$(cd "$1" && pwd)
RTL=$(cd "$(dirname "$0")" && pwd)
YOSYS=${YOSYS:-yosys}
NEXTPNR=${NEXTPNR:-nextpnr-himbaechel}
PACK=${PACK:-gowin_pack}
cd "$BUILD"

$YOSYS -q -l yosys.log -p "read_verilog -I. $RTL/top_tangnano20k.v $RTL/snn_core.v $RTL/uart.v; \
    synth_gowin -family gw2a -top top_tangnano20k -json top.json"

$NEXTPNR --json top.json --write pnr.json --device GW2AR-LV18QN88C8/I7 \
    --vopt family=GW2A-18C --vopt cst="$RTL/tangnano20k.cst" \
    --freq 27 --log nextpnr.log

$PACK -d GW2A-18C -o snn_tangnano20k.fs pnr.json
echo "Fertig: $BUILD/snn_tangnano20k.fs"
echo "Laden (fluechtig):   openFPGALoader -b tangnano20k $BUILD/snn_tangnano20k.fs"
echo "Flashen (dauerhaft): openFPGALoader -b tangnano20k -f $BUILD/snn_tangnano20k.fs"
