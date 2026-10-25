#!/bin/sh
# Simulation von snn_core gegen die Testvektoren aus export_variant_e.py
#   ./sim.sh <build-verzeichnis> [verilator|iverilog]
# Das Build-Verzeichnis enthaelt params.vh, tv.vh und alle *.hex.
set -e
BUILD=$(cd "$1" && pwd)
SIM=${2:-verilator}
RTL=$(cd "$(dirname "$0")" && pwd)

if [ "$SIM" = iverilog ]; then
    iverilog -g2012 -I "$BUILD" -P tb_snn_core.MEM_DIR="\"$BUILD/\"" \
        -o "$BUILD/tb_snn_core.vvp" "$RTL/tb/tb_snn_core.v" "$RTL/snn_core.v"
    vvp -n "$BUILD/tb_snn_core.vvp"
else
    verilator --binary -j 0 -O3 -Wno-fatal -Wno-WIDTH -I"$BUILD" \
        -GMEM_DIR="\"$BUILD/\"" --top-module tb_snn_core -Mdir "$BUILD/obj_dir" \
        "$RTL/tb/tb_snn_core.v" "$RTL/snn_core.v" > "$BUILD/verilator.log"
    "$BUILD/obj_dir/Vtb_snn_core"
fi
