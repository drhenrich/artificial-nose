// tb_top_full.v - top_tangnano20k_full mit SDRAM-Modell ueber UART (Bitebene):
//   1. W/R: 64 Woerter in einen freien Bereich schreiben und zuruecklesen
//   2. C: Proben klassifizieren, alle Logits und die Klasse mit tv_logits/tv_class vergleichen
// w_out wird direkt ins SDRAM-Modell geladen (das Laden ueber 'W' prueft Schritt 1).
`timescale 1ns/1ps

module tb_top_full;
`include "params.vh"
`include "tv.vh"

    parameter MEM_DIR = "";
    localparam CPB = 8;
    localparam N_WOUT = N_HID / 4;
    localparam N_CHECK = (N_TV < 2) ? N_TV : 2;

    reg clk = 1'b0;
    always #9.259 clk = ~clk;                  // 54 MHz (SIM = 1: ohne PLL)
    reg  rx = 1'b1;
    wire tx;
    wire [5:0] led;
    wire        s_clk, s_cke, s_cs, s_ras, s_cas, s_we;
    wire [3:0]  s_dqm;
    wire [10:0] s_addr;
    wire [1:0]  s_ba;
    wire [31:0] s_dq;

    top_tangnano20k_full #(.MEM_DIR(MEM_DIR), .CLK_HZ(54_000_000), .BAUD(54_000_000 / CPB), .SIM(1)) dut (
        .clk(clk), .uart_rx(rx), .uart_tx(tx), .led(led),
        .O_sdram_clk(s_clk), .O_sdram_cke(s_cke), .O_sdram_cs_n(s_cs), .O_sdram_ras_n(s_ras),
        .O_sdram_cas_n(s_cas), .O_sdram_wen_n(s_we), .O_sdram_dqm(s_dqm), .O_sdram_addr(s_addr),
        .O_sdram_ba(s_ba), .IO_sdram_dq(s_dq));

    sdram_model #(.WORDS(1 << 18)) mem (
        .clk(s_clk), .cke(s_cke), .cs_n(s_cs), .ras_n(s_ras), .cas_n(s_cas), .we_n(s_we),
        .dqm(s_dqm), .addr(s_addr), .ba(s_ba), .dq(s_dq));

    reg [8:0]  tv_r      [0:N_TV*N_IN-1];
    reg [31:0] tv_seed   [0:N_TV-1];
    reg [31:0] tv_logits [0:N_TV*N_CLS-1];
    reg [15:0] tv_class  [0:N_TV-1];
    reg [7:0]  tv_counts [0:N_TV*N_HID-1];
    reg [31:0] wref      [0:N_CLS*N_WOUT-1];
    initial begin
        $readmemh({MEM_DIR, "tv_counts.hex"}, tv_counts);
        $readmemh({MEM_DIR, "w_out.hex"},     wref);
        $readmemh({MEM_DIR, "tv_r.hex"},      tv_r);
        $readmemh({MEM_DIR, "tv_seed.hex"},   tv_seed);
        $readmemh({MEM_DIR, "tv_logits.hex"}, tv_logits);
        $readmemh({MEM_DIR, "tv_class.hex"},  tv_class);
        $readmemh({MEM_DIR, "w_out.hex"},     mem.mem);     // Adresse = {ba, row, col}
    end

    task send_byte(input [7:0] b);
        integer i;
        begin
            rx = 1'b0; repeat (CPB) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin rx = b[i]; repeat (CPB) @(posedge clk); end
            rx = 1'b1; repeat (CPB) @(posedge clk);
        end
    endtask
    task send_word(input [31:0] w);
        begin send_byte(w[7:0]); send_byte(w[15:8]); send_byte(w[23:16]); send_byte(w[31:24]); end
    endtask

    reg [7:0]  got;
    reg [31:0] gw;
    task recv_byte;
        integer i;
        begin
            @(negedge tx);
            repeat (CPB + CPB / 2) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin got[i] = tx; repeat (CPB) @(posedge clk); end
        end
    endtask
    task recv_word;
        begin
            recv_byte; gw[7:0] = got; recv_byte; gw[15:8] = got;
            recv_byte; gw[23:16] = got; recv_byte; gw[31:24] = got;
        end
    endtask

    function [31:0] pat(input [31:0] a);
        pat = a * 32'h01000193 ^ 32'hC0FFEE00;
    endfunction

    integer i, b, errors = 0, t0, lat, bad, good_lat;
    reg [31:0] sum, cls_w;
    localparam [23:0] TADDR = 24'd200000;
    initial begin
        wait (led[2] == 1'b0);                 // SDRAM initialisiert
        repeat (50) @(posedge clk);
        // ---- 1. W und R
        send_byte(8'hA5); send_byte("W");
        send_byte(TADDR[7:0]); send_byte(TADDR[15:8]); send_byte(TADDR[23:16]);
        send_byte(8'd64); send_byte(8'd0);
        sum = 0;
        for (i = 0; i < 64; i = i + 1) begin send_word(pat(i)); sum = sum + pat(i); end
        recv_byte; if (got !== 8'h5A) errors = errors + 1;
        recv_byte; if (got !== "W") errors = errors + 1;
        recv_word; if (gw !== sum) begin $display("W-Summe %h statt %h", gw, sum); errors = errors + 1; end
        send_byte(8'hA5); send_byte("R");
        send_byte(TADDR[7:0]); send_byte(TADDR[15:8]); send_byte(TADDR[23:16]);
        send_byte(8'd64); send_byte(8'd0);
        recv_byte; if (got !== 8'h5A) errors = errors + 1;
        recv_byte; if (got !== "R") errors = errors + 1;
        for (i = 0; i < 64; i = i + 1) begin
            recv_word;
            if (gw !== pat(i)) begin
                if (errors < 5) $display("R-Wort %0d: %h statt %h", i, gw, pat(i));
                errors = errors + 1;
            end
        end
        $display("W/R: 64 Woerter geschrieben und zurueckgelesen, Fehler bisher %0d", errors);
        // ---- L + B: Latenz durchprobieren, Burst ueber Zeilengrenze (Adresse 230..429)
        good_lat = 0;
        for (lat = 2; lat <= 5; lat = lat + 1) begin
            send_byte(8'hA5); send_byte("L"); send_byte(lat);
            recv_byte; recv_byte; recv_byte;
            if (got !== lat) begin $display("L-Antwort %0d statt %0d", got, lat); errors = errors + 1; end
            send_byte(8'hA5); send_byte("B");
            send_byte(8'd230); send_byte(8'd0); send_byte(8'd0); send_byte(8'd200); send_byte(8'd0);
            recv_byte; recv_byte;
            bad = 0;
            for (i = 0; i < 200; i = i + 1) begin recv_word; if (gw !== wref[230 + i]) bad = bad + 1; end
            $display("Burst mit Latenz %0d: %0d von 200 Woertern falsch", lat, bad);
            if (bad == 0) good_lat = lat;
        end
        if (good_lat != 3) begin $display("erwartet: nur Latenz 3 korrekt"); errors = errors + 1; end
        send_byte(8'hA5); send_byte("L"); send_byte(8'd3); recv_byte; recv_byte; recv_byte;
        // ---- 2. Klassifizieren
        for (b = 0; b < N_CHECK; b = b + 1) begin
            send_byte(8'hA5); send_byte("C");
            send_word(tv_seed[b]);
            for (i = 0; i < N_IN; i = i + 1) begin
                send_byte(tv_r[b*N_IN + i][7:0]); send_byte({7'd0, tv_r[b*N_IN + i][8]});
            end
            t0 = $time;
            recv_byte; if (got !== 8'h5A) errors = errors + 1;
            recv_byte; if (got !== "C") errors = errors + 1;
            recv_byte; cls_w[7:0] = got; recv_byte; cls_w[15:8] = got;
            recv_word;                         // logit_max
            sum = 0;
            for (i = 0; i < N_CLS; i = i + 1) begin
                recv_word;
                sum = sum + gw;
                if (gw !== tv_logits[b*N_CLS + i]) begin
                    if (errors < 5) $display("Probe %0d Logit %0d: %0d statt %0d", b, i, $signed(gw), $signed(tv_logits[b*N_CLS + i]));
                    errors = errors + 1;
                end
            end
            recv_word; if (gw !== sum) begin $display("C-Summe falsch"); errors = errors + 1; end
            if (cls_w[15:0] !== tv_class[b]) begin $display("Klasse %0d statt %0d", cls_w[15:0], tv_class[b]); errors = errors + 1; end
            $display("Probe %0d: Klasse %0d (erwartet %0d), %0d Logits geprueft", b, cls_w[15:0], tv_class[b], N_CLS);
            // ---- Z: Spikezaehler dieser Probe
            send_byte(8'hA5); send_byte("Z");
            recv_byte; recv_byte;
            bad = 0; sum = 0;
            for (i = 0; i < N_HID; i = i + 1) begin
                recv_byte; sum = sum + got;
                if (got !== tv_counts[b*N_HID + i]) bad = bad + 1;
            end
            recv_byte; if (got !== sum[7:0]) bad = bad + 1;
            if (bad != 0) begin $display("Z: %0d Zaehler falsch", bad); errors = errors + 1; end
            else $display("Z: alle %0d Zaehler korrekt", N_HID);
        end
        $display("SDRAM-Modell: Protokollfehler %0d, groesster Refresh-Abstand %0d Takte", mem.errors, mem.max_ref_gap);
        if (errors == 0 && mem.errors == 0) $display("PASS"); else $display("FAIL (%0d Fehler)", errors + mem.errors);
        $finish;
    end
endmodule
