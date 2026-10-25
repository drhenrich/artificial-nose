// tb_sdram.v - sdram_ctrl gegen sdram_model: Schreiben, Lese-Bursts ueber Zeilengrenzen, Refresh
`timescale 1ns/1ps

module tb_sdram;
    reg clk = 1'b0, rst = 1'b1;
    always #9.259 clk = ~clk;                  // 54 MHz

    wire        init_done, rd_valid, wr_done;
    wire [31:0] rd_data;
    reg         rd_req = 1'b0, wr_req = 1'b0;
    reg  [20:0] rd_addr, wr_addr;
    reg  [8:0]  rd_len;
    reg  [31:0] wr_data;
    wire        s_clk, s_cke, s_cs, s_ras, s_cas, s_we;
    wire [3:0]  s_dqm;
    wire [10:0] s_addr;
    wire [1:0]  s_ba;
    wire [31:0] s_dq;

    sdram_ctrl dut (
        .clk(clk), .rst(rst), .init_done(init_done), .rd_lat(3'd3), .rd_neg(1'b0),
        .rd_req(rd_req), .rd_addr(rd_addr), .rd_len(rd_len), .rd_valid(rd_valid), .rd_data(rd_data),
        .wr_req(wr_req), .wr_addr(wr_addr), .wr_data(wr_data), .wr_done(wr_done),
        .O_sdram_clk(s_clk), .O_sdram_cke(s_cke), .O_sdram_cs_n(s_cs), .O_sdram_ras_n(s_ras),
        .O_sdram_cas_n(s_cas), .O_sdram_wen_n(s_we), .O_sdram_dqm(s_dqm), .O_sdram_addr(s_addr),
        .O_sdram_ba(s_ba), .IO_sdram_dq(s_dq));

    sdram_model #(.WORDS(1 << 16)) mem (
        .clk(s_clk), .cke(s_cke), .cs_n(s_cs), .ras_n(s_ras), .cas_n(s_cas), .we_n(s_we),
        .dqm(s_dqm), .addr(s_addr), .ba(s_ba), .dq(s_dq));

    localparam N = 4096;
    reg [31:0] ref_mem [0:N-1];
    function [31:0] pat(input [31:0] a);
        pat = a * 32'h9E3779B1 ^ 32'h5A5A1234;
    endfunction

    integer i, b, errors = 0, got;
    reg [20:0] base;
    initial begin
        repeat (5) @(negedge clk);
        rst = 1'b0;
        wait (init_done);
        repeat (10) @(negedge clk);
        // Schreiben: N Woerter ab 0 (16 Zeilen)
        for (i = 0; i < N; i = i + 1) begin
            @(negedge clk);
            wr_addr = i; wr_data = pat(i); wr_req = 1'b1; ref_mem[i] = pat(i);
            @(negedge clk); wr_req = 1'b0;
            @(posedge wr_done);
        end
        // Leerlauf (Refresh muss weiterlaufen)
        repeat (5000) @(negedge clk);
        // Lese-Bursts a 200 Woerter, verschiedene Startadressen
        for (b = 0; b < 15; b = b + 1) begin
            base = b * 200 + (b % 3) * 37;
            @(negedge clk);
            rd_addr = base; rd_len = 200; rd_req = 1'b1;
            @(negedge clk); rd_req = 1'b0;
            got = 0;
            while (got < 200) begin
                @(posedge clk);
                if (rd_valid) begin
                    if (rd_data !== ref_mem[base + got]) begin
                        if (errors < 5) $display("LESEFEHLER Burst %0d Wort %0d: %h statt %h", b, got, rd_data, ref_mem[base + got]);
                        errors = errors + 1;
                    end
                    got = got + 1;
                end
            end
            repeat (b * 7) @(negedge clk);
        end
        repeat (2000) @(negedge clk);
        $display("SDRAM: %0d Schreib-, %0d Lesezugriffe, %0d Refreshes, groesster Refresh-Abstand %0d Takte",
                 mem.n_wr, mem.n_rd, mem.n_ref, mem.max_ref_gap);
        $display("Lesefehler %0d, Protokollfehler %0d", errors, mem.errors);
        if (errors == 0 && mem.errors == 0) $display("PASS"); else $display("FAIL");
        $finish;
    end
endmodule
