// sdram_model.v - Verhaltensmodell 2M x 32 SDRAM (4 Baenke, 11 Zeilen-, 8 Spaltenbits)
// fuer die Simulation. CL 2, Burstlaenge 1. Prueft Protokoll und Zeitabstaende in Takten des
// SDRAM-Takts und zaehlt Verstoesse in "errors". Lesedaten erscheinen T_AC nach der Flanke
// CL Takte nach dem READ und bleiben bis T_OH nach der folgenden Flanke stehen.
`timescale 1ns/1ps

module sdram_model #(
    parameter WORDS   = 1 << 18,      // simulierter Speicher (Woerter ab Adresse 0)
    parameter T_AC    = 5.4,
    parameter T_OH    = 2.7,
    parameter T_RCD   = 2,
    parameter T_RP    = 2,
    parameter T_RFC   = 4,
    parameter REF_MAX = 843           // 15,6 us bei 54 MHz
) (
    input  wire        clk,
    input  wire        cke,
    input  wire        cs_n, ras_n, cas_n, we_n,
    input  wire [3:0]  dqm,
    input  wire [10:0] addr,
    input  wire [1:0]  ba,
    inout  wire [31:0] dq
);
    reg [31:0] mem [0:WORDS-1];
    reg        open_ [0:3];
    reg [10:0] row  [0:3];
    integer    t_act [0:3], t_pre [0:3];
    integer    cyc = 0, t_ref = -100000, t_ref_last = 0, max_ref_gap = 0, errors = 0;
    integer    n_rd = 0, n_wr = 0, n_ref = 0;
    reg        mode_ok = 1'b0;
    integer    i;

    reg [31:0] dq_drv;
    reg        dq_en = 1'b0;
    assign dq = dq_en ? dq_drv : 32'bz;

    reg        rd_p1 = 1'b0, rd_p2 = 1'b0;
    reg [31:0] rd_d1, rd_d2;

    initial for (i = 0; i < 4; i = i + 1) begin open_[i] = 1'b0; t_act[i] = -1000; t_pre[i] = -1000; end

    function [20:0] wa(input [1:0] b, input [10:0] r, input [7:0] c);
        wa = {b, r, c};
    endfunction

    task err(input [8*48-1:0] msg);
        begin
            if (errors < 10) $display("SDRAM-FEHLER @%0d: %0s", cyc, msg);
            errors = errors + 1;
        end
    endtask

    reg [31:0] out_d;
    always @(posedge clk) begin
        cyc = cyc + 1;
        // Lesepipeline (CL = 2): Daten der Flanke k erscheinen T_AC nach Flanke k+2.
        // Wert vor der Verzoegerung festhalten (rd_d2 aendert sich mit dieser Flanke).
        if (rd_p2) begin
            out_d = rd_d2;
            #(T_AC) dq_drv = out_d; dq_en = 1'b1;
        end
    end

    always @(posedge clk) begin
        // Halten bis T_OH nach der naechsten Flanke, dann freigeben falls keine Folgedaten
        if (!rd_p2 && dq_en) begin
            #(T_OH) if (!rd_p2) dq_en = 1'b0;
        end
    end

    always @(posedge clk) begin
        rd_p2 <= rd_p1; rd_d2 <= rd_d1;
        rd_p1 <= 1'b0;
        if (cke && !cs_n) begin
            case ({ras_n, cas_n, we_n})
            3'b011: begin                                                // ACTIVATE
                if (open_[ba]) err("ACT auf offene Bank");
                if (cyc - t_pre[ba] < T_RP) err("tRP verletzt");
                if (cyc - t_ref < T_RFC) err("tRFC verletzt");
                open_[ba] <= 1'b1; row[ba] <= addr; t_act[ba] = cyc;
            end
            3'b101: begin                                                // READ
                if (!mode_ok) err("READ vor Mode-Register");
                if (!open_[ba]) err("READ auf geschlossene Bank");
                if (cyc - t_act[ba] < T_RCD) err("tRCD verletzt (READ)");
                rd_p1 <= 1'b1;
                rd_d1 <= mem[wa(ba, row[ba], addr[7:0]) % WORDS];
                n_rd = n_rd + 1;
                if (addr[10]) begin open_[ba] <= 1'b0; t_pre[ba] = cyc + 1; end
            end
            3'b100: begin                                                // WRITE
                if (!open_[ba]) err("WRITE auf geschlossene Bank");
                if (cyc - t_act[ba] < T_RCD) err("tRCD verletzt (WRITE)");
                if (dqm != 4'b0000) err("DQM erwartet 0");
                mem[wa(ba, row[ba], addr[7:0]) % WORDS] <= dq;
                n_wr = n_wr + 1;
                if (addr[10]) begin open_[ba] <= 1'b0; t_pre[ba] = cyc + 2; end
            end
            3'b010: begin                                                // PRECHARGE
                for (i = 0; i < 4; i = i + 1)
                    if (addr[10] || ba == i) begin open_[i] <= 1'b0; t_pre[i] = cyc; end
            end
            3'b001: begin                                                // AUTO REFRESH
                for (i = 0; i < 4; i = i + 1) if (open_[i]) err("REFRESH mit offener Bank");
                if (mode_ok && cyc - t_ref_last > max_ref_gap) max_ref_gap = cyc - t_ref_last;
                t_ref = cyc; t_ref_last = cyc; n_ref = n_ref + 1;
            end
            3'b000: begin                                                // MODE REGISTER
                for (i = 0; i < 4; i = i + 1) if (open_[i]) err("MRS mit offener Bank");
                if (addr[6:4] != 3'd2 || addr[2:0] != 3'd0) err("Mode: CL 2 / BL 1 erwartet");
                mode_ok <= 1'b1;
                t_ref_last = cyc;
            end
            default: ;
            endcase
        end
        if (mode_ok && cyc - t_ref_last > REF_MAX && cyc - t_ref_last == REF_MAX + 1) err("Refresh-Intervall ueberschritten");
    end
endmodule
