// sdram_ctrl.v - Controller fuer das eingebettete SDRAM des GW2AR-18 (64 Mbit, 2M x 32)
//
// Wortadresse addr[20:0] = {bank[1:0], row[10:0], col[7:0]}.
// - Lese-Burst: rd_req (Puls) mit rd_addr, rd_len; Daten folgen in Reihenfolge mit rd_valid.
//   Ein READ pro Takt, Zeilenwechsel werden mit PRECHARGE/ACTIVATE ueberbrueckt.
// - Schreiben: wr_req (Puls) mit wr_addr, wr_data; wr_done-Puls nach dem Schreiben
//   (ACTIVATE + WRITE mit Auto-Precharge).
// Anfragen werden zwischengespeichert und nach einem faelligen Refresh bedient.
// SDRAM-Takt = invertierter Systemtakt: Befehle wechseln mit der steigenden Flanke von clk,
// das SDRAM uebernimmt sie mit der fallenden. Lesedaten (CL = 2) werden RD_LAT Takte nach
// dem READ mit der steigenden Flanke von clk uebernommen.

module sdram_ctrl #(
    parameter CLK_MHZ  = 54,
    parameter RD_LAT   = 3,        // Takte vom READ bis zur Uebernahme der Daten
    parameter T_RP     = 2,        // Precharge -> naechster Befehl
    parameter T_RCD    = 2,        // Activate -> Read/Write
    parameter T_RFC    = 6,        // Refresh -> naechster Befehl
    parameter T_WRP    = 4,        // Write mit Auto-Precharge -> naechster Befehl (tWR + tRP)
    parameter REF_INT  = 600       // Takte zwischen Refreshes (Vorgabe 15,6 us)
) (
    input  wire        clk,
    input  wire        rst,
    output reg         init_done,
    // Lesen
    input  wire        rd_req,
    input  wire [20:0] rd_addr,
    input  wire [8:0]  rd_len,
    output reg         rd_valid,
    output reg  [31:0] rd_data,
    // Schreiben
    input  wire        wr_req,
    input  wire [20:0] wr_addr,
    input  wire [31:0] wr_data,
    output reg         wr_done,
    // SDRAM
    output wire        O_sdram_clk,
    output wire        O_sdram_cke,
    output wire        O_sdram_cs_n,
    output wire        O_sdram_ras_n,
    output wire        O_sdram_cas_n,
    output wire        O_sdram_wen_n,
    output wire [3:0]  O_sdram_dqm,
    output reg  [10:0] O_sdram_addr,
    output reg  [1:0]  O_sdram_ba,
    inout  wire [31:0] IO_sdram_dq
);
    localparam INIT_WAIT = 200 * CLK_MHZ;          // 200 us nach dem Einschalten
    // Befehle {ras_n, cas_n, we_n}
    localparam C_NOP = 3'b111, C_ACT = 3'b011, C_RD = 3'b101, C_WR = 3'b100,
               C_PRE = 3'b010, C_REF = 3'b001, C_MRS = 3'b000;
    // Mode: Burstlaenge 1, sequentiell, CL = 2, Burst-Read/Burst-Write
    localparam [10:0] MODE = 11'b000_0_010_0_000;

    reg [2:0]  cmd = C_NOP;
    reg [31:0] dq_out;
    reg        dq_oe = 1'b0;
    assign O_sdram_clk   = ~clk;
    assign O_sdram_cke   = 1'b1;
    assign O_sdram_cs_n  = 1'b0;
    assign {O_sdram_ras_n, O_sdram_cas_n, O_sdram_wen_n} = cmd;
    assign O_sdram_dqm   = 4'b0000;
    assign IO_sdram_dq   = dq_oe ? dq_out : 32'bz;

    localparam S_INIT = 4'd0, S_IPRE = 4'd1, S_IREF1 = 4'd2, S_IREF2 = 4'd3, S_IMRS = 4'd4,
               S_IDLE = 4'd5, S_REF = 4'd6, S_WACT = 4'd7, S_WWR = 4'd8, S_RACT = 4'd9,
               S_RRD = 4'd10, S_RPRE = 4'd11, S_WAIT = 4'd12;
    reg [3:0]  state = S_INIT, after;
    reg [15:0] wait_cnt;
    reg [15:0] ref_cnt;
    reg        ref_due;

    reg        rd_pend = 1'b0, wr_pend = 1'b0;
    reg [20:0] ra, wa;
    reg [8:0]  rleft;
    reg [31:0] wd;
    reg [7:0]  rd_pipe = 8'd0;          // Bit n: READ lag vor n+1 Takten auf dem Bus

    // Lesedaten im IO-nahen Register uebernehmen: READ ab Flanke n auf dem Bus,
    // Uebernahme mit Flanke n + RD_LAT (2 <= RD_LAT <= 9)
    always @(posedge clk) begin
        rd_pipe  <= {rd_pipe[6:0], cmd == C_RD};
        rd_valid <= rd_pipe[RD_LAT-2];
        rd_data  <= IO_sdram_dq;
    end

    always @(posedge clk) begin
        cmd     <= C_NOP;
        dq_oe   <= 1'b0;
        wr_done <= 1'b0;
        if (rd_req) begin rd_pend <= 1'b1; ra <= rd_addr; rleft <= rd_len; end
        if (wr_req) begin wr_pend <= 1'b1; wa <= wr_addr; wd <= wr_data; end

        if (ref_cnt != 0) ref_cnt <= ref_cnt - 16'd1;
        else begin ref_cnt <= REF_INT; ref_due <= 1'b1; end

        if (rst) begin
            state     <= S_INIT;
            wait_cnt  <= INIT_WAIT;
            init_done <= 1'b0;
            rd_pend   <= 1'b0;
            wr_pend   <= 1'b0;
            ref_due   <= 1'b0;
            ref_cnt   <= REF_INT;
        end else case (state)
        S_INIT: if (wait_cnt != 0) wait_cnt <= wait_cnt - 16'd1;
                else begin
                    cmd <= C_PRE; O_sdram_addr <= 11'h400;              // alle Baenke
                    wait_cnt <= T_RP; after <= S_IREF1; state <= S_WAIT;
                end
        S_IREF1: begin cmd <= C_REF; wait_cnt <= T_RFC; after <= S_IREF2; state <= S_WAIT; end
        S_IREF2: begin cmd <= C_REF; wait_cnt <= T_RFC; after <= S_IMRS;  state <= S_WAIT; end
        S_IMRS: begin
            cmd <= C_MRS; O_sdram_addr <= MODE; O_sdram_ba <= 2'b00;
            wait_cnt <= 3; after <= S_IDLE; state <= S_WAIT;
            init_done <= 1'b1;
        end

        S_IDLE: begin
            if (ref_due) begin
                ref_due <= 1'b0;
                cmd <= C_REF; wait_cnt <= T_RFC; after <= S_IDLE; state <= S_WAIT;
            end else if (wr_pend) begin
                cmd <= C_ACT; O_sdram_ba <= wa[20:19]; O_sdram_addr <= wa[18:8];
                wait_cnt <= T_RCD - 1; after <= S_WWR; state <= S_WAIT;
            end else if (rd_pend) begin
                state <= S_RACT;
            end
        end

        S_WWR: begin
            cmd <= C_WR; O_sdram_addr <= {3'b100, wa[7:0]};          // A10 = Auto-Precharge
            dq_out <= wd; dq_oe <= 1'b1;
            if (!wr_req) wr_pend <= 1'b0;
            wr_done <= 1'b1;
            wait_cnt <= T_WRP; after <= S_IDLE; state <= S_WAIT;
        end

        S_RACT: begin
            cmd <= C_ACT; O_sdram_ba <= ra[20:19]; O_sdram_addr <= ra[18:8];
            wait_cnt <= T_RCD - 1; after <= S_RRD; state <= S_WAIT;
        end

        S_RRD: begin                                                  // ein READ pro Takt
            cmd <= C_RD; O_sdram_addr <= {3'b000, ra[7:0]};
            ra    <= ra + 21'd1;
            rleft <= rleft - 9'd1;
            if (rleft == 9'd1 || ra[7:0] == 8'hFF) state <= S_RPRE;  // Ende oder Zeilenende
        end

        S_RPRE: begin
            cmd <= C_PRE; O_sdram_addr <= 11'h400;
            if (rleft == 9'd0 && !rd_req) rd_pend <= 1'b0;
            wait_cnt <= T_RP; after <= (rleft == 9'd0) ? S_IDLE : S_RACT; state <= S_WAIT;
        end

        S_WAIT: if (wait_cnt > 1) wait_cnt <= wait_cnt - 16'd1;
                else state <= after;
        default: state <= S_INIT;
        endcase
    end
endmodule
