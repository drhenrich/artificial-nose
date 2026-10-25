// snn_core.v - MaskedRecurrentLIFSNN auf einem GW2AR-18 (Variante E)
//
// Bitgenaue Umsetzung von fixed_forward() aus tools/snn_ref.py.
// Parameter und Speicherabbilder erzeugt tools/export_variant_e.py (params.vh, *.hex).
//
// Ablauf einer Inferenz (start):
//   fuer jeden Monte-Carlo-Lauf m < N_RUNS, jeden Schritt t < T_STEPS:
//     S_GEN    52 Eingangsspikes aus xorshift32:  e[k] = x[31:24] < r[k]
//     je Neuron i (sammelnd, "gather"):
//       S_N0..3  Listenanfaenge ptr[2i], ptr[2i+1], ptr[2i+2] lesen
//       S_MAC    parallel: acc_in += e*r*w_in (2 Gewichte/Takt), acc_rec += s_prev[pre]*w_rec
//       S_LIF1/2 LIF-Update wie Norse 1.1.0 (Eingang zuerst), Zustaende und Zaehler schreiben
//   S_RO_*   Readout klassenweise: logit[c] = b[c] + sum_i counts[i]*w_out[c,i], Argmax
//
// w_out liegt extern (SDRAM): pro Klasse ein Burst von N_HID/4 32-Bit-Woertern ab wout_addr,
// Daten kommen in Reihenfolge mit wout_valid (Luecken erlaubt).
// READOUT = 0 (Bring-up ohne SDRAM): statt des Readouts werden die Zaehlerwoerter
// (4 x 7 Bit, Neuron 4j+n in Bits 7n+6:7n) ueber cnt_valid/cnt_idx/cnt_word ausgegeben.

module snn_core #(
    parameter MEM_DIR = "",
    parameter READOUT = 1
) (
    input  wire               clk,
    input  wire               rst,
    // Eingangsraten r[k] in Q0.8 (0..256)
    input  wire               r_we,
    input  wire [5:0]         r_addr,
    input  wire [8:0]         r_data,
    input  wire [31:0]        seed,
    input  wire               start,
    output reg                busy,
    output reg                done,
    output reg  [15:0]        cls,
    output reg  signed [31:0] cls_logit,
    // w_out-Stream (SDRAM)
    output reg                wout_req,
    output reg  [23:0]        wout_addr,
    input  wire               wout_valid,
    input  wire [31:0]        wout_data,
    // Debug / Verifikation
    output reg                step_valid,
    output wire [799:0]       step_spikes,
    output reg                logit_valid,
    output reg  [15:0]        logit_idx,
    output reg  signed [31:0] logit_data,
    // Zaehler-Ausgabe (READOUT = 0)
    output reg                cnt_valid,
    output reg  [7:0]         cnt_idx,
    output reg  [27:0]        cnt_word
);

`include "params.vh"

    localparam N_WIN  = N_IN / 2;          // 18-Bit-Woerter w_in pro Neuron
    localparam N_WOUT = N_HID / 4;         // 32-Bit-Woerter w_out pro Klasse
    localparam N_PTR  = 2 * N_HID + 1;
    localparam signed [15:0] KM = K_MEM;
    localparam signed [15:0] KS = K_SYN;
    localparam [31:0] GOLDEN = 32'h9E3779B9;

    // ------------------------------------------------------------ Speicher (BSRAM)
    reg [13:0] ptr_mem [0:N_PTR-1];
    reg [17:0] syn_mem [0:N_SYN-1];
    reg [17:0] win_mem [0:N_HID*N_WIN-1];
    reg [31:0] bout_mem[0:N_CLS-1];
    reg signed [15:0] v_mem [0:N_HID-1];
    reg signed [15:0] i_mem [0:N_HID-1];
    reg [27:0] cnt_mem [0:N_HID/4-1];      // 4 Zaehler a 7 Bit pro Wort

    initial begin
        $readmemh({MEM_DIR, "ptr.hex"},   ptr_mem);
        $readmemh({MEM_DIR, "syn.hex"},   syn_mem);
        $readmemh({MEM_DIR, "w_in.hex"},  win_mem);
        $readmemh({MEM_DIR, "b_out.hex"}, bout_mem);
    end

    reg  [8:0]  r_reg [0:N_IN-1];
    reg  [N_IN-1:0] e;

    // ------------------------------------------------------------ Zustand
    localparam S_IDLE = 4'd0, S_GEN = 4'd1, S_N0 = 4'd2, S_N1 = 4'd3, S_N2 = 4'd4, S_N3 = 4'd5,
               S_MAC = 4'd6, S_LIF1 = 4'd7, S_LIF2 = 4'd8, S_STEP = 4'd9, S_RUN = 4'd10,
               S_RO0 = 4'd11, S_RO1 = 4'd12, S_RO2 = 4'd13, S_CO = 4'd14;
    reg [3:0]  state;

    reg [31:0] x, run_seed;
    reg [5:0]  k;
    reg [9:0]  ni;                         // Neuron
    reg [4:0]  step;
    reg [2:0]  run;
    reg [N_HID-1:0] s_prev, s_cur;
    assign step_spikes = s_prev;

    reg [13:0] p0, p1, p2, a_iss, a_d;
    reg [4:0]  c_iss, c_d;
    reg        win_v, syn_v;
    reg [14:0] win_base;
    reg signed [23:0] acc_in;
    reg signed [19:0] acc_rec;
    reg signed [15:0] i_new_r;
    reg signed [16:0] dv_r;

    reg [9:0]  cls_i;                      // Readout-Klasse
    reg [7:0]  jcnt, jmac;
    reg [31:0] wd_r;
    reg        wv_r;
    reg signed [31:0] acc_o;
    reg        co_v;
    reg [7:0]  co_idx;

    // ------------------------------------------------------------ Leseports (1 Takt Latenz)
    wire        in_ro     = (state == S_RO0) || (state == S_RO1) || (state == S_RO2) || (state == S_CO);
    wire [13:0] ptr_raddr = (state == S_N0) ? {ni, 1'b0} :
                            (state == S_N1) ? {ni, 1'b0} + 14'd1 : {ni, 1'b0} + 14'd2;
    wire [14:0] win_raddr = win_base + c_iss;
    wire [7:0]  cnt_raddr = in_ro ? jcnt : ni[9:2];
    reg  [13:0] ptr_q;
    reg  [17:0] syn_q, win_q;
    reg  signed [15:0] v_q, i_q;
    reg  [27:0] cnt_q;
    reg  [31:0] bout_q;

    always @(posedge clk) begin
        ptr_q  <= ptr_mem[ptr_raddr];
        syn_q  <= syn_mem[a_iss];
        win_q  <= win_mem[win_raddr];
        v_q    <= v_mem[ni];
        i_q    <= i_mem[ni];
        cnt_q  <= cnt_mem[cnt_raddr];
        bout_q <= bout_mem[cls_i];
    end

    // ------------------------------------------------------------ Eingangsspikes
    wire [31:0] x1 = x  ^ (x  << 13);
    wire [31:0] x2 = x1 ^ (x1 >> 17);
    wire [31:0] xn = x2 ^ (x2 << 5);

    // ------------------------------------------------------------ MAC-Pfade
    wire [5:0]  k0 = {c_d, 1'b0};
    wire [5:0]  k1 = {c_d, 1'b1};
    wire signed [8:0]  w0 = win_q[8:0];
    wire signed [8:0]  w1 = win_q[17:9];
    wire signed [9:0]  r0 = {1'b0, r_reg[k0]};
    wire signed [9:0]  r1 = {1'b0, r_reg[k1]};
    wire signed [18:0] m0 = e[k0] ? w0 * r0 : 19'sd0;
    wire signed [18:0] m1 = e[k1] ? w1 * r1 : 19'sd0;

    wire        half = (a_d >= p1);
    wire [9:0]  pre  = half ? HALF + syn_q[17:9] : {1'b0, syn_q[17:9]};
    wire signed [8:0] wr = syn_q[8:0];

    // ------------------------------------------------------------ LIF (Norse 1.1.0)
    wire signed [23:0] cur_in, cur_rec;
    generate
        if (SH_IN > 0) begin : g_in
            assign cur_in = (acc_in + (24'sd1 <<< (SH_IN - 1))) >>> SH_IN;
        end else begin : g_in_l
            assign cur_in = acc_in <<< (-SH_IN);
        end
        if (SH_REC > 0) begin : g_rec
            assign cur_rec = (acc_rec + (20'sd1 <<< (SH_REC - 1))) >>> SH_REC;
        end else begin : g_rec_l
            assign cur_rec = acc_rec <<< (-SH_REC);
        end
    endgenerate

    wire               first_step = (step == 5'd0);
    wire signed [15:0] v_eff = first_step ? 16'sd0 : v_q;
    wire signed [15:0] i_eff = first_step ? 16'sd0 : i_q;

    function signed [15:0] sat16(input signed [24:0] val);
        sat16 = (val > 25'sd32767) ? 16'sh7fff : (val < -25'sd32768) ? 16'sh8000 : val[15:0];
    endfunction

    // Takt LIF1: i_new = sat(i + cur), dv = i_new - v
    wire signed [24:0] i_sum  = i_eff + cur_in + cur_rec;
    wire signed [15:0] i_new  = sat16(i_sum);
    // Takt LIF2: v_dec = sat(v + mulq16(K_MEM, dv)), i = i_new - mulq16(K_SYN, i_new)
    wire signed [33:0] pm     = KM * dv_r;
    wire signed [31:0] ps     = KS * i_new_r;
    wire signed [17:0] dmem   = (pm + 34'sd32768) >>> 16;
    wire signed [15:0] dsyn   = (ps + 32'sd32768) >>> 16;
    wire signed [24:0] v_sum  = v_eff + dmem;
    wire signed [15:0] v_dec  = sat16(v_sum);
    wire signed [15:0] i_out  = i_new_r - dsyn;
    wire               z      = v_dec > V_TH;

    // Zaehler-Lane des aktuellen Neurons (im ersten Schritt der Inferenz als 0 behandelt)
    wire [1:0]  lane    = ni[1:0];
    wire [6:0]  cnt_old = (run == 3'd0 && first_step) ? 7'd0 : cnt_q[7*lane +: 7];
    reg  [27:0] cnt_new;
    always @(*) begin
        cnt_new = cnt_q;
        cnt_new[7*lane +: 7] = cnt_old + {6'd0, z};
    end

    // ------------------------------------------------------------ Readout-MAC (4 Lanes)
    wire signed [31:0] mac4 =
        $signed({1'b0, cnt_q[ 6: 0]}) * $signed(wd_r[ 7: 0]) +
        $signed({1'b0, cnt_q[13: 7]}) * $signed(wd_r[15: 8]) +
        $signed({1'b0, cnt_q[20:14]}) * $signed(wd_r[23:16]) +
        $signed({1'b0, cnt_q[27:21]}) * $signed(wd_r[31:24]);
    wire signed [31:0] logit = acc_o + $signed(bout_q);

    // ------------------------------------------------------------ Steuerung
    always @(posedge clk) begin
        done        <= 1'b0;
        wout_req    <= 1'b0;
        step_valid  <= 1'b0;
        logit_valid <= 1'b0;
        cnt_valid   <= 1'b0;

        if (r_we) r_reg[r_addr] <= r_data;

        if (rst) begin
            state <= S_IDLE;
            busy  <= 1'b0;
        end else begin
            case (state)
            S_IDLE: if (start) begin
                busy     <= 1'b1;
                run      <= 3'd0;
                step     <= 5'd0;
                run_seed <= seed;
                x        <= (seed == 32'd0) ? 32'd1 : seed;
                s_prev   <= {N_HID{1'b0}};
                k        <= 6'd0;
                state    <= S_GEN;
            end

            S_GEN: begin
                x    <= xn;
                e[k] <= (xn[31:24] < r_reg[k]);
                k    <= k + 6'd1;
                if (k == N_IN - 1) begin
                    ni       <= 10'd0;
                    win_base <= 15'd0;
                    state    <= S_N0;
                end
            end

            S_N0: begin
                acc_in  <= 24'sd0;
                acc_rec <= 20'sd0;
                state   <= S_N1;
            end
            S_N1: begin p0 <= ptr_q; state <= S_N2; end
            S_N2: begin p1 <= ptr_q; state <= S_N3; end
            S_N3: begin
                p2    <= ptr_q;
                a_iss <= p0;
                c_iss <= 5'd0;
                win_v <= 1'b0;
                syn_v <= 1'b0;
                state <= S_MAC;
            end

            S_MAC: begin
                win_v <= (c_iss < N_WIN);
                c_d   <= c_iss;
                if (c_iss < N_WIN) c_iss <= c_iss + 5'd1;
                syn_v <= (a_iss < p2);
                a_d   <= a_iss;
                if (a_iss < p2) a_iss <= a_iss + 14'd1;

                if (win_v) acc_in <= acc_in + m0 + m1;
                if (syn_v && s_prev[pre]) acc_rec <= acc_rec + wr;

                if (c_iss == N_WIN && a_iss >= p2 && !win_v && !syn_v) state <= S_LIF1;
            end

            S_LIF1: begin
                i_new_r <= i_new;
                dv_r    <= i_new - v_eff;
                state   <= S_LIF2;
            end

            S_LIF2: begin
                v_mem[ni]          <= z ? 16'sd0 : v_dec;
                i_mem[ni]          <= i_out;
                cnt_mem[ni[9:2]]   <= cnt_new;
                s_cur[ni]          <= z;
                win_base           <= win_base + N_WIN;
                if (ni == N_HID - 1) state <= S_STEP;
                else begin
                    ni    <= ni + 10'd1;
                    state <= S_N0;
                end
            end

            S_STEP: begin
                s_prev     <= s_cur;
                step_valid <= 1'b1;            // step_spikes gilt im naechsten Takt
                k          <= 6'd0;
                if (step == T_STEPS - 1) state <= S_RUN;
                else begin
                    step  <= step + 5'd1;
                    state <= S_GEN;
                end
            end

            S_RUN: begin
                s_prev <= {N_HID{1'b0}};
                step   <= 5'd0;
                if (run == N_RUNS - 1) begin
                    cls_i     <= 10'd0;
                    wout_addr <= 24'd0;
                    jcnt      <= 8'd0;
                    co_v      <= 1'b0;
                    state     <= READOUT ? S_RO0 : S_CO;
                end else begin
                    run      <= run + 3'd1;
                    run_seed <= run_seed + GOLDEN;
                    x        <= (run_seed + GOLDEN == 32'd0) ? 32'd1 : run_seed + GOLDEN;
                    state    <= S_GEN;
                end
            end

            S_RO0: begin                       // Burst fuer Klasse cls_i anfordern
                wout_req <= 1'b1;
                jcnt     <= 8'd0;
                jmac     <= 8'd0;
                wv_r     <= 1'b0;
                acc_o    <= 32'sd0;
                state    <= S_RO1;
            end

            S_RO1: begin
                wv_r <= wout_valid;
                wd_r <= wout_data;
                if (wout_valid) jcnt <= jcnt + 8'd1;
                if (wv_r) begin
                    acc_o <= acc_o + mac4;
                    jmac  <= jmac + 8'd1;
                    if (jmac == N_WOUT - 1) state <= S_RO2;
                end
            end

            S_RO2: begin
                logit_valid <= 1'b1;
                logit_idx   <= {6'd0, cls_i};
                logit_data  <= logit;
                if (cls_i == 10'd0 || logit > cls_logit) begin
                    cls       <= {6'd0, cls_i};
                    cls_logit <= logit;
                end
                wout_addr <= wout_addr + N_WOUT;
                if (cls_i == N_CLS - 1) begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end else begin
                    cls_i <= cls_i + 10'd1;
                    state <= S_RO0;
                end
            end

            S_CO: begin                        // Zaehlerwoerter ausgeben (1 Takt Leselatenz)
                co_v   <= (jcnt < N_WOUT);
                co_idx <= jcnt;
                if (jcnt < N_WOUT) jcnt <= jcnt + 8'd1;
                if (co_v) begin
                    cnt_valid <= 1'b1;
                    cnt_idx   <= co_idx;
                    cnt_word  <= cnt_q;
                end
                if (jcnt == N_WOUT && !co_v) begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule
