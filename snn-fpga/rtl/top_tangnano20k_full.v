// top_tangnano20k_full.v - Variante E vollstaendig auf dem Tang Nano 20K
// (54 MHz aus der PLL, w_out im eingebetteten SDRAM, Readout und Argmax auf dem FPGA)
//
// Protokoll (8N1, BAUD), alle Mehrbytewerte little endian:
//   PC -> FPGA  0xA5 'W' addr[3] n[2] daten[4n]   n Woerter ab addr ins SDRAM schreiben
//               Antwort 0x5A 'W' summe[4]           (Summe der Woerter mod 2^32)
//   PC -> FPGA  0xA5 'R' addr[3] n[2]              n Woerter lesen
//               Antwort 0x5A 'R' daten[4n]
//   PC -> FPGA  0xA5 'C' seed[4] r[2 x 52]         klassifizieren (w_out muss geladen sein)
//               Antwort 0x5A 'C' klasse[2] logit_max[4] logits[4 x N_CLS] summe[4]
//   Diagnose:
//   PC -> FPGA  0xA5 'L' lat                       SDRAM-Leselatenz setzen (2..7, Start 3)
//               Antwort 0x5A 'L' lat
//   PC -> FPGA  0xA5 'B' addr[3] n[2]              n <= 256 Woerter als EIN Burst lesen (wie Readout)
//               Antwort 0x5A 'B' daten[4n]
//   PC -> FPGA  0xA5 'Z'                           Spikezaehler der letzten Klassifikation
//               Antwort 0x5A 'Z' zaehler[N_HID] summe[1]
// LEDs (aktiv low): LED0 rechnet, LED1 Herzschlag, LED2 SDRAM bereit, LED3 wechselt je Befehl

module top_tangnano20k_full #(
    parameter MEM_DIR = "",
    parameter CLK_HZ  = 54_000_000,
    parameter BAUD    = 1_000_000,
    parameter SIM     = 0                      // 1: Systemtakt direkt (Simulation, keine PLL)
) (
    input  wire        clk,                    // 27 MHz
    input  wire        uart_rx,
    output wire        uart_tx,
    output wire [5:0]  led,
    output wire        O_sdram_clk,
    output wire        O_sdram_cke,
    output wire        O_sdram_cs_n,
    output wire        O_sdram_ras_n,
    output wire        O_sdram_cas_n,
    output wire        O_sdram_wen_n,
    output wire [3:0]  O_sdram_dqm,
    output wire [10:0] O_sdram_addr,
    output wire [1:0]  O_sdram_ba,
    inout  wire [31:0] IO_sdram_dq
);
`include "params.vh"
    localparam CPB    = CLK_HZ / BAUD;
    localparam N_WOUT = N_HID / 4;

    // ------------------------------------------------------------ Takt und Reset
    wire sclk, locked;
    generate
        if (SIM) begin : g_sim
            assign sclk = clk; assign locked = 1'b1;
        end else begin : g_pll
            rPLL #(
                .FCLKIN("27"), .DEVICE("GW2AR-18C"),
                .IDIV_SEL(0), .FBDIV_SEL(1), .ODIV_SEL(16)      // 27 -> 54 MHz (VCO 864 MHz)
            ) u_pll (
                .CLKIN(clk), .CLKOUT(sclk), .LOCK(locked), .CLKOUTP(), .CLKOUTD(), .CLKOUTD3(),
                .RESET(1'b0), .RESET_P(1'b0), .CLKFB(1'b0), .FBDSEL(6'b0), .IDSEL(6'b0),
                .ODSEL(6'b0), .PSDA(4'b0), .DUTYDA(4'b0), .FDLY(4'b0));
        end
    endgenerate

    reg [7:0] por = 8'd0;
    wire rst = ~por[7];
    always @(posedge sclk) if (!locked) por <= 8'd0; else if (!por[7]) por <= por + 8'd1;

    // ------------------------------------------------------------ UART
    wire       rx_valid;
    wire [7:0] rx_data;
    reg        tx_send = 1'b0;
    reg  [7:0] tx_data;
    wire       tx_ready;
    uart_rx #(.CLKS_PER_BIT(CPB)) u_rx (.clk(sclk), .rx(uart_rx), .valid(rx_valid), .data(rx_data));
    uart_tx #(.CLKS_PER_BIT(CPB)) u_tx (.clk(sclk), .send(tx_send), .data(tx_data), .ready(tx_ready), .tx(uart_tx));

    // ------------------------------------------------------------ SDRAM
    wire        sd_init, sd_rvalid, sd_wdone;
    wire [31:0] sd_rdata;
    reg         t_rreq = 1'b0, sd_wreq = 1'b0;
    reg  [20:0] t_raddr, sd_waddr;
    reg  [31:0] sd_wdata;
    wire        c_rreq;
    wire [23:0] c_raddr;
    reg         classify = 1'b0;                 // SDRAM-Leseport gehoert dem Kern
    reg  [8:0]  t_rlen = 9'd1;
    reg  [2:0]  rd_lat = 3'd3;

    sdram_ctrl #(.CLK_MHZ(CLK_HZ / 1_000_000)) u_sdram (
        .clk(sclk), .rst(rst), .init_done(sd_init), .rd_lat(rd_lat),
        .rd_req(classify ? c_rreq : t_rreq), .rd_addr(classify ? c_raddr[20:0] : t_raddr),
        .rd_len(classify ? N_WOUT[8:0] : t_rlen), .rd_valid(sd_rvalid), .rd_data(sd_rdata),
        .wr_req(sd_wreq), .wr_addr(sd_waddr), .wr_data(sd_wdata), .wr_done(sd_wdone),
        .O_sdram_clk(O_sdram_clk), .O_sdram_cke(O_sdram_cke), .O_sdram_cs_n(O_sdram_cs_n),
        .O_sdram_ras_n(O_sdram_ras_n), .O_sdram_cas_n(O_sdram_cas_n), .O_sdram_wen_n(O_sdram_wen_n),
        .O_sdram_dqm(O_sdram_dqm), .O_sdram_addr(O_sdram_addr), .O_sdram_ba(O_sdram_ba),
        .IO_sdram_dq(IO_sdram_dq));

    // ------------------------------------------------------------ Kern
    reg        r_we = 1'b0, start = 1'b0;
    reg  [5:0] r_addr;
    reg  [8:0] r_data;
    reg  [31:0] seed;
    reg  [7:0] dbg_addr = 8'd0;
    wire [27:0] dbg_word;
    wire       busy, done, logit_valid;
    wire [15:0] cls, logit_idx;
    wire signed [31:0] cls_logit, logit_data;

    snn_core #(.MEM_DIR(MEM_DIR), .READOUT(1)) u_core (
        .clk(sclk), .rst(rst), .r_we(r_we), .r_addr(r_addr), .r_data(r_data), .seed(seed),
        .start(start), .busy(busy), .done(done), .cls(cls), .cls_logit(cls_logit),
        .wout_req(c_rreq), .wout_addr(c_raddr), .wout_valid(classify & sd_rvalid), .wout_data(sd_rdata),
        .step_valid(), .step_spikes(), .logit_valid(logit_valid), .logit_idx(logit_idx),
        .logit_data(logit_data), .cnt_valid(), .cnt_idx(), .cnt_word(),
        .dbg_addr(dbg_addr), .dbg_word(dbg_word));

    // Puffer fuer Logits bzw. Burst-Daten (Diagnose 'B')
    localparam N_BUF = (N_CLS > 256) ? N_CLS : 256;
    reg [31:0] lbuf [0:N_BUF-1];
    reg        bcap = 1'b0;
    reg [9:0]  bidx;
    always @(posedge sclk)
        if (logit_valid) lbuf[logit_idx] <= logit_data;
        else if (bcap && sd_rvalid) lbuf[bidx] <= sd_rdata;

    // ------------------------------------------------------------ Befehlsablauf
    localparam P_SYNC = 5'd0, P_CMD = 5'd1, P_ARG = 5'd2, P_WDAT = 5'd3, P_WWAIT = 5'd4,
               P_RREQ = 5'd5, P_RWAIT = 5'd6, P_RLO = 5'd7, P_RHI = 5'd8, P_RUN = 5'd9,
               P_TX = 5'd10, P_CHDR = 5'd11, P_CLS = 5'd12, P_CMAX = 5'd13, P_CLOG = 5'd14,
               P_CLOG2 = 5'd15, P_CSUM = 5'd16, P_WREP = 5'd17, P_RNEXT = 5'd18,
               P_LSET = 5'd19, P_BREQ = 5'd20, P_BCAP = 5'd21, P_BSEND = 5'd22, P_BSEND2 = 5'd23,
               P_ZRD = 5'd24, P_ZRD2 = 5'd25, P_ZSEND = 5'd26, P_ZSUM = 5'd27;
    reg [4:0]  ps = P_SYNC, tret;
    reg [7:0]  cmd;
    reg [2:0]  na;                               // Argumentbytes
    reg [39:0] arg;
    reg [15:0] n;
    reg [1:0]  nb;
    reg [31:0] word, sum;
    reg [5:0]  kr;
    reg [7:0]  rlo;
    reg [9:0]  li;
    reg [31:0] lw;
    reg [31:0] tw;                               // Sendewort
    reg [2:0]  tn;
    reg        cmd_led = 1'b1;

    always @(posedge sclk) lw <= lbuf[li];

    always @(posedge sclk) begin
        r_we    <= 1'b0;
        start   <= 1'b0;
        tx_send <= 1'b0;
        t_rreq  <= 1'b0;
        sd_wreq <= 1'b0;
        if (rst) begin
            ps <= P_SYNC; classify <= 1'b0;
        end else case (ps)
        P_SYNC: if (rx_valid && rx_data == 8'hA5) ps <= P_CMD;
        P_CMD: if (rx_valid) begin
            cmd <= rx_data; na <= 3'd0; nb <= 2'd0; sum <= 32'd0; kr <= 6'd0;
            cmd_led <= ~cmd_led;
            case (rx_data)
            "W", "R", "B": ps <= P_ARG;
            "C":      ps <= P_ARG;
            "L":      ps <= P_LSET;
            "Z": begin
                dbg_addr <= 8'd0;
                tw <= {16'd0, "Z", 8'h5A}; tn <= 3'd2; tret <= P_ZRD; ps <= P_TX;
            end
            default:  ps <= P_SYNC;
            endcase
        end
        P_ARG: if (rx_valid) begin               // W/R: addr[3] n[2]; C: seed[4]
            arg <= {rx_data, arg[39:8]};
            na  <= na + 3'd1;
            if (cmd == "C" && na == 3'd3) begin
                seed <= {rx_data, arg[39:16]};
                ps   <= P_RLO;
            end else if (cmd != "C" && na == 3'd4) begin
                // bisher empfangen: arg[39:8] = {n_lo, addr[23:16], addr[15:8], addr[7:0]}
                t_raddr  <= arg[28:8];
                sd_waddr <= arg[28:8];
                n        <= {rx_data, arg[39:32]};
                ps       <= (cmd == "W") ? P_WDAT : (cmd == "B") ? P_BREQ : P_RREQ;
            end
        end
        // ---- W: Woerter empfangen und schreiben
        P_WDAT: if (n == 16'd0) begin
            tw <= {sum[15:0], "W", 8'h5A}; tn <= 3'd4; tret <= P_WREP; ps <= P_TX;
        end else if (rx_valid) begin
            word <= {rx_data, word[31:8]};
            nb   <= nb + 2'd1;
            if (nb == 2'd3) begin
                sd_wdata <= {rx_data, word[31:8]};
                sd_wreq  <= 1'b1;
                sum      <= sum + {rx_data, word[31:8]};
                ps       <= P_WWAIT;
            end
        end
        P_WWAIT: if (sd_wdone) begin
            sd_waddr <= sd_waddr + 21'd1;
            n        <= n - 16'd1;
            ps       <= P_WDAT;
        end
        P_WREP: begin tw <= {16'd0, sum[31:16]}; tn <= 3'd2; tret <= P_SYNC; ps <= P_TX; end
        // ---- R: Woerter einzeln lesen und senden
        P_RREQ: if (n == 16'd0) ps <= P_SYNC;
                else begin tw <= {16'd0, "R", 8'h5A}; tn <= 3'd2; tret <= P_RNEXT; ps <= P_TX; end
        P_RNEXT: if (n == 16'd0) ps <= P_SYNC;
                 else begin t_rreq <= 1'b1; ps <= P_RWAIT; end
        P_RWAIT: if (sd_rvalid) begin
            t_raddr <= t_raddr + 21'd1;
            n       <= n - 16'd1;
            tw <= sd_rdata; tn <= 3'd4; tret <= P_RNEXT; ps <= P_TX;
        end
        // ---- L: Leselatenz setzen
        P_LSET: if (rx_valid) begin
            rd_lat <= (rx_data[2:0] < 3'd2) ? 3'd2 : rx_data[2:0];
            tw <= {8'd0, (rx_data[2:0] < 3'd2) ? 8'd2 : {5'd0, rx_data[2:0]}, "L", 8'h5A};
            tn <= 3'd3; tret <= P_SYNC; ps <= P_TX;
        end
        // ---- B: ein Burst von n Woertern in den Puffer, dann senden
        P_BREQ: begin
            t_rlen <= (n == 16'd0 || n > 16'd256) ? 9'd256 : n[8:0];
            n      <= (n == 16'd0 || n > 16'd256) ? 16'd256 : n;
            t_rreq <= 1'b1;
            bcap   <= 1'b1;
            bidx   <= 10'd0;
            ps     <= P_BCAP;
        end
        P_BCAP: if (sd_rvalid) begin
            bidx <= bidx + 10'd1;
            if (bidx + 10'd1 == n[9:0]) begin
                bcap   <= 1'b0;
                t_rlen <= 9'd1;
                li     <= 10'd0;
                tw <= {16'd0, "B", 8'h5A}; tn <= 3'd2; tret <= P_BSEND; ps <= P_TX;
            end
        end
        P_BSEND: if (li == n[9:0]) ps <= P_SYNC; else ps <= P_BSEND2;
        P_BSEND2: begin tw <= lw; tn <= 3'd4; li <= li + 10'd1; tret <= P_BSEND; ps <= P_TX; end
        // ---- Z: Spikezaehler senden (4 Neuronen pro Zaehlerwort)
        P_ZRD:  if (dbg_addr == N_HID / 4 && sum[31]) begin
                    tw <= {24'd0, sum[7:0]}; tn <= 3'd1; tret <= P_SYNC; sum <= 32'd0; ps <= P_TX;
                end else ps <= P_ZRD2;                       // dbg_word ab dem naechsten Takt gueltig
        P_ZRD2: ps <= P_ZSEND;
        P_ZSEND: begin
            tw  <= {1'b0, dbg_word[27:21], 1'b0, dbg_word[20:14], 1'b0, dbg_word[13:7], 1'b0, dbg_word[6:0]};
            tn  <= 3'd4;
            sum <= {1'b1, 23'd0, sum[7:0] + {1'b0, dbg_word[6:0]} + {1'b0, dbg_word[13:7]}
                                + {1'b0, dbg_word[20:14]} + {1'b0, dbg_word[27:21]}};
            dbg_addr <= dbg_addr + 8'd1;
            tret <= P_ZRD; ps <= P_TX;
        end
        // ---- C: Raten empfangen, rechnen, antworten
        P_RLO: if (rx_valid) begin rlo <= rx_data; ps <= P_RHI; end
        P_RHI: if (rx_valid) begin
            r_we   <= 1'b1;
            r_addr <= kr;
            r_data <= {rx_data[0], rlo};
            kr     <= kr + 6'd1;
            if (kr == N_IN - 1) begin
                classify <= 1'b1;
                start    <= 1'b1;
                ps       <= P_RUN;
            end else ps <= P_RLO;
        end
        P_RUN: if (done) begin
            classify <= 1'b0;
            tw <= {16'd0, "C", 8'h5A}; tn <= 3'd2; tret <= P_CLS; ps <= P_TX;
        end
        P_CLS:  begin tw <= {16'd0, cls}; tn <= 3'd2; tret <= P_CMAX; ps <= P_TX; end
        P_CMAX: begin tw <= cls_logit; tn <= 3'd4; tret <= P_CLOG; li <= 10'd0; sum <= 32'd0; ps <= P_TX; end
        P_CLOG: if (li == N_CLS) begin
                    tw <= sum; tn <= 3'd4; tret <= P_SYNC; ps <= P_TX;
                end else ps <= P_CLOG2;            // lw = lbuf[li] ist ab dem naechsten Takt gueltig
        P_CLOG2: begin
            tw <= lw; tn <= 3'd4; sum <= sum + lw; li <= li + 10'd1; tret <= P_CLOG; ps <= P_TX;
        end
        // ---- Bytes senden: tn Bytes aus tw (niederwertigstes zuerst), dann tret
        P_TX: if (tx_ready && !tx_send) begin
            tx_data <= tw[7:0];
            tx_send <= 1'b1;
            tw      <= {8'd0, tw[31:8]};
            tn      <= tn - 3'd1;
            if (tn == 3'd1) ps <= tret;
        end
        default: ps <= P_SYNC;
        endcase
    end

    reg [24:0] hb = 25'd0;
    always @(posedge sclk) hb <= hb + 25'd1;
    assign led = {2'b11, cmd_led, ~sd_init, ~hb[24], ~busy};
endmodule
