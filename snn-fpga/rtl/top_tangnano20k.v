// top_tangnano20k.v - Bring-up von Variante E auf dem Tang Nano 20K (ohne SDRAM)
//
// Der FPGA rechnet das SNN (alle Monte-Carlo-Laeufe) und schickt die 800 Spikezaehler per UART
// zurueck; der PC (tools/board_test.py) vergleicht sie mit snn_ref.py und rechnet den Readout.
//
// Protokoll (8N1, BAUD):
//   PC -> FPGA:  0xA5, seed[4] (little endian), 52 x r[2] (little endian, 0..256)
//   FPGA -> PC:  0x5A, 800 x count[1], Pruefsumme (Summe der Zaehler mod 256)
// LEDs (aktiv low): LED0 = rechnet, LED1 = Herzschlag, LED2 = Frame empfangen, LED3 = Antwort gesendet

module top_tangnano20k #(
    parameter MEM_DIR = "",
    parameter CLK_HZ  = 27_000_000,
    parameter BAUD    = 115_200
) (
    input  wire       clk,                     // 27 MHz
    input  wire       uart_rx,
    output wire       uart_tx,
    output wire [5:0] led
);
    localparam CPB = CLK_HZ / BAUD;

    // Power-on-Reset
    reg [7:0] por = 8'd0;
    wire rst = ~por[7];
    always @(posedge clk) if (!por[7]) por <= por + 8'd1;

    // ------------------------------------------------------------ UART
    wire       rx_valid;
    wire [7:0] rx_data;
    reg        tx_send = 1'b0;
    reg  [7:0] tx_data;
    wire       tx_ready;
    uart_rx #(.CLKS_PER_BIT(CPB)) u_rx (.clk(clk), .rx(uart_rx), .valid(rx_valid), .data(rx_data));
    uart_tx #(.CLKS_PER_BIT(CPB)) u_tx (.clk(clk), .send(tx_send), .data(tx_data), .ready(tx_ready), .tx(uart_tx));

    // ------------------------------------------------------------ Kern
    reg        r_we = 1'b0, start = 1'b0;
    reg  [5:0] r_addr;
    reg  [8:0] r_data;
    reg  [31:0] seed;
    wire       busy, done, cnt_valid;
    wire [7:0] cnt_idx;
    wire [27:0] cnt_word;

    snn_core #(.MEM_DIR(MEM_DIR), .READOUT(0)) u_core (
        .clk(clk), .rst(rst), .r_we(r_we), .r_addr(r_addr), .r_data(r_data), .seed(seed),
        .start(start), .busy(busy), .done(done), .cls(), .cls_logit(),
        .wout_req(), .wout_addr(), .wout_valid(1'b0), .wout_data(32'd0),
        .step_valid(), .step_spikes(), .logit_valid(), .logit_idx(), .logit_data(),
        .cnt_valid(cnt_valid), .cnt_idx(cnt_idx), .cnt_word(cnt_word),
        .dbg_addr(8'd0), .dbg_word());

    // Zaehlerpuffer: 200 Woerter a 4 x 7 Bit
    reg [27:0] cbuf [0:199];
    always @(posedge clk) if (cnt_valid) cbuf[cnt_idx] <= cnt_word;

    // ------------------------------------------------------------ Ablauf
    localparam P_SYNC = 3'd0, P_SEED = 3'd1, P_RLO = 3'd2, P_RHI = 3'd3, P_RUN = 3'd4,
               P_HDR = 3'd5, P_CNT = 3'd6, P_SUM = 3'd7;
    reg [2:0]  ps = P_SYNC;
    reg [1:0]  nb;
    reg [5:0]  kr;
    reg [7:0]  rlo;
    reg [9:0]  ci;                             // Neuron beim Senden
    reg [7:0]  sum;
    reg [27:0] cw;
    reg        rx_led = 1'b1, tx_led = 1'b1;

    wire [6:0] cnt_lane = cw[7*ci[1:0] +: 7];

    always @(posedge clk) begin
        r_we    <= 1'b0;
        start   <= 1'b0;
        tx_send <= 1'b0;
        cw      <= cbuf[ci[9:2]];
        if (rst) ps <= P_SYNC;
        else case (ps)
        P_SYNC: if (rx_valid && rx_data == 8'hA5) begin nb <= 2'd0; ps <= P_SEED; end
        P_SEED: if (rx_valid) begin
            seed <= {rx_data, seed[31:8]};
            nb   <= nb + 2'd1;
            if (nb == 2'd3) begin kr <= 6'd0; ps <= P_RLO; end
        end
        P_RLO: if (rx_valid) begin rlo <= rx_data; ps <= P_RHI; end
        P_RHI: if (rx_valid) begin
            r_we   <= 1'b1;
            r_addr <= kr;
            r_data <= {rx_data[0], rlo};
            kr     <= kr + 6'd1;
            if (kr == 6'd51) begin
                start  <= 1'b1;
                rx_led <= ~rx_led;
                ps     <= P_RUN;
            end else ps <= P_RLO;
        end
        P_RUN: if (done) ps <= P_HDR;
        P_HDR: if (tx_ready && !tx_send) begin
            tx_data <= 8'h5A;
            tx_send <= 1'b1;
            ci      <= 10'd0;
            sum     <= 8'd0;
            ps      <= P_CNT;
        end
        P_CNT: if (tx_ready && !tx_send) begin    // cw ist seit mind. 1 Takt gueltig
            tx_data <= {1'b0, cnt_lane};
            tx_send <= 1'b1;
            sum     <= sum + {1'b0, cnt_lane};
            ci      <= ci + 10'd1;
            if (ci == 10'd799) ps <= P_SUM;
        end
        P_SUM: if (tx_ready && !tx_send) begin
            tx_data <= sum;
            tx_send <= 1'b1;
            tx_led  <= ~tx_led;
            ps      <= P_SYNC;
        end
        endcase
    end

    // Herzschlag ~1,6 Hz
    reg [23:0] hb = 24'd0;
    always @(posedge clk) hb <= hb + 24'd1;

    assign led = {2'b11, tx_led, rx_led, ~hb[23], ~busy};
endmodule
