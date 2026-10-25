// uart.v - 8N1-UART, CLKS_PER_BIT = Takt / Baudrate

module uart_rx #(parameter CLKS_PER_BIT = 234) (
    input  wire       clk,
    input  wire       rx,
    output reg        valid,
    output reg  [7:0] data
);
    reg [1:0]  sync = 2'b11;
    reg [15:0] cnt  = 16'd0;
    reg [3:0]  bitn = 4'd0;
    reg        busy = 1'b0;
    always @(posedge clk) begin
        sync  <= {sync[0], rx};
        valid <= 1'b0;
        if (!busy) begin
            if (!sync[1]) begin                       // Startbit
                busy <= 1'b1;
                cnt  <= CLKS_PER_BIT / 2;
                bitn <= 4'd0;
            end
        end else if (cnt != 0) begin
            cnt <= cnt - 16'd1;
        end else begin
            cnt <= CLKS_PER_BIT - 1;
            if (bitn == 4'd0) begin
                if (sync[1]) busy <= 1'b0;            // Startbit war Stoerung
            end else if (bitn <= 4'd8) begin
                data <= {sync[1], data[7:1]};
            end else begin
                busy  <= 1'b0;
                valid <= sync[1];                     // Stoppbit muss 1 sein
            end
            bitn <= bitn + 4'd1;
        end
    end
endmodule

module uart_tx #(parameter CLKS_PER_BIT = 234) (
    input  wire       clk,
    input  wire       send,
    input  wire [7:0] data,
    output wire       ready,
    output wire       tx
);
    reg [9:0]  sh   = 10'h3ff;
    reg [15:0] cnt  = 16'd0;
    reg [3:0]  left = 4'd0;
    assign ready = (left == 0);
    assign tx    = sh[0];
    always @(posedge clk) begin
        if (left == 0) begin
            if (send) begin
                sh   <= {1'b1, data, 1'b0};
                left <= 4'd10;
                cnt  <= CLKS_PER_BIT - 1;
            end
        end else if (cnt != 0) begin
            cnt <= cnt - 16'd1;
        end else begin
            cnt  <= CLKS_PER_BIT - 1;
            sh   <= {1'b1, sh[9:1]};
            left <= left - 4'd1;
        end
    end
endmodule
