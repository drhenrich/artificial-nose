// tb_top.v - top_tangnano20k ueber UART (Bitebene): Frame senden, 802 Bytes empfangen,
// Zaehler mit tv_counts.hex vergleichen. Schnelle Baudrate (CPB = 8), damit die Simulation kurz bleibt.
`timescale 1ns/1ps

module tb_top;
`include "params.vh"
`include "tv.vh"

    parameter MEM_DIR = "";
    localparam CPB = 8;

    reg clk = 1'b0;
    always #18.518 clk = ~clk;                 // 27 MHz
    reg  rx = 1'b1;
    wire tx;
    wire [5:0] led;

    top_tangnano20k #(.MEM_DIR(MEM_DIR), .CLK_HZ(27_000_000), .BAUD(27_000_000 / CPB)) dut (
        .clk(clk), .uart_rx(rx), .uart_tx(tx), .led(led));

    reg [8:0]  tv_r      [0:N_TV*N_IN-1];
    reg [31:0] tv_seed   [0:N_TV-1];
    reg [7:0]  tv_counts [0:N_TV*N_HID-1];
    initial begin
        $readmemh({MEM_DIR, "tv_r.hex"},      tv_r);
        $readmemh({MEM_DIR, "tv_seed.hex"},   tv_seed);
        $readmemh({MEM_DIR, "tv_counts.hex"}, tv_counts);
    end

    task send_byte(input [7:0] b);
        integer i;
        begin
            rx = 1'b0; repeat (CPB) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin rx = b[i]; repeat (CPB) @(posedge clk); end
            rx = 1'b1; repeat (CPB) @(posedge clk);
        end
    endtask

    reg [7:0] got;
    task recv_byte;
        integer i;
        begin
            @(negedge tx);
            repeat (CPB + CPB / 2) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin got[i] = tx; repeat (CPB) @(posedge clk); end
        end
    endtask

    integer b, k, n, errors = 0, sum;
    initial begin
        repeat (300) @(posedge clk);           // Power-on-Reset abwarten
        for (b = 0; b < N_TV; b = b + 1) begin
            send_byte(8'hA5);
            for (k = 0; k < 4; k = k + 1) send_byte(tv_seed[b] >> (8 * k));
            for (k = 0; k < N_IN; k = k + 1) begin
                send_byte(tv_r[b*N_IN + k][7:0]);
                send_byte({7'd0, tv_r[b*N_IN + k][8]});
            end
            recv_byte;
            if (got !== 8'h5A) begin $display("Kopfbyte %h statt 5a", got); errors = errors + 1; end
            sum = 0;
            for (n = 0; n < N_HID; n = n + 1) begin
                recv_byte;
                sum = sum + got;
                if (got !== tv_counts[b*N_HID + n]) begin
                    if (errors < 5) $display("Probe %0d Neuron %0d: %0d statt %0d", b, n, got, tv_counts[b*N_HID + n]);
                    errors = errors + 1;
                end
            end
            recv_byte;
            if (got !== sum[7:0]) begin $display("Pruefsumme %h statt %h", got, sum[7:0]); errors = errors + 1; end
            $display("Probe %0d: 800 Zaehler empfangen, Summe %0d", b, sum);
        end
        if (errors == 0) $display("PASS"); else $display("FAIL (%0d Fehler)", errors);
        $finish;
    end
endmodule
