// tb_snn_core.v - vergleicht snn_core Schritt fuer Schritt mit tools/snn_ref.py
//
// Erwartet im Include-/Arbeitsverzeichnis (MEM_DIR) die Dateien aus export_variant_e.py:
// params.vh, tv.vh, *.hex. SDRAM-Modell: w_out.hex, Latenz SD_LAT Takte, zufaellige Luecken.
`timescale 1ns/1ps

module tb_snn_core;
`include "params.vh"
`include "tv.vh"

    parameter MEM_DIR = "";
    localparam SD_LAT = 6;
    localparam N_WOUT = N_HID / 4;
    localparam N_STEPS_TV = N_TV * N_RUNS * T_STEPS;

    reg clk = 1'b0, rst = 1'b1;
    always #9.259 clk = ~clk;                  // 54 MHz

    reg         r_we = 1'b0;
    reg  [5:0]  r_addr;
    reg  [8:0]  r_data;
    reg  [31:0] seed;
    reg         start = 1'b0;
    wire        busy, done, wout_req, step_valid, logit_valid;
    wire [15:0] cls, logit_idx;
    wire signed [31:0] cls_logit, logit_data;
    wire [23:0] wout_addr;
    reg         wout_valid = 1'b0;
    reg  [31:0] wout_data;
    wire [799:0] step_spikes;

    snn_core #(.MEM_DIR(MEM_DIR)) dut (
        .clk(clk), .rst(rst), .r_we(r_we), .r_addr(r_addr), .r_data(r_data), .seed(seed),
        .start(start), .busy(busy), .done(done), .cls(cls), .cls_logit(cls_logit),
        .wout_req(wout_req), .wout_addr(wout_addr), .wout_valid(wout_valid), .wout_data(wout_data),
        .step_valid(step_valid), .step_spikes(step_spikes),
        .logit_valid(logit_valid), .logit_idx(logit_idx), .logit_data(logit_data),
        .cnt_valid(), .cnt_idx(), .cnt_word(), .dbg_addr(8'd0), .dbg_word());

    // ------------------------------------------------------------ Testvektoren
    reg [8:0]   tv_r      [0:N_TV*N_IN-1];
    reg [31:0]  tv_seed   [0:N_TV-1];
    reg [799:0] tv_spikes [0:N_STEPS_TV-1];
    reg [31:0]  tv_logits [0:N_TV*N_CLS-1];
    reg [15:0]  tv_class  [0:N_TV-1];
    reg [31:0]  wout_mem  [0:N_CLS*N_WOUT-1];

    initial begin
        $readmemh({MEM_DIR, "tv_r.hex"},      tv_r);
        $readmemh({MEM_DIR, "tv_seed.hex"},   tv_seed);
        $readmemh({MEM_DIR, "tv_spikes.hex"}, tv_spikes);
        $readmemh({MEM_DIR, "tv_logits.hex"}, tv_logits);
        $readmemh({MEM_DIR, "tv_class.hex"},  tv_class);
        $readmemh({MEM_DIR, "w_out.hex"},     wout_mem);
    end

    // ------------------------------------------------------------ SDRAM-Modell (Burst, Luecken)
    integer sd_wait = -1, sd_cnt = 0;
    reg [23:0] sd_addr;
    reg [31:0] lfsr = 32'h1234_5678;
    always @(posedge clk) begin
        lfsr       <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
        wout_valid <= 1'b0;
        if (wout_req) begin
            sd_addr <= wout_addr;
            sd_wait <= SD_LAT;
            sd_cnt  <= 0;
        end else if (sd_wait > 0) begin
            sd_wait <= sd_wait - 1;
        end else if (sd_wait == 0 && sd_cnt < N_WOUT && lfsr[3:0] != 4'd0) begin  // ~6 % Luecken
            wout_valid <= 1'b1;
            wout_data  <= wout_mem[sd_addr + sd_cnt];
            sd_cnt     <= sd_cnt + 1;
        end
    end

    // ------------------------------------------------------------ Vergleich
    integer sample = 0, step_idx = 0, err_spk = 0, err_logit = 0, err_cls = 0, n_spk_steps = 0;
    integer cyc = 0, t_start = 0;
    always @(posedge clk) cyc <= cyc + 1;

    always @(posedge clk) begin
        if (step_valid) begin
            if (step_spikes !== tv_spikes[step_idx]) begin
                if (err_spk < 5)
                    $display("SPIKE-FEHLER Probe %0d Schritt %0d (Lauf %0d, t %0d): %0d statt %0d Spikes",
                             sample, step_idx % (N_RUNS*T_STEPS), (step_idx % (N_RUNS*T_STEPS)) / T_STEPS,
                             step_idx % T_STEPS, $countones(step_spikes), $countones(tv_spikes[step_idx]));
                err_spk = err_spk + 1;
            end
            n_spk_steps = n_spk_steps + 1;
            step_idx = step_idx + 1;
        end
        if (logit_valid) begin
            if (logit_data !== $signed(tv_logits[sample*N_CLS + logit_idx])) begin
                if (err_logit < 5)
                    $display("LOGIT-FEHLER Probe %0d Klasse %0d: %0d statt %0d", sample, logit_idx,
                             logit_data, $signed(tv_logits[sample*N_CLS + logit_idx]));
                err_logit = err_logit + 1;
            end
        end
    end

    integer b, kk;
    initial begin
        // Eingaenge auf der fallenden Flanke treiben, damit nichts mit der steigenden kollidiert
        repeat (4) @(negedge clk);
        rst = 1'b0;
        for (b = 0; b < N_TV; b = b + 1) begin
            for (kk = 0; kk < N_IN; kk = kk + 1) begin
                @(negedge clk);
                r_we = 1'b1; r_addr = kk; r_data = tv_r[b*N_IN + kk];
            end
            @(negedge clk);
            r_we  = 1'b0;
            seed  = tv_seed[b];
            start = 1'b1;
            t_start = cyc;
            @(negedge clk);
            start = 1'b0;
            @(posedge done);
            @(negedge clk);                    // letztes logit_valid ist geprueft
            @(negedge clk);
            if (cls !== tv_class[b]) err_cls = err_cls + 1;
            $display("Probe %0d: Klasse %0d (erwartet %0d), Logit %0d, %0d Takte = %.2f ms bei 54 MHz",
                     b, cls, tv_class[b], cls_logit, cyc - t_start, (cyc - t_start) / 54.0e3);
            sample = sample + 1;
        end
        $display("ERGEBNIS: %0d Schritte geprueft, Spike-Fehler %0d, Logit-Fehler %0d, Klassen-Fehler %0d",
                 n_spk_steps, err_spk, err_logit, err_cls);
        if (err_spk == 0 && err_logit == 0 && err_cls == 0 && n_spk_steps == N_STEPS_TV)
            $display("PASS");
        else
            $display("FAIL");
        $finish;
    end
endmodule
