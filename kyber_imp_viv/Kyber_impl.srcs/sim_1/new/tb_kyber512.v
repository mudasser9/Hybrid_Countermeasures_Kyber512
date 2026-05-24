`timescale 1ns / 1ps

module tb_kyber512;

reg clk;
reg rst_n;
reg start;
reg trng_valid;

reg [3071:0] sk_poly_0;
reg [3071:0] sk_poly_1;

reg [2559:0] ct_u_0;
reg [2559:0] ct_u_1;

reg [1023:0] ct_v;

reg [127:0] trng_seed;

wire [255:0] m_prime;
wire done;
wire valid;
wire ct_match;

// DUT
kyber512_decap_hybrid DUT (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .done(done),
    .valid(valid),

    .sk_poly_0(sk_poly_0),
    .sk_poly_1(sk_poly_1),

    .ct_u_0(ct_u_0),
    .ct_u_1(ct_u_1),

    .ct_v(ct_v),

    .trng_seed(trng_seed),
    .trng_valid(trng_valid),

    .m_prime(m_prime),
    .ct_match(ct_match)
);

// CLOCK
always #5 clk = ~clk;

// TEST
initial begin

    clk = 0;
    rst_n = 0;
    start = 0;
    trng_valid = 0;

    sk_poly_0 = 0;
    sk_poly_1 = 0;

    ct_u_0 = 0;
    ct_u_1 = 0;

    ct_v = 0;

    trng_seed = 128'h123456789ABCDEF00112233445566778;

    // RESET
    #20;
    rst_n = 1;

    // VALID RNG
    #20;
    trng_valid = 1;

    // START
    #20;
    start = 1;

    #10;
    start = 0;

    // WAIT
    wait(done == 1);

    #20;

    $display("--------------------------------");
    $display("DECAP DONE");
    $display("m_prime  = %h", m_prime);
    $display("ct_match = %b", ct_match);
    $display("--------------------------------");

    #50;
    $finish;

end

// WAVES
initial begin

    $dumpfile("kyber512.vcd");
    $dumpvars(0, tb_kyber512);

end

endmodule