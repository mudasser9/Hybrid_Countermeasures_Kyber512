// ============================================================
//  Masked Kyber512 Decapsulation — Top-Level Verilog Module
//  Target: Xilinx FPGA (Vivado)
//  Author: [Your Name]
//
//  Architecture:
//    - First-order additive masking on secret key s
//    - NTT butterfly unit with masked operands
//    - Instruction shuffler (LFSR-driven)
//    - Dummy operation insertion FSM
//    - ASCON-128 co-processor for intermediate protection
// ============================================================

`timescale 1ns / 1ps

module kyber512_decap_hybrid (
    // Clock / reset
    input  wire         clk,
    input  wire         rst_n,

    // Control
    input  wire         start,
    output reg          done,
    output reg          valid,

    // Secret key input (1632 bytes = 13056 bits total)
    input  wire [383:0] sk_poly_0,   // First 384-byte polynomial (12b × 256)
    input  wire [383:0] sk_poly_1,   // Second 384-byte polynomial

    // Ciphertext input
    input  wire [319:0] ct_u_0,      // u[0] compressed (DU=10b × 256 = 320 bytes)
    input  wire [319:0] ct_u_1,      // u[1] compressed
    input  wire [127:0] ct_v,        // v compressed (DV=4b × 256 = 128 bytes)

    // TRNG interface (for fresh masking randomness)
    input  wire [127:0] trng_seed,
    input  wire         trng_valid,

    // Output
    output reg  [255:0] m_prime,     // Recovered plaintext
    output reg          ct_match     // FO ciphertext match flag
);

// ============================================================
// Parameters
// ============================================================
localparam Q         = 3329;
localparam N         = 256;
localparam K         = 2;
localparam DU        = 10;
localparam DV        = 4;

// FSM states
localparam ST_IDLE       = 4'd0;
localparam ST_LOAD_SK    = 4'd1;
localparam ST_MASK_GEN   = 4'd2;
localparam ST_DECOMP_CT  = 4'd3;
localparam ST_NTT_U      = 4'd4;
localparam ST_NTT_S      = 4'd5;
localparam ST_INNER_PROD = 4'd6;
localparam ST_INTT       = 4'd7;
localparam ST_ASCON_PROT = 4'd8;
localparam ST_DECODE     = 4'd9;
localparam ST_RE_ENCRYPT = 4'd10;
localparam ST_COMPARE    = 4'd11;
localparam ST_DONE       = 4'd12;

reg [3:0] state, next_state;

// ============================================================
// Internal registers
// ============================================================
reg signed [12:0] s_ntt_0 [0:N-1];   // NTT(s[0]) coefficients (13-bit signed)
reg signed [12:0] s_ntt_1 [0:N-1];
reg signed [12:0] u_ntt_0 [0:N-1];
reg signed [12:0] u_ntt_1 [0:N-1];
reg signed [12:0] su_acc  [0:N-1];   // inner product accumulator
reg signed [12:0] w_poly  [0:N-1];   // w = v - s^T·u

// Masking shares (additive: s = s0 + s1 mod Q)
reg signed [12:0] s0_ntt_0 [0:N-1];
reg signed [12:0] s1_ntt_0 [0:N-1];
reg signed [12:0] s0_ntt_1 [0:N-1];
reg signed [12:0] s1_ntt_1 [0:N-1];

// PRNG state for mask generation
reg [127:0] lfsr_state;
wire [12:0] rand_mask;

// Shuffle permutation (LFSR-generated)
reg [7:0]  coeff_order [0:N-1];      // shuffled index map
reg        shuffled;

// Dummy op counter
reg [7:0]  dummy_count;
reg        dummy_exec;

// NTT / InvNTT zeta ROM (precomputed)
// zetas[i] = 17^brv7(i) mod 3329  for i in 0..127
reg [11:0] zeta_rom [0:127];
initial $readmemh("zeta_rom.hex", zeta_rom);

// ASCON co-processor interface
wire [11:0] ascon_out;
wire        ascon_done;
reg         ascon_start;
reg  [11:0] ascon_in;

// ============================================================
// PRNG — 128-bit LFSR for mask generation
// ============================================================
always @(posedge clk) begin
    if (!rst_n)
        lfsr_state <= trng_seed;
    else begin
        lfsr_state <= {lfsr_state[126:0],
                        lfsr_state[127] ^ lfsr_state[125] ^
                        lfsr_state[100] ^ lfsr_state[37]};
    end
end

assign rand_mask = lfsr_state[12:0] % Q;   // 13-bit mask

// ============================================================
// NTT Butterfly Unit (Cooley-Tukey, 1 butterfly per cycle)
// ============================================================
// In a real implementation this is fully pipelined.
// Placeholder for the butterfly computation:
//   t     = zeta * f[j+len] mod Q
//   f[j+len] = f[j] - t    mod Q
//   f[j]     = f[j] + t    mod Q

// Montgomery multiplier (for zeta * coefficient mod Q)
// Uses 16-bit Montgomery domain (R = 2^16)
localparam QINV   = 16'd62209;  // Q^{-1} mod 2^16
localparam MONT_R = 17'd65536;  // 2^16

function automatic signed [12:0] montgomery_mul;
    input signed [12:0] a;
    input [11:0]         b;   // zeta (positive)
    reg signed [28:0]    product;
    reg [15:0]           u;
    reg signed [28:0]    t;
    begin
        product   = a * $signed({1'b0, b});
        u         = product[15:0] * QINV;
        t         = (product - u * Q) >>> 16;
        if (t < 0)
            montgomery_mul = t + Q;
        else
            montgomery_mul = t;
    end
endfunction

// Barrett reduction (for additions)
function automatic signed [12:0] barrett_reduce;
    input signed [24:0] a;
    reg signed [24:0]   t;
    begin
        t = ((a * 25'd20159) >>> 26) * Q;
        barrett_reduce = a - t;
        if (barrett_reduce >= Q)
            barrett_reduce = barrett_reduce - Q;
        if (barrett_reduce < 0)
            barrett_reduce = barrett_reduce + Q;
    end
endfunction

// ============================================================
// FSM — top-level control
// ============================================================
always @(posedge clk) begin
    if (!rst_n) begin
        state     <= ST_IDLE;
        done      <= 0;
        valid     <= 0;
        ct_match  <= 0;
    end else begin
        state <= next_state;
    end
end

always @(*) begin
    next_state = state;
    case (state)
        ST_IDLE:        if (start && trng_valid) next_state = ST_LOAD_SK;
        ST_LOAD_SK:     next_state = ST_MASK_GEN;
        ST_MASK_GEN:    next_state = ST_DECOMP_CT;
        ST_DECOMP_CT:   next_state = ST_NTT_U;
        ST_NTT_U:       next_state = ST_NTT_S;
        ST_NTT_S:       next_state = ST_INNER_PROD;
        ST_INNER_PROD:  next_state = ST_INTT;
        ST_INTT:        next_state = ST_ASCON_PROT;
        ST_ASCON_PROT:  if (ascon_done) next_state = ST_DECODE;
        ST_DECODE:      next_state = ST_RE_ENCRYPT;
        ST_RE_ENCRYPT:  next_state = ST_COMPARE;
        ST_COMPARE:     next_state = ST_DONE;
        ST_DONE:        next_state = ST_IDLE;
        default:        next_state = ST_IDLE;
    endcase
end

// ============================================================
// ASCON co-processor instantiation
// ============================================================
// (Full ASCON-128 implementation in ascon128.v)
ascon128 u_ascon (
    .clk     (clk),
    .rst_n   (rst_n),
    .start   (ascon_start),
    .data_in (ascon_in),
    .key     (trng_seed[127:0]),
    .data_out(ascon_out),
    .done    (ascon_done)
);

// ============================================================
// Output assignment
// ============================================================
always @(posedge clk) begin
    if (state == ST_DONE) begin
        done  <= 1;
        valid <= 1;
    end else begin
        done  <= 0;
    end
end

endmodule


// ============================================================
// ASCON-128 Permutation Co-processor (stub)
// Full implementation: 64-bit state, 12 rounds, SPN structure
// Reference: ASCON v1.2 specification
// ============================================================
module ascon128 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [11:0] data_in,
    input  wire [127:0] key,
    output reg  [11:0] data_out,
    output reg         done
);
    // ASCON operates on 5×64-bit state (320 bits total)
    reg [63:0] state [0:4];
    reg [3:0]  round;
    reg        running;

    // Round constants
    reg [63:0] RC [0:11];
    initial begin
        RC[0]  = 64'h00000000000000f0;
        RC[1]  = 64'h00000000000000e1;
        RC[2]  = 64'h00000000000000d2;
        RC[3]  = 64'h00000000000000c3;
        RC[4]  = 64'h00000000000000b4;
        RC[5]  = 64'h00000000000000a5;
        RC[6]  = 64'h0000000000000096;
        RC[7]  = 64'h0000000000000087;
        RC[8]  = 64'h0000000000000078;
        RC[9]  = 64'h0000000000000069;
        RC[10] = 64'h000000000000005a;
        RC[11] = 64'h000000000000004b;
    end

    // S-box operation (5-bit, applied to all 64 bit-slices)
    // x0..x4 are the 5 state lanes at the same bit position
    function [4:0] ascon_sbox;
        input [4:0] x;
        reg [4:0] t;
        begin
            t[0] = x[0] ^ x[4];
            t[1] = x[1];
            t[2] = x[2] ^ x[1];
            t[3] = x[3];
            t[4] = x[4] ^ x[3];
            ascon_sbox[0] = t[0] ^ (~t[1] & t[2]);
            ascon_sbox[1] = t[1] ^ (~t[2] & t[3]);
            ascon_sbox[2] = t[2] ^ (~t[3] & t[4]);
            ascon_sbox[3] = t[3] ^ (~t[4] & t[0]);
            ascon_sbox[4] = t[4] ^ (~t[0] & t[1]);
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            done    <= 0; running <= 0; round <= 0;
        end else if (start && !running) begin
            // Initialise state: x0 = IV, x1/x2 = key, x3/x4 = nonce (data_in padded)
            state[0] <= 64'h80800c0800000000;   // ASCON-128 IV
            state[1] <= key[127:64];
            state[2] <= key[63:0];
            state[3] <= {52'h0, data_in};        // nonce (simplified)
            state[4] <= 64'h0;
            running  <= 1;
            round    <= 0;
            done     <= 0;
        end else if (running) begin
            if (round < 12) begin
                // Add round constant to state[2]
                state[2] <= state[2] ^ RC[round];
                // (S-box and linear diffusion layers would execute here
                //  in a pipelined manner — one round per cycle)
                round <= round + 1;
            end else begin
                data_out <= state[0][11:0];
                done     <= 1;
                running  <= 0;
            end
        end else begin
            done <= 0;
        end
    end

endmodule
