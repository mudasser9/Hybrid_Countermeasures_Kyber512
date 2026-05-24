// =============================================================================
// kyber512_decap_hybrid.v  (FIXED VERSION)
// =============================================================================

`timescale 1ns / 1ps

module kyber512_decap_hybrid (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    output reg          done,
    output reg          valid,

    input  wire [3071:0] sk_poly_0,
    input  wire [3071:0] sk_poly_1,

    input  wire [2559:0] ct_u_0,
    input  wire [2559:0] ct_u_1,

    input  wire [1023:0] ct_v,

    input  wire [127:0] trng_seed,
    input  wire         trng_valid,

    output reg  [255:0] m_prime,
    output reg          ct_match
);

localparam Q = 3329;
localparam N = 256;

// FSM STATES
localparam ST_IDLE       = 4'd0;
localparam ST_LOAD_SK    = 4'd1;
localparam ST_MASK_GEN   = 4'd2;
localparam ST_DECOMP_CT  = 4'd3;
localparam ST_NTT_U      = 4'd4;
localparam ST_INNER_PROD = 4'd5;
localparam ST_INTT       = 4'd6;
localparam ST_DECODE     = 4'd7;
localparam ST_COMPARE    = 4'd8;
localparam ST_DONE       = 4'd9;

reg [3:0] state, next_state;

integer idx;

// INTERNAL ARRAYS
reg [12:0] s_ntt_0 [0:N-1];
reg [12:0] s_ntt_1 [0:N-1];

reg [12:0] u_ntt_0 [0:N-1];
reg [12:0] u_ntt_1 [0:N-1];

reg [12:0] su_acc [0:N-1];
reg [12:0] w_poly [0:N-1];

reg [12:0] v_decomp   [0:N-1];
reg [12:0] u_decomp_0 [0:N-1];
reg [12:0] u_decomp_1 [0:N-1];

reg [12:0] s0_ntt_0 [0:N-1];
reg [12:0] s1_ntt_0 [0:N-1];

reg [12:0] s0_ntt_1 [0:N-1];
reg [12:0] s1_ntt_1 [0:N-1];

// RANDOMNESS
reg [127:0] lfsr_state;

// =============================================================================
// LFSR
// =============================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        lfsr_state <= trng_seed;
    else
        lfsr_state <= {
            lfsr_state[126:0],
            lfsr_state[127] ^
            lfsr_state[125] ^
            lfsr_state[100] ^
            lfsr_state[37]
        };
end

// =============================================================================
// FSM REGISTER
// =============================================================================

integer i;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin

        state     <= ST_IDLE;
        done      <= 0;
        valid     <= 0;
        ct_match  <= 0;
        m_prime   <= 0;

        for (i = 0; i < N; i = i + 1) begin

            s_ntt_0[i] <= 0;
            s_ntt_1[i] <= 0;

            u_ntt_0[i] <= 0;
            u_ntt_1[i] <= 0;

            su_acc[i]  <= 0;
            w_poly[i]  <= 0;

            v_decomp[i]   <= 0;
            u_decomp_0[i] <= 0;
            u_decomp_1[i] <= 0;

            s0_ntt_0[i] <= 0;
            s1_ntt_0[i] <= 0;

            s0_ntt_1[i] <= 0;
            s1_ntt_1[i] <= 0;
        end

    end else begin
        state <= next_state;
    end
end

// =============================================================================
// FSM NEXT STATE
// =============================================================================

always @(*) begin

    next_state = state;

    case (state)

        ST_IDLE:
            if (start && trng_valid)
                next_state = ST_LOAD_SK;

        ST_LOAD_SK:
            next_state = ST_MASK_GEN;

        ST_MASK_GEN:
            next_state = ST_DECOMP_CT;

        ST_DECOMP_CT:
            next_state = ST_NTT_U;

        ST_NTT_U:
            next_state = ST_INNER_PROD;

        ST_INNER_PROD:
            next_state = ST_INTT;

        ST_INTT:
            next_state = ST_DECODE;

        ST_DECODE:
            next_state = ST_COMPARE;

        ST_COMPARE:
            next_state = ST_DONE;

        ST_DONE:
            next_state = ST_IDLE;

        default:
            next_state = ST_IDLE;

    endcase
end

// =============================================================================
// LOAD SECRET KEY
// =============================================================================

always @(posedge clk) begin

    if (state == ST_LOAD_SK) begin

        for (idx = 0; idx < N; idx = idx + 1) begin

            s_ntt_0[idx] <= sk_poly_0[(idx*12) +: 12];
            s_ntt_1[idx] <= sk_poly_1[(idx*12) +: 12];

        end
    end
end

// =============================================================================
// MASK GENERATION
// =============================================================================

always @(posedge clk) begin

    if (state == ST_MASK_GEN) begin

        for (idx = 0; idx < N; idx = idx + 1) begin

            s0_ntt_0[idx] <= lfsr_state[(idx % 115) +: 13] % Q;
            s1_ntt_0[idx] <= (s_ntt_0[idx] + Q - s0_ntt_0[idx]) % Q;

            s0_ntt_1[idx] <= lfsr_state[((idx+13) % 115) +: 13] % Q;
            s1_ntt_1[idx] <= (s_ntt_1[idx] + Q - s0_ntt_1[idx]) % Q;

        end
    end
end

// =============================================================================
// DECOMPRESS CIPHERTEXT
// =============================================================================

always @(posedge clk) begin

    if (state == ST_DECOMP_CT) begin

        for (idx = 0; idx < N; idx = idx + 1) begin

            v_decomp[idx] <= ((ct_v[(idx*4) +: 4] * Q) + 8) >> 4;

            u_decomp_0[idx] <=
                ((ct_u_0[(idx*10) +: 10] * Q) + 512) >> 10;

            u_decomp_1[idx] <=
                ((ct_u_1[(idx*10) +: 10] * Q) + 512) >> 10;

        end
    end
end

// =============================================================================
// COPY TO NTT BUFFERS
// =============================================================================

always @(posedge clk) begin

    if (state == ST_NTT_U) begin

        for (idx = 0; idx < N; idx = idx + 1) begin

            u_ntt_0[idx] <= u_decomp_0[idx];
            u_ntt_1[idx] <= u_decomp_1[idx];

        end
    end
end

// =============================================================================
// INNER PRODUCT
// =============================================================================

always @(posedge clk) begin

    if (state == ST_INNER_PROD) begin

        for (idx = 0; idx < N; idx = idx + 1) begin

            su_acc[idx] <= (
                (
                    ((s0_ntt_0[idx] + s1_ntt_0[idx]) % Q)
                    * u_ntt_0[idx]
                )
                +
                (
                    ((s0_ntt_1[idx] + s1_ntt_1[idx]) % Q)
                    * u_ntt_1[idx]
                )
            ) % Q;

        end
    end
end

// =============================================================================
// SUBTRACT
// =============================================================================

always @(posedge clk) begin

    if (state == ST_INTT) begin

        for (idx = 0; idx < N; idx = idx + 1) begin

            if (v_decomp[idx] >= su_acc[idx])
                w_poly[idx] <= v_decomp[idx] - su_acc[idx];
            else
                w_poly[idx] <= v_decomp[idx] + Q - su_acc[idx];

        end
    end
end

// =============================================================================
// MESSAGE DECODE
// =============================================================================

always @(posedge clk) begin

    if (state == ST_DECODE) begin

        for (idx = 0; idx < 256; idx = idx + 1) begin

            if ((w_poly[idx] > 13'd832) &&
                (w_poly[idx] < 13'd2497))
                m_prime[idx] <= 1'b1;
            else
                m_prime[idx] <= 1'b0;

        end
    end
end

// =============================================================================
// COMPARE
// =============================================================================

always @(posedge clk) begin

    if (state == ST_COMPARE) begin

        if (m_prime != 256'h0)
            ct_match <= 1'b1;
        else
            ct_match <= 1'b0;

    end
end

// =============================================================================
// DONE SIGNALS
// =============================================================================

always @(posedge clk or negedge rst_n) begin

    if (!rst_n) begin

        done  <= 0;
        valid <= 0;

    end else begin

        if (state == ST_DONE) begin
            done  <= 1;
            valid <= 1;
        end else begin
            done  <= 0;
            valid <= 0;
        end
    end
end

endmodule