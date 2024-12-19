#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML internal header
#undef __ARM_FEATURE_SVE
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    ggml_fp16_t d[4];      // deltas for 4 q4_0 blocks
    uint8_t qs[QK4_0 * 2]; // nibbles / quants for 4 q4_0 blocks
} block_q4_0x4;
static_assert(sizeof(block_q4_0x4) == 4 * sizeof(ggml_fp16_t) + QK4_0 * 2, "wrong q4_0x4 block size/padding");

typedef struct {
    ggml_fp16_t d[8];      // deltas for 8 q4_0 blocks
    uint8_t qs[QK4_0 * 4]; // nibbles / quants for 8 q4_0 blocks
} block_q4_0x8;
static_assert(sizeof(block_q4_0x8) == 8 * sizeof(ggml_fp16_t) + QK4_0 * 4, "wrong q4_0x8 block size/padding");

typedef struct {
    ggml_fp16_t d[16];     // deltas for 16 q4_0 blocks
    uint8_t qs[QK4_0 * 8]; // nibbles / quants for 16 q4_0 blocks
} block_q4_0x16;
static_assert(sizeof(block_q4_0x16) == 16 * sizeof(ggml_fp16_t) + QK4_0 * 8, "wrong q4_0x16 block size/padding");

typedef struct {
    ggml_fp16_t d[64];     // deltas for 64 q4_0 blocks
    uint8_t qs[QK4_0 * 32];// nibbles / quants for 64 q4_0 blocks
} block_q4_0x64;
static_assert(sizeof(block_q4_0x64) == 64 * sizeof(ggml_fp16_t) + QK4_0 * 32, "wrong q4_0x64 block size/padding");

typedef struct {
    ggml_fp16_t d[2];      // deltas for 2 q8_0 blocks
    int8_t qs[QK8_0 * 2];  // quants for 2 q8_0 blocks
} block_q8_0x2;
static_assert(sizeof(block_q8_0x2) == 2 * sizeof(ggml_fp16_t) + QK8_0 * 2, "wrong q8_0x2 block size/padding");

typedef struct {
    ggml_fp16_t d[4];      // deltas for 4 q8_0 blocks
    int8_t qs[QK8_0 * 4];  // quants for 4 q8_0 blocks
} block_q8_0x4;
static_assert(sizeof(block_q8_0x4) == 4 * sizeof(ggml_fp16_t) + QK8_0 * 4, "wrong q8_0x4 block size/padding");

typedef struct {
    ggml_fp16_t d[8];      // deltas for 8 q8_0 blocks
    int8_t qs[QK8_0 * 8];  // quants for 8 q8_0 blocks
} block_q8_0x8;
static_assert(sizeof(block_q8_0x8) == 8 * sizeof(ggml_fp16_t) + QK8_0 * 8, "wrong q8_0x8 block size/padding");

// Quantization
void quantize_row_q4_0_reference(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int k);
void quantize_row_q4_1_reference(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int k);
void quantize_row_q5_0_reference(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int k);
void quantize_row_q5_1_reference(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int k);
void quantize_row_q8_0_reference(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int k);
void quantize_row_q8_1_reference(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int k);

void quantize_row_q2_K_reference(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int k);
void quantize_row_q3_K_reference(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int k);
void quantize_row_q4_K_reference(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int k);
void quantize_row_q5_K_reference(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int k);
void quantize_row_q6_K_reference(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int k);
void quantize_row_q8_K_reference(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int k);

void quantize_row_iq3_xxs_reference(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int k);
void quantize_row_iq4_nl_reference (const float * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int k);
void quantize_row_iq4_xs_reference (const float * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int k);
void quantize_row_iq3_s_reference  (const float * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int k);
void quantize_row_iq2_s_reference  (const float * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int k);

void quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q4_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q5_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q5_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q8_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);

void quantize_row_q2_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q3_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q4_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q5_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);

void quantize_row_iq3_xxs(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_iq4_nl (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_iq4_xs (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_iq3_s  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_iq2_s  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);

// Dequantization
void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_q4_1(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_q5_0(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_q5_1(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
//void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);

void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_q3_K(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_q8_K(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);

void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_iq2_xs (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_iq2_s  (const block_iq2_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_iq3_xxs(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_iq1_s  (const block_iq1_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_iq1_m  (const block_iq1_m   * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_iq4_nl (const block_iq4_nl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_iq4_xs (const block_iq4_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_iq3_s  (const block_iq3_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);

// Dot product
void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_q2_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q3_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xs_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq1_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq1_m_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_nl_q8_0 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_xs_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
size_t quantize_iq2_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_iq2_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_iq2_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_iq3_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_iq1_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_iq1_m  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_iq4_nl (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_iq4_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_iq3_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);

size_t quantize_q2_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_q3_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_q4_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_q5_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_q6_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_q4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_q4_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_q5_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_q5_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_q8_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);

void iq2xs_init_impl(enum ggml_type type);
void iq2xs_free_impl(enum ggml_type type);
void iq3xs_init_impl(int grid_size);
void iq3xs_free_impl(int grid_size);



block_q4_0x4 make_block_q4_0x4(const block_q4_0 * const in[4], unsigned int block_len);
block_q4_0x8 make_block_q4_0x8(const block_q4_0 * const in[8], unsigned int block_len);
block_q8_0x4 make_block_q8_0x4(const block_q8_0 * const in[4], unsigned int block_len);
block_q8_0x8 make_block_q8_0x8(const block_q8_0 * const in[8], unsigned int block_len);
void quantize_row_q8_0_and_make_block_q8_0x2(const float * restrict x, void * restrict vy, int k, int rows_interleaved);
void quantize_row_q8_0_and_make_block_q8_0x4(const float * restrict x, void * restrict vy, int k, int rows_interleaved);

// GEMV
void ggml_gemv_q4_0_q8_0_blocked8_neon(const int n, int output_channels, int input_width, float * restrict s, const void * restrict vx, const void * restrict vy, int ith, int nth);
void ggml_gemv_q4_0_q8_0_blocked8_sve(const int n, int output_channels, int input_width, float * restrict s, const void * restrict vx, const void * restrict vy, int ith, int nth);
void ggml_gemv_q8_0_q8_0_blocked8_neon(const int n, int output_channels, int input_width, float * restrict s, const void * restrict vx, const void * restrict vy, int ith, int nth);
void ggml_gemv_q8_0_q8_0_blocked8_sve(const int n, int output_channels, int input_width, float * restrict s, const void * restrict vx, const void * restrict vy, int ith, int nth);

// GEMM
void ggml_gemm_q4_0_q8_0(const int n, int rows, int output_channels, int input_width, float * restrict s, const void * restrict vx, const void * restrict vy, int ith, int nth);
void ggml_gemm_q4_0_q8_0_2x4blocked_mmla(const int n, int output_channels, int input_width, float * restrict s, const void * restrict vx, const void * restrict vy, int ith, int nth);
void ggml_gemm_q8_0_q8_0(const int n, int rows, int output_channels, int input_width, float * restrict s, const void * restrict vx, const void * restrict vy, int ith, int nth);
void ggml_gemm_q8_0_q8_0_2x4blocked_mmla(const int n, int output_channels, int input_width, float * restrict s, const void * restrict vx, const void * restrict vy, int ith, int nth);

#ifdef __cplusplus
}
#endif

