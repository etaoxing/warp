// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "builtin.h"

// stubs for platforms where there is no CUDA
#if !WP_ENABLE_CUDA || !WP_ENABLE_MATHDX

extern "C" {

WP_API
bool wp_cuda_compile_fft(
    const char* ltoir_output_path,
    const char* symbol_name,
    int num_include_dirs,
    const char** include_dirs,
    const char* mathdx_include_dir,
    int arch,
    int size,
    int elements_per_thread,
    int direction,
    int precision,
    int* shared_memory_size
)
{
    printf("CUDA is disabled and/or Warp was not compiled with MathDx support.\n");
    return false;
}

WP_API bool wp_cuda_compile_dot(
    const char* ltoir_output_path,
    const char* symbol_name,
    int num_include_dirs,
    const char** include_dirs,
    const char* mathdx_include_dir,
    int arch,
    int M,
    int N,
    int K,
    int precision_A,
    int precision_B,
    int precision_C,
    int type,
    int arrangement_A,
    int arrangement_B,
    int arrangement_C,
    int num_threads,
    int lda,
    int ldb,
    int ldc
)
{
    printf("CUDA is disabled and/or Warp was not compiled with MathDx support.\n");
    return false;
}

WP_API bool wp_cuda_compile_gemm_pipeline(
    const char* ltoir_output_path,
    const char* symbol_prefix,
    int arch,
    int tile_m,
    int tile_n,
    int tile_k,
    int full_m,
    int full_n,
    int full_k,
    int storage_precision_a,
    int storage_precision_b,
    int storage_precision_c,
    int compute_precision_a,
    int compute_precision_b,
    int accumulator_precision,
    int type,
    int arrangement_a,
    int arrangement_b,
    int arrangement_c,
    int num_threads,
    int pipeline_depth,
    int* device_pipeline_storage_size,
    int* device_pipeline_storage_alignment,
    int* shared_memory_buffer_size,
    int* shared_memory_buffer_alignment,
    int* tile_pipeline_storage_size,
    int* tile_pipeline_storage_alignment,
    int* accumulator_storage_size,
    int* accumulator_storage_alignment,
    int* pipeline_block_dim
)
{
    printf("CUDA is disabled and/or Warp was not compiled with MathDx support.\n");
    return false;
}

WP_API bool wp_cuda_compile_solver(
    const char* fatbin_output_path,
    const char* ltoir_output_path,
    const char* symbol_name,
    int num_include_dirs,
    const char** include_dirs,
    const char* mathdx_include_dir,
    int arch,
    int M,
    int N,
    int K,
    int function,
    int side,
    int diag,
    int precision,
    int arrangement_A,
    int arrangement_B,
    int fill_mode,
    int num_threads
)
{
    printf("CUDA is disabled and/or Warp was not compiled with MathDx support.\n");
    return false;
}

}  // extern "C"

#endif  // !WP_ENABLE_CUDA || !WP_ENABLE_MATHDX
