// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// tile_matmul: cooperative scalar GEMM and the tile_matmul / tile_matmul_acc
// entry templates.
//
// Performance note: the cooperative scalar GEMM (`partitioned_gemm::scalar_matmul`)
// is not solely a fallback for builds without libmathdx. For small tiles
// and certain shape/dtype combinations it outperforms cuBLASDx because it
// avoids the LTO compilation cost and the per-launch setup overhead
// inherent to libmathdx. Users can deliberately route a kernel through
// the scalar path on a libmathdx-enabled build by setting the module
// option `enable_mathdx_gemm=False` (or globally via
// `wp.config.enable_mathdx_gemm = False`). The crossover point is
// shape- and dtype-dependent -- benchmark your configuration.

#pragma once

#include "tile.h"

// libmathdx's tensor and pipeline device functions pass these opaque handles
// by value. tile_gemm specializes the complete matrix shape and dense strides,
// so global and shared tensors carry only their backing pointer.
struct libmathdx_tensor_0s_0s {
    void* ptr;
};
struct libmathdx_pipeline {
    void* ptr;
};

#ifdef __clang__
// disable warnings related to C++17 extensions on CPU JIT builds
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif  // __clang__

namespace wp {

namespace partitioned_gemm {

template <typename T> inline CUDA_CALLABLE const T& index(const T* __restrict__ p, int i, int j, int stride)
{
    return p[i * stride + j];
}

template <typename T> inline CUDA_CALLABLE T& index(T* __restrict__ p, int i, int j, int stride)
{
    return p[i * stride + j];
}

template <int PartitionM, int PartitionN, typename Tile> struct partition_t {
    static constexpr int M = PartitionM;
    static constexpr int N = PartitionN;
    static constexpr int Stride = Tile::Layout::Shape::dim(1);

    using T = typename Tile::Type;

    inline partition_t(Tile& A)
    {
        data = A.data.ptr;

        // todo: do ceil div for non-multiples of M,N
        shape[0] = Tile::Layout::Shape::dim(0) / PartitionM;
        shape[1] = Tile::Layout::Shape::dim(1) / PartitionN;
    }

    // underlying data
    T* data;

    // partition dimensions
    int shape[2];
};

template <typename Partition> inline int partition_size(const Partition& part) { return part.shape[0] * part.shape[1]; }

// returns the x, y coordinates of a tile given a linear index
template <typename Partition> inline void partition_coord(const Partition& part, const int t, int& i, int& j)
{
    i = t / part.shape[1];
    j = t % part.shape[1];
}

template <typename Partition> inline auto partition_load(const Partition& tile, int i, int j)
{
    mat_t<Partition::M, Partition::N, typename Partition::T> out;

    const int tile_i = i * Partition::M;
    const int tile_j = j * Partition::N;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Partition::M; ++i) {
        WP_PRAGMA_UNROLL
        for (int j = 0; j < Partition::N; ++j) {
            out.data[i][j] = partitioned_gemm::index(tile.data, tile_i + i, tile_j + j, Partition::Stride);
        }
    }

    return out;
}

template <typename Partition, typename Value>
inline void partition_store(const Partition& tile, int i, int j, const Value& value)
{
    const int tile_i = Partition::M * i;
    const int tile_j = Partition::N * j;

    WP_PRAGMA_UNROLL
    for (int i = 0; i < Partition::M; ++i) {
        WP_PRAGMA_UNROLL
        for (int j = 0; j < Partition::N; ++j) {
            index(tile.data, tile_i + i, tile_j + j, Partition::Stride) = value.data[i][j];
        }
    }
}


template <typename TileA, typename TileB, typename TileC>
inline CUDA_CALLABLE void matmul(TileA& A, TileB& B, TileC& out)
{
    const int TILE_M = 4;
    const int TILE_N = 4;
    const int TILE_K = 4;

    auto A_tile = partition_t<TILE_M, TILE_K, TileA>(A);
    auto B_tile = partition_t<TILE_K, TILE_N, TileB>(B);
    auto C_tile = partition_t<TILE_M, TILE_N, TileC>(out);

    // static_assert(is_same<typename TileA::Type, typename TileB::Type>::value);

    const int length = partition_size(C_tile);

    for (int t = WP_TILE_THREAD_IDX; t < length; t += WP_TILE_BLOCK_DIM) {
        int i, j;
        partition_coord(C_tile, t, i, j);

        // accumulator
        auto sum = partition_load(C_tile, i, j);

        WP_PRAGMA_UNROLL
        for (int k = 0; k < A_tile.shape[1]; k++) {
            const auto a = partition_load(A_tile, i, k);
            const auto b = partition_load(B_tile, k, j);

            sum += mul(a, b);
        }

        partition_store(C_tile, i, j, sum);
    }
}

// Register-blocked scalar GEMM with direct pointer arithmetic.
//
// Each thread computes a BM x BN sub-tile of C by iterating over K,
// loading BM values from A and BN values from B per step, and
// accumulating via an outer product into BM*BN registers.
//
// Optimizations:
//   - Direct pointer access with __restrict__ and compile-time strides
//     (bypasses tile_coord / index_from_coord abstraction in the hot loop)
//   - Precomputed row/column offsets outside the K loop
//   - Compile-time boundary elimination when M%BM==0 and N%BN==0
//   - K loop fully unrolled for small K (<=32), improving scheduling and
//     reducing branch overhead; left to compiler for large K to limit I-cache use
//   - Adaptive sub-tile size: 8x4, 4x4, 4x2, 2x2, or 1x1 based on parallelism
template <
    bool Accumulate,
    typename LayoutA,
    typename LayoutB,
    typename LayoutC,
    typename StorageA,
    typename StorageB,
    typename StorageC,
    typename T>
inline CUDA_CALLABLE void scalar_matmul(const StorageA& A, const StorageB& B, StorageC& C, T& alpha, T& beta)
{
    constexpr int M = LayoutC::Shape::dim(0);
    constexpr int N = LayoutC::Shape::dim(1);
    constexpr int K = LayoutA::Shape::dim(1);

    // Compile-time strides for direct pointer arithmetic
    constexpr int sa0 = LayoutA::Stride::dim(0);
    constexpr int sa1 = LayoutA::Stride::dim(1);
    constexpr int sb0 = LayoutB::Stride::dim(0);
    constexpr int sb1 = LayoutB::Stride::dim(1);
    constexpr int sc0 = LayoutC::Stride::dim(0);
    constexpr int sc1 = LayoutC::Stride::dim(1);

    // Use actual storage element types for pointer declarations.
    // A, B, C may have different element types in the backward pass
    // (e.g. adj_C is T_C*, B is T_B*). T is used only for the accumulator.
    using ElemA = typename remove_reference<decltype(A.ptr[0])>::type;
    using ElemB = typename remove_reference<decltype(B.ptr[0])>::type;
    using ElemC = typename remove_reference<decltype(C.ptr[0])>::type;

    // Direct pointer access with __restrict__ to enable compiler optimizations
    const ElemA* __restrict__ a_ptr = A.ptr;
    const ElemB* __restrict__ b_ptr = B.ptr;
    ElemC* __restrict__ c_ptr = C.ptr;

    // Choose register sub-tile size to maximize effective throughput, balancing
    // arithmetic intensity (FMAs per shared-memory load) against thread
    // utilization.  Higher-intensity sub-tiles (>= 4x2, intensity >= 1.33) are
    // worth a modest utilization drop because the reduced memory traffic more
    // than compensates; we allow down to 75 % utilization for those.  For
    // low-intensity sub-tiles (2x2 / 1x1) we require full utilization since
    // their throughput relies on parallelism rather than reuse.
    constexpr int min_blocks_full = WP_TILE_BLOCK_DIM;
    constexpr int min_blocks_75 = (WP_TILE_BLOCK_DIM * 3 + 3) / 4;  // ceil(bd*3/4)
    constexpr int blocks_8x4 = ((M + 7) / 8) * ((N + 3) / 4);
    constexpr int blocks_4x4 = ((M + 3) / 4) * ((N + 3) / 4);
    constexpr int blocks_4x2 = ((M + 3) / 4) * ((N + 1) / 2);
    constexpr int blocks_2x2 = ((M + 1) / 2) * ((N + 1) / 2);
    constexpr int BM = (blocks_8x4 >= min_blocks_75) ? 8
        : (blocks_4x4 >= min_blocks_75)              ? 4
        : (blocks_4x2 >= min_blocks_75)              ? 4
        : (blocks_2x2 >= min_blocks_full)            ? 2
                                                     : 1;
    constexpr int BN = (BM == 8) ? 4 : (BM == 4 && blocks_4x4 >= min_blocks_75) ? 4 : (BM == 4) ? 2 : BM;

    // Number of sub-tile blocks covering the output (ceiling division)
    constexpr int blocks_m = (M + BM - 1) / BM;
    constexpr int blocks_n = (N + BN - 1) / BN;
    constexpr int num_blocks = blocks_m * blocks_n;

    // Whether boundary checks can be eliminated at compile time
    constexpr bool aligned_m = (M % BM == 0);
    constexpr bool aligned_n = (N % BN == 0);

    for (int t = WP_TILE_THREAD_IDX; t < num_blocks; t += WP_TILE_BLOCK_DIM) {
        const int block_i = t / blocks_n;
        const int block_j = t % blocks_n;

        const int base_i = block_i * BM;
        const int base_j = block_j * BN;

        // Precompute base offsets for A rows and B columns (constant across K)
        int a_offsets[BM];
        WP_PRAGMA_UNROLL
        for (int si = 0; si < BM; si++)
            a_offsets[si] = (base_i + si) * sa0;

        int b_offsets[BN];
        WP_PRAGMA_UNROLL
        for (int sj = 0; sj < BN; sj++)
            b_offsets[sj] = (base_j + sj) * sb1;

        // Accumulator in registers
        T sum[BM][BN];
        WP_PRAGMA_UNROLL
        for (int si = 0; si < BM; si++)
            WP_PRAGMA_UNROLL
        for (int sj = 0; sj < BN; sj++)
            sum[si][sj] = T(0);

        // Reduction along K with register-blocked outer product.
        // For small K (<= 32), fully unroll to eliminate branch overhead and
        // enable better load/FMA scheduling.  For large K, leave unrolling to
        // the compiler to limit I-cache pressure.
        // The if-constexpr duplicates the body so the pragma applies correctly.
        if constexpr (K <= 32) {
            WP_PRAGMA_UNROLL
            for (int k = 0; k < K; k++) {
                const int ka = k * sa1;
                const int kb = k * sb0;

                T a_reg[BM];
                WP_PRAGMA_UNROLL
                for (int si = 0; si < BM; si++) {
                    if constexpr (aligned_m)
                        a_reg[si] = T(a_ptr[a_offsets[si] + ka]);
                    else
                        a_reg[si] = (base_i + si < M) ? T(a_ptr[a_offsets[si] + ka]) : T(0);
                }

                T b_reg[BN];
                WP_PRAGMA_UNROLL
                for (int sj = 0; sj < BN; sj++) {
                    if constexpr (aligned_n)
                        b_reg[sj] = T(b_ptr[kb + b_offsets[sj]]);
                    else
                        b_reg[sj] = (base_j + sj < N) ? T(b_ptr[kb + b_offsets[sj]]) : T(0);
                }

                WP_PRAGMA_UNROLL
                for (int si = 0; si < BM; si++)
                    WP_PRAGMA_UNROLL
                for (int sj = 0; sj < BN; sj++)
                    sum[si][sj] = muladd<T>(a_reg[si], b_reg[sj], sum[si][sj]);
            }
        } else {
            for (int k = 0; k < K; k++) {
                const int ka = k * sa1;
                const int kb = k * sb0;

                T a_reg[BM];
                WP_PRAGMA_UNROLL
                for (int si = 0; si < BM; si++) {
                    if constexpr (aligned_m)
                        a_reg[si] = T(a_ptr[a_offsets[si] + ka]);
                    else
                        a_reg[si] = (base_i + si < M) ? T(a_ptr[a_offsets[si] + ka]) : T(0);
                }

                T b_reg[BN];
                WP_PRAGMA_UNROLL
                for (int sj = 0; sj < BN; sj++) {
                    if constexpr (aligned_n)
                        b_reg[sj] = T(b_ptr[kb + b_offsets[sj]]);
                    else
                        b_reg[sj] = (base_j + sj < N) ? T(b_ptr[kb + b_offsets[sj]]) : T(0);
                }

                WP_PRAGMA_UNROLL
                for (int si = 0; si < BM; si++)
                    WP_PRAGMA_UNROLL
                for (int sj = 0; sj < BN; sj++)
                    sum[si][sj] = muladd<T>(a_reg[si], b_reg[sj], sum[si][sj]);
            }
        }

        // Store results with direct pointer arithmetic
        WP_PRAGMA_UNROLL
        for (int si = 0; si < BM; si++) {
            WP_PRAGMA_UNROLL
            for (int sj = 0; sj < BN; sj++) {
                if constexpr (aligned_m && aligned_n) {
                    const int idx = (base_i + si) * sc0 + (base_j + sj) * sc1;
                    if constexpr (Accumulate)
                        c_ptr[idx] = ElemC(alpha * sum[si][sj] + beta * T(c_ptr[idx]));
                    else
                        c_ptr[idx] = ElemC(alpha * sum[si][sj]);
                } else {
                    if (base_i + si < M && base_j + sj < N) {
                        const int idx = (base_i + si) * sc0 + (base_j + sj) * sc1;
                        if constexpr (Accumulate)
                            c_ptr[idx] = ElemC(alpha * sum[si][sj] + beta * T(c_ptr[idx]));
                        else
                            c_ptr[idx] = ElemC(alpha * sum[si][sj]);
                    }
                }
            }
        }
    }
}

}  // namespace partitioned_gemm


// tile_matmul: C = alpha * A @ B (does not read from C)
template <
    typename Fwd,
    typename AdjA,
    typename AdjB,
    typename TileA,
    typename TileB,
    typename TileC,
    typename Alpha,
    typename Beta>
TileC& tile_matmul(
    Fwd fun_forward, AdjA fun_backward_A, AdjB fun_backward_B, TileA& A, TileB& B, TileC& C, Alpha& alpha, Beta& beta
)
{
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;
    using ShapeC = typename TileC::Layout::Shape;

    static_assert(ShapeA::N == 2, "Expected ShapeA::N == 2");
    static_assert(ShapeB::N == 2, "Expected ShapeB::N == 2");
    static_assert(ShapeC::N == 2, "Expected ShapeC::N == 2");

    static_assert(ShapeA::dim(1) == ShapeB::dim(0), "Expected ShapeA::dim(1) == ShapeB::dim(0)");
    static_assert(ShapeC::dim(0) == ShapeA::dim(0), "Expected ShapeC::dim(0) == ShapeA::dim(0)");
    static_assert(ShapeC::dim(1) == ShapeB::dim(1), "Expected ShapeC::dim(1) == ShapeB::dim(1)");

    using T = typename TileC::Type;

    T alphaT = T(alpha);
    T betaT = T(beta);

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_matmul<false, typename TileA::Layout, typename TileB::Layout, typename TileC::Layout>(
        A.data, B.data, C.data, alphaT, betaT
    );
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        partitioned_gemm::scalar_matmul<false, typename TileA::Layout, typename TileB::Layout, typename TileC::Layout>(
            A.data, B.data, C.data, alphaT, betaT
        );
    } else {
        fun_forward(&alphaT, A.data.ptr, B.data.ptr, &betaT, C.data.ptr);
    }
#endif

    WP_TILE_SYNC();

    return C;
}

// tile_matmul_acc: C = alpha * A @ B + beta * C (accumulates into C)
template <
    typename Fwd,
    typename AdjA,
    typename AdjB,
    typename TileA,
    typename TileB,
    typename TileC,
    typename Alpha,
    typename Beta>
TileC& tile_matmul_acc(
    Fwd fun_forward, AdjA fun_backward_A, AdjB fun_backward_B, TileA& A, TileB& B, TileC& C, Alpha& alpha, Beta& beta
)
{
    using ShapeA = typename TileA::Layout::Shape;
    using ShapeB = typename TileB::Layout::Shape;
    using ShapeC = typename TileC::Layout::Shape;

    static_assert(ShapeA::N == 2, "Expected ShapeA::N == 2");
    static_assert(ShapeB::N == 2, "Expected ShapeB::N == 2");
    static_assert(ShapeC::N == 2, "Expected ShapeC::N == 2");

    static_assert(ShapeA::dim(1) == ShapeB::dim(0), "Expected ShapeA::dim(1) == ShapeB::dim(0)");
    static_assert(ShapeC::dim(0) == ShapeA::dim(0), "Expected ShapeC::dim(0) == ShapeA::dim(0)");
    static_assert(ShapeC::dim(1) == ShapeB::dim(1), "Expected ShapeC::dim(1) == ShapeB::dim(1)");

    using T = typename TileC::Type;

    T alphaT = T(alpha);
    T betaT = T(beta);

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_matmul<true, typename TileA::Layout, typename TileB::Layout, typename TileC::Layout>(
        A.data, B.data, C.data, alphaT, betaT
    );
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        partitioned_gemm::scalar_matmul<true, typename TileA::Layout, typename TileB::Layout, typename TileC::Layout>(
            A.data, B.data, C.data, alphaT, betaT
        );
    } else {
        fun_forward(&alphaT, A.data.ptr, B.data.ptr, &betaT, C.data.ptr);
    }
#endif

    WP_TILE_SYNC();

    return C;
}


// backward for tile_matmul_acc (the wp.tile_matmul(a, b, out) syntax)
template <
    typename Fwd,
    typename AdjA,
    typename AdjB,
    typename TileA,
    typename TileB,
    typename TileC,
    typename Alpha,
    typename Beta,
    typename AdjAlpha,
    typename AdjBeta>
void adj_tile_matmul_acc(
    Fwd fun_forward,
    AdjA fun_backward_A,
    AdjB fun_backward_B,
    TileA& A,
    TileB& B,
    TileC& C,
    Alpha& alpha,
    Beta& beta,
    Fwd adj_fun_forward,
    AdjA adj_fun_backward_A,
    AdjB adj_fun_backward_B,
    TileA& adj_A,
    TileB& adj_B,
    TileC& adj_C,
    AdjAlpha& adj_alpha,
    AdjBeta& adj_beta
)
{
    using T_A = typename TileA::Type;
    using T_B = typename TileB::Type;
    using T_C = typename TileC::Type;

    T_A alpha_A = T_A(alpha);
    T_A beta_A = T_A(1.0);
    T_B alpha_B = T_B(alpha);
    T_B beta_B = T_B(1.0);

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    auto At = tile_transpose(A);
    auto Bt = tile_transpose(B);

    // Backward always accumulates into gradients (beta=1.0)
    partitioned_gemm::scalar_matmul<
        true, typename TileC::Layout, typename decltype(Bt)::Layout, typename TileA::Layout>(
        adj_C.grad, Bt.data, adj_A.grad, alpha_A, beta_A
    );
    partitioned_gemm::scalar_matmul<
        true, typename decltype(At)::Layout, typename TileC::Layout, typename TileB::Layout>(
        At.data, adj_C.grad, adj_B.grad, alpha_B, beta_B
    );
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        auto At = tile_transpose(A);
        auto Bt = tile_transpose(B);

        partitioned_gemm::scalar_matmul<
            true, typename TileC::Layout, typename decltype(Bt)::Layout, typename TileA::Layout>(
            adj_C.grad, Bt.data, adj_A.grad, alpha_A, beta_A
        );
        partitioned_gemm::scalar_matmul<
            true, typename decltype(At)::Layout, typename TileC::Layout, typename TileB::Layout>(
            At.data, adj_C.grad, adj_B.grad, alpha_B, beta_B
        );
    } else {
        fun_backward_A(&alpha_A, adj_C.grad.ptr, B.data.ptr, &beta_A, adj_A.grad.ptr);
        fun_backward_B(&alpha_B, A.data.ptr, adj_C.grad.ptr, &beta_B, adj_B.grad.ptr);
    }
#endif

    if (T_C(beta) != T_C(1.0)) {
        for (int i = WP_TILE_THREAD_IDX; i < TileC::Layout::Size; i += WP_TILE_BLOCK_DIM)
            adj_C.grad(i) *= T_C(beta);
    }

    WP_TILE_SYNC();
}

// backward for tile_matmul (the out = wp.tile_matmul(a, b) syntax)
template <
    typename Fwd,
    typename AdjA,
    typename AdjB,
    typename TileA,
    typename TileB,
    typename TileC,
    typename Alpha,
    typename Beta,
    typename AdjAlpha,
    typename AdjBeta>
void adj_tile_matmul(
    Fwd fun_forward,
    AdjA fun_backward_A,
    AdjB fun_backward_B,
    TileA& A,
    TileB& B,
    TileC& C,
    Alpha& alpha,
    Beta& beta,
    Fwd adj_fun_forward,
    AdjA adj_fun_backward_A,
    AdjB adj_fun_backward_B,
    TileA& adj_A,
    TileB& adj_B,
    TileC& adj_C,
    AdjAlpha& adj_alpha,
    AdjBeta& adj_beta,
    TileC& adj_ret
)
{
    using T_A = typename TileA::Type;
    using T_B = typename TileB::Type;
    using T_C = typename TileC::Type;

    T_A alpha_A = T_A(alpha);
    T_A beta_A = T_A(1.0);
    T_B alpha_B = T_B(alpha);
    T_B beta_B = T_B(1.0);

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    auto At = tile_transpose(A);
    auto Bt = tile_transpose(B);

    // Backward always accumulates into gradients (beta=1.0)
    partitioned_gemm::scalar_matmul<
        true, typename TileC::Layout, typename decltype(Bt)::Layout, typename TileA::Layout>(
        adj_C.grad, Bt.data, adj_A.grad, alpha_A, beta_A
    );
    partitioned_gemm::scalar_matmul<
        true, typename decltype(At)::Layout, typename TileC::Layout, typename TileB::Layout>(
        At.data, adj_C.grad, adj_B.grad, alpha_B, beta_B
    );
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        auto At = tile_transpose(A);
        auto Bt = tile_transpose(B);

        partitioned_gemm::scalar_matmul<
            true, typename TileC::Layout, typename decltype(Bt)::Layout, typename TileA::Layout>(
            adj_C.grad, Bt.data, adj_A.grad, alpha_A, beta_A
        );
        partitioned_gemm::scalar_matmul<
            true, typename decltype(At)::Layout, typename TileC::Layout, typename TileB::Layout>(
            At.data, adj_C.grad, adj_B.grad, alpha_B, beta_B
        );
    } else {
        fun_backward_A(&alpha_A, adj_C.grad.ptr, B.data.ptr, &beta_A, adj_A.grad.ptr);
        fun_backward_B(&alpha_B, A.data.ptr, adj_C.grad.ptr, &beta_B, adj_B.grad.ptr);
    }
#endif

    WP_TILE_SYNC();
}


template <bool Transpose, typename T> inline CUDA_CALLABLE array_t<T> tile_gemm_array_view(array_t<T>& array)
{
    array_t<T> view = array;
    if constexpr (Transpose) {
        const int shape = view.shape[0];
        const int stride = view.strides[0];
        view.shape.dims[0] = view.shape[1];
        view.shape.dims[1] = shape;
        view.strides[0] = view.strides[1];
        view.strides[1] = stride;
    }
    return view;
}


template <bool Transpose, int Rows, int Columns, typename T>
inline CUDA_CALLABLE auto tile_gemm_load_stored(array_t<T>& array, int row, int column)
{
    if constexpr (Transpose) {
        using Shape = tile_shape_t<Columns, Rows>;
        using Stride = tile_stride_t<Rows, 1>;
        auto stored = tile_alloc_empty<T, Shape, Stride, false>();
        stored = tile_load<T, true, false, Columns, Rows>(array, column, row);
        return stored;
    } else {
        using Shape = tile_shape_t<Rows, Columns>;
        using Stride = tile_stride_t<Columns, 1>;
        auto stored = tile_alloc_empty<T, Shape, Stride, false>();
        stored = tile_load<T, true, false, Rows, Columns>(array, row, column);
        return stored;
    }
}


template <bool Transpose, typename Tile> inline CUDA_CALLABLE auto tile_gemm_logical_view(Tile& stored)
{
    if constexpr (Transpose) {
        return tile_transpose(stored);
    } else {
        return tile_shared_t<typename Tile::Type, typename Tile::Layout, false>(stored.data.ptr, stored.grad.ptr);
    }
}


template <
    typename CreateTilePipeline,
    typename CreateAccumulator,
    typename FinishAccumulator,
    typename ExecutePipeline,
    typename CopyAccumulator,
    typename DestroyAccumulator,
    typename DestroyTilePipeline>
struct tile_gemm_pipeline_functions_t {
    CreateTilePipeline create_tile_pipeline;
    CreateAccumulator create_accumulator;
    FinishAccumulator finish_accumulation;
    ExecutePipeline execute_pipeline;
    CopyAccumulator copy_accumulator;
    DestroyAccumulator destroy_accumulator;
    DestroyTilePipeline destroy_tile_pipeline;
};


template <
    typename CreateTilePipeline,
    typename CreateAccumulator,
    typename FinishAccumulator,
    typename ExecutePipeline,
    typename CopyAccumulator,
    typename DestroyAccumulator,
    typename DestroyTilePipeline>
inline CUDA_CALLABLE auto make_tile_gemm_pipeline_functions(
    CreateTilePipeline create_tile_pipeline,
    CreateAccumulator create_accumulator,
    FinishAccumulator finish_accumulation,
    ExecutePipeline execute_pipeline,
    CopyAccumulator copy_accumulator,
    DestroyAccumulator destroy_accumulator,
    DestroyTilePipeline destroy_tile_pipeline
)
{
    return tile_gemm_pipeline_functions_t<
        CreateTilePipeline, CreateAccumulator, FinishAccumulator, ExecutePipeline, CopyAccumulator, DestroyAccumulator,
        DestroyTilePipeline> {
        create_tile_pipeline, create_accumulator,  finish_accumulation,   execute_pipeline,
        copy_accumulator,     destroy_accumulator, destroy_tile_pipeline,
    };
}


template <
    int TileM,
    int TileN,
    int TileK,
    int FullK,
    bool TransposeA,
    bool TransposeB,
    typename TA,
    typename TB,
    typename TileC,
    typename Alpha,
    typename Beta>
inline CUDA_CALLABLE void
tile_gemm_fallback_impl(array_t<TA>& A, array_t<TB>& B, int i, int j, TileC& C, const Alpha& alpha, const Beta& beta)
{
    using TC = typename TileC::Type;

    TC alpha_t = TC(alpha);
    TC output_scale = TC(beta);
    TC beta_t = TC(1);
    if (output_scale == TC(0)) {
        C = TC(0);
    } else if (output_scale != TC(1)) {
        for (int index = WP_TILE_THREAD_IDX; index < TileC::Layout::Size; index += WP_TILE_BLOCK_DIM)
            C.data(index) *= output_scale;
        WP_TILE_SYNC();
    }

    constexpr int reduction_tiles = (FullK + TileK - 1) / TileK;

    auto multiply_tile = [&](int k) {
        auto a_stored = tile_gemm_load_stored<TransposeA, TileM, TileK>(A, i * TileM, k * TileK);
        auto b_stored = tile_gemm_load_stored<TransposeB, TileK, TileN>(B, k * TileK, j * TileN);
        auto a = tile_gemm_logical_view<TransposeA>(a_stored);
        auto b = tile_gemm_logical_view<TransposeB>(b_stored);

        partitioned_gemm::scalar_matmul<
            true, typename decltype(a)::Layout, typename decltype(b)::Layout, typename TileC::Layout>(
            a.data, b.data, C.data, alpha_t, beta_t
        );
        WP_TILE_SYNC();
    };

    // Match Warp's unrolled static range loops for short reductions without
    // multiplying code size for large full-K fallbacks.
    if constexpr (reduction_tiles <= 4) {
        WP_PRAGMA_UNROLL
        for (int k = 0; k < reduction_tiles; ++k)
            multiply_tile(k);
    } else {
        for (int k = 0; k < reduction_tiles; ++k)
            multiply_tile(k);
    }
}


// A pipeline-capable CUDA kernel retains a runtime portable fallback. Keep its
// scalar GEMM frame out of the accelerated path; pure portable specializations
// call the inline implementation above directly.
#if defined(__CUDA_ARCH__)
#define WP_TILE_GEMM_DEVICE_NOINLINE __attribute__((noinline))
#else
#define WP_TILE_GEMM_DEVICE_NOINLINE
#endif

template <
    int TileM,
    int TileN,
    int TileK,
    int FullK,
    bool TransposeA,
    bool TransposeB,
    typename TA,
    typename TB,
    typename TileC,
    typename Alpha,
    typename Beta>
static WP_TILE_GEMM_DEVICE_NOINLINE CUDA_CALLABLE void tile_gemm_pipeline_fallback_impl(
    TA* A_data,
    int A_shape0,
    int A_shape1,
    int A_stride0,
    int A_stride1,
    TB* B_data,
    int B_shape0,
    int B_shape1,
    int B_stride0,
    int B_stride1,
    int i,
    int j,
    typename TileC::Type* C_data,
    typename TileC::Type* C_grad,
    bool C_initialized,
    Alpha alpha,
    Beta beta
)
{
    array_t<TA> A(A_data, A_shape0, A_shape1);
    A.strides[0] = A_stride0;
    A.strides[1] = A_stride1;
    array_t<TB> B(B_data, B_shape0, B_shape1);
    B.strides[0] = B_stride0;
    B.strides[1] = B_stride1;
    tile_shared_t<typename TileC::Type, typename TileC::Layout, false> C(C_data, C_grad, C_initialized);
    tile_gemm_fallback_impl<TileM, TileN, TileK, FullK, TransposeA, TransposeB>(A, B, i, j, C, alpha, beta);
}

#undef WP_TILE_GEMM_DEVICE_NOINLINE


// tile_gemm: C = alpha * A[i, :] @ B[:, j] over the complete global K dimension.
//
// The function table carries the cuBLASDx pipeline ABI. A table of integer
// zeroes selects the portable cooperative fallback at compile time.
template <
    int TileM,
    int TileN,
    int TileK,
    int FullM,
    int FullN,
    int FullK,
    bool TransposeA,
    bool TransposeB,
    bool Accumulate,
    bool ReuseOutputWorkspace,
    int PipelineBufferSize,
    int PipelineBufferAlignment,
    int TilePipelineStorageSize,
    int TilePipelineStorageAlignment,
    int AccumulatorStorageSize,
    int AccumulatorStorageAlignment,
    typename PipelineFunctions,
    typename TA,
    typename TB,
    typename TileC,
    typename Alpha,
    typename Beta>
void tile_gemm_impl(
    PipelineFunctions functions,
    void* device_pipeline_ptr,
    array_t<TA>& A,
    array_t<TB>& B,
    int i,
    int j,
    TileC& C,
    const Alpha& alpha,
    const Beta& beta
)
{
    using TC = typename TileC::Type;

    static_assert(TileC::Layout::Shape::dim(0) == TileM, "Expected C rows to match TileM");
    static_assert(TileC::Layout::Shape::dim(1) == TileN, "Expected C columns to match TileN");

    auto logical_A = tile_gemm_array_view<TransposeA>(A);
    auto logical_B = tile_gemm_array_view<TransposeB>(B);
    assert(
        logical_A.shape[0] == FullM && logical_A.shape[1] == FullK && "tile_gemm() left operand does not match shape"
    );
    assert(
        logical_B.shape[0] == FullK && logical_B.shape[1] == FullN && "tile_gemm() right operand does not match shape"
    );

#if defined(__CUDA_ARCH__) && WP_ENABLE_MATHDX
    if constexpr (!wp_is_null_func<decltype(functions.execute_pipeline)>::value) {
        if (device_pipeline_ptr != nullptr) {
            constexpr int buffer_alignment = PipelineBufferAlignment > 0 ? PipelineBufferAlignment : 1;
            constexpr int buffer_reservation = PipelineBufferSize + buffer_alignment - 1;
            constexpr int tile_pipeline_size = TilePipelineStorageSize > 0 ? TilePipelineStorageSize : 1;
            constexpr int tile_pipeline_alignment = TilePipelineStorageAlignment > 0 ? TilePipelineStorageAlignment : 1;
            constexpr int accumulator_size = AccumulatorStorageSize > 0 ? AccumulatorStorageSize : 1;
            constexpr int accumulator_alignment = AccumulatorStorageAlignment > 0 ? AccumulatorStorageAlignment : 1;
            constexpr int output_values_per_thread = (TileC::Layout::Size + WP_TILE_BLOCK_DIM - 1) / WP_TILE_BLOCK_DIM;

            TC prior_output[Accumulate ? output_values_per_thread : 1];

            // Align in shared address space to preserve vector shared loads.
            char* pipeline_buffer;
            int workspace_allocation;
            if constexpr (ReuseOutputWorkspace) {
                constexpr int output_bytes = TileC::Layout::Size * int(sizeof(TC));
                constexpr int output_reservation = (output_bytes + 15) / 16 * 16;
                char* output_buffer = reinterpret_cast<char*>(C.data.ptr);
                const size_t output_address = __cvta_generic_to_shared(output_buffer);
                pipeline_buffer = static_cast<char*>(
                    __cvta_shared_to_generic((output_address + buffer_alignment - 1) & ~(size_t(buffer_alignment) - 1))
                );
                const int pipeline_span = int(pipeline_buffer - output_buffer) + PipelineBufferSize;
                workspace_allocation = pipeline_span > output_reservation ? pipeline_span - output_reservation : 0;
                tile_shared_storage_t::alloc(workspace_allocation);
            } else {
                char* unaligned_buffer = static_cast<char*>(tile_shared_storage_t::alloc(buffer_reservation));
                const size_t buffer_address = __cvta_generic_to_shared(unaligned_buffer);
                pipeline_buffer = static_cast<char*>(
                    __cvta_shared_to_generic((buffer_address + buffer_alignment - 1) & ~(size_t(buffer_alignment) - 1))
                );
                workspace_allocation = buffer_reservation;
            }
            alignas(tile_pipeline_alignment) char tile_pipeline_storage[tile_pipeline_size];
            alignas(accumulator_alignment) char accumulator_storage[accumulator_size];

            libmathdx_pipeline device_pipeline { device_pipeline_ptr };
            libmathdx_pipeline tile_pipeline { tile_pipeline_storage };
            libmathdx_tensor_0s_0s accumulator { accumulator_storage };
            libmathdx_tensor_0s_0s output { C.data.ptr };
            int tile_i = i;
            int tile_j = j;

            functions.create_tile_pipeline(device_pipeline, tile_pipeline, pipeline_buffer, &tile_i, &tile_j);
            functions.create_accumulator(tile_pipeline, accumulator);
            functions.execute_pipeline(tile_pipeline, accumulator);
            functions.finish_accumulation(accumulator);
            if constexpr (Accumulate) {
                int thread_value = 0;
                for (int index = WP_TILE_THREAD_IDX; index < TileC::Layout::Size; index += WP_TILE_BLOCK_DIM)
                    prior_output[thread_value++] = C.data(index);
                WP_TILE_SYNC();
            }
            if constexpr (ReuseOutputWorkspace) {
                functions.destroy_tile_pipeline(tile_pipeline);
                WP_TILE_SYNC();
                functions.copy_accumulator(accumulator, output);
                functions.destroy_accumulator(accumulator);
            } else {
                functions.copy_accumulator(accumulator, output);
                functions.destroy_accumulator(accumulator);
                functions.destroy_tile_pipeline(tile_pipeline);
            }
            WP_TILE_SYNC();
            tile_shared_storage_t::alloc(-workspace_allocation);

            TC alpha_t = TC(alpha);
            if constexpr (Accumulate) {
                TC beta_t = TC(beta);
                int thread_value = 0;
                for (int index = WP_TILE_THREAD_IDX; index < TileC::Layout::Size; index += WP_TILE_BLOCK_DIM) {
                    if (beta_t == TC(0))
                        C.data(index) *= alpha_t;
                    else
                        C.data(index) = alpha_t * C.data(index) + beta_t * prior_output[thread_value];
                    ++thread_value;
                }
                WP_TILE_SYNC();
            } else if (alpha_t != TC(1)) {
                for (int index = WP_TILE_THREAD_IDX; index < TileC::Layout::Size; index += WP_TILE_BLOCK_DIM)
                    C.data(index) *= alpha_t;
                WP_TILE_SYNC();
            }

            C.initialized = true;
            return;
        }
    }
#else
    (void)functions;
    (void)device_pipeline_ptr;
#endif

#if defined(__CUDA_ARCH__) && WP_ENABLE_MATHDX
    if constexpr (!wp_is_null_func<decltype(functions.execute_pipeline)>::value) {
        tile_gemm_pipeline_fallback_impl<TileM, TileN, TileK, FullK, TransposeA, TransposeB, TA, TB, TileC>(
            A.data, A.shape[0], A.shape[1], A.strides[0], A.strides[1], B.data, B.shape[0], B.shape[1], B.strides[0],
            B.strides[1], i, j, C.data.ptr, C.grad.ptr, C.initialized, alpha, beta
        );
        C.initialized = true;
    } else
#endif
    {
        tile_gemm_fallback_impl<TileM, TileN, TileK, FullK, TransposeA, TransposeB>(A, B, i, j, C, alpha, beta);
    }
}


template <
    int TileM,
    int TileN,
    int TileK,
    int FullM,
    int FullN,
    int FullK,
    bool TransposeA,
    bool TransposeB,
    bool Accumulate,
    bool ReuseOutputWorkspace,
    int PipelineBufferSize,
    int PipelineBufferAlignment,
    int TilePipelineStorageSize,
    int TilePipelineStorageAlignment,
    int AccumulatorStorageSize,
    int AccumulatorStorageAlignment,
    typename PipelineFunctions,
    typename TA,
    typename TB,
    typename TileC,
    typename Alpha,
    typename Beta>
TileC& tile_gemm(
    PipelineFunctions functions,
    void* device_pipeline_ptr,
    array_t<TA>& A,
    array_t<TB>& B,
    int i,
    int j,
    TileC& C,
    const Alpha& alpha,
    const Beta& beta
)
{
    tile_gemm_impl<
        TileM, TileN, TileK, FullM, FullN, FullK, TransposeA, TransposeB, Accumulate, ReuseOutputWorkspace,
        PipelineBufferSize, PipelineBufferAlignment, TilePipelineStorageSize, TilePipelineStorageAlignment,
        AccumulatorStorageSize, AccumulatorStorageAlignment>(
        functions, device_pipeline_ptr, A, B, i, j, C, alpha, beta
    );

    return C;
}


template <
    int TileM,
    int TileN,
    int TileK,
    int FullM,
    int FullN,
    int FullK,
    bool TransposeA,
    bool TransposeB,
    bool Accumulate,
    bool ReuseOutputWorkspace,
    int PipelineBufferSize,
    int PipelineBufferAlignment,
    int TilePipelineStorageSize,
    int TilePipelineStorageAlignment,
    int AccumulatorStorageSize,
    int AccumulatorStorageAlignment,
    typename PipelineFunctions,
    typename TA,
    typename TB,
    typename TileC,
    typename Alpha,
    typename Beta>
void tile_gemm_acc(
    PipelineFunctions functions,
    void* device_pipeline_ptr,
    array_t<TA>& A,
    array_t<TB>& B,
    int i,
    int j,
    TileC& C,
    const Alpha& alpha,
    const Beta& beta
)
{
    tile_gemm_impl<
        TileM, TileN, TileK, FullM, FullN, FullK, TransposeA, TransposeB, Accumulate, ReuseOutputWorkspace,
        PipelineBufferSize, PipelineBufferAlignment, TilePipelineStorageSize, TilePipelineStorageAlignment,
        AccumulatorStorageSize, AccumulatorStorageAlignment>(
        functions, device_pipeline_ptr, A, B, i, j, C, alpha, beta
    );
}


template <
    int TileM,
    int TileN,
    int TileK,
    int FullM,
    int FullN,
    int FullK,
    bool TransposeA,
    bool TransposeB,
    bool Accumulate,
    bool ReuseOutputWorkspace,
    int PipelineBufferSize,
    int PipelineBufferAlignment,
    int TilePipelineStorageSize,
    int TilePipelineStorageAlignment,
    int AccumulatorStorageSize,
    int AccumulatorStorageAlignment,
    typename PipelineFunctions,
    typename TA,
    typename TB,
    typename TileC,
    typename Alpha,
    typename Beta,
    typename AdjAlpha>
void adj_tile_gemm_impl(
    PipelineFunctions functions,
    void* device_pipeline_ptr,
    array_t<TA>& A,
    array_t<TB>& B,
    int i,
    int j,
    TileC& C,
    const Alpha& alpha,
    const Beta& beta,
    PipelineFunctions adj_functions,
    void* adj_device_pipeline_ptr,
    array_t<TA>& adj_A,
    array_t<TB>& adj_B,
    int adj_i,
    int adj_j,
    TileC& adj_C,
    const AdjAlpha& adj_alpha
)
{
    using TC = typename TileC::Type;
    using ShapeA = tile_shape_t<TileM, TileK>;
    using ShapeB = tile_shape_t<TileK, TileN>;
    using StrideA = tile_stride_t<TileK, 1>;
    using StrideB = tile_stride_t<TileN, 1>;

    (void)functions;
    (void)device_pipeline_ptr;
    (void)adj_functions;
    (void)adj_device_pipeline_ptr;
    (void)C;
    (void)adj_i;
    (void)adj_j;
    (void)adj_alpha;

    auto logical_A = tile_gemm_array_view<TransposeA>(A);
    auto logical_B = tile_gemm_array_view<TransposeB>(B);
    auto logical_adj_A = tile_gemm_array_view<TransposeA>(adj_A);
    auto logical_adj_B = tile_gemm_array_view<TransposeB>(adj_B);
    assert(
        logical_A.shape[0] == FullM && logical_A.shape[1] == FullK && "tile_gemm() left operand does not match shape"
    );
    assert(
        logical_B.shape[0] == FullK && logical_B.shape[1] == FullN && "tile_gemm() right operand does not match shape"
    );

    TC alpha_t = TC(alpha);
    TC one = TC(1);
    constexpr int reduction_tiles = (FullK + TileK - 1) / TileK;
    for (int k = 0; k < reduction_tiles; ++k) {
        auto a_stored = tile_gemm_load_stored<TransposeA, TileM, TileK>(A, i * TileM, k * TileK);
        auto b_stored = tile_gemm_load_stored<TransposeB, TileK, TileN>(B, k * TileK, j * TileN);
        auto a = tile_gemm_logical_view<TransposeA>(a_stored);
        auto b = tile_gemm_logical_view<TransposeB>(b_stored);

        if (adj_A.data) {
            auto partial_a = tile_alloc_empty<TA, ShapeA, StrideA, false>();
            auto b_transpose = tile_transpose(b);
            partitioned_gemm::scalar_matmul<
                false, typename TileC::Layout, typename decltype(b_transpose)::Layout,
                typename decltype(partial_a)::Layout>(adj_C.grad, b_transpose.data, partial_a.data, alpha_t, one);
            WP_TILE_SYNC();
            auto global_a = tile_global_t<TA, ShapeA>(logical_adj_A, tile_coord(i * TileM, k * TileK));
            partial_a.atomic_add(global_a);
            WP_TILE_SYNC();
        }

        if (adj_B.data) {
            auto partial_b = tile_alloc_empty<TB, ShapeB, StrideB, false>();
            auto a_transpose = tile_transpose(a);
            partitioned_gemm::scalar_matmul<
                false, typename decltype(a_transpose)::Layout, typename TileC::Layout,
                typename decltype(partial_b)::Layout>(a_transpose.data, adj_C.grad, partial_b.data, alpha_t, one);
            WP_TILE_SYNC();
            auto global_b = tile_global_t<TB, ShapeB>(logical_adj_B, tile_coord(k * TileK, j * TileN));
            partial_b.atomic_add(global_b);
            WP_TILE_SYNC();
        }
    }

    TC output_scale = TC(beta);
    if (output_scale != TC(1)) {
        for (int index = WP_TILE_THREAD_IDX; index < TileC::Layout::Size; index += WP_TILE_BLOCK_DIM)
            adj_C.grad(index) *= output_scale;
        WP_TILE_SYNC();
    }
}


template <
    int TileM,
    int TileN,
    int TileK,
    int FullM,
    int FullN,
    int FullK,
    bool TransposeA,
    bool TransposeB,
    bool Accumulate,
    bool ReuseOutputWorkspace,
    int PipelineBufferSize,
    int PipelineBufferAlignment,
    int TilePipelineStorageSize,
    int TilePipelineStorageAlignment,
    int AccumulatorStorageSize,
    int AccumulatorStorageAlignment,
    typename PipelineFunctions,
    typename TA,
    typename TB,
    typename TileC,
    typename Alpha,
    typename Beta,
    typename AdjAlpha,
    typename AdjBeta>
void adj_tile_gemm(
    PipelineFunctions functions,
    void* device_pipeline_ptr,
    array_t<TA>& A,
    array_t<TB>& B,
    int i,
    int j,
    TileC& C,
    const Alpha& alpha,
    const Beta& beta,
    PipelineFunctions adj_functions,
    void* adj_device_pipeline_ptr,
    array_t<TA>& adj_A,
    array_t<TB>& adj_B,
    int adj_i,
    int adj_j,
    TileC& adj_C,
    const AdjAlpha& adj_alpha,
    const AdjBeta& adj_beta,
    TileC& adj_ret
)
{
    (void)adj_beta;
    (void)adj_ret;
    adj_tile_gemm_impl<
        TileM, TileN, TileK, FullM, FullN, FullK, TransposeA, TransposeB, Accumulate, ReuseOutputWorkspace,
        PipelineBufferSize, PipelineBufferAlignment, TilePipelineStorageSize, TilePipelineStorageAlignment,
        AccumulatorStorageSize, AccumulatorStorageAlignment>(
        functions, device_pipeline_ptr, A, B, i, j, C, alpha, beta, adj_functions, adj_device_pipeline_ptr, adj_A,
        adj_B, adj_i, adj_j, adj_C, adj_alpha
    );
}


template <
    int TileM,
    int TileN,
    int TileK,
    int FullM,
    int FullN,
    int FullK,
    bool TransposeA,
    bool TransposeB,
    bool Accumulate,
    bool ReuseOutputWorkspace,
    int PipelineBufferSize,
    int PipelineBufferAlignment,
    int TilePipelineStorageSize,
    int TilePipelineStorageAlignment,
    int AccumulatorStorageSize,
    int AccumulatorStorageAlignment,
    typename PipelineFunctions,
    typename TA,
    typename TB,
    typename TileC,
    typename Alpha,
    typename Beta,
    typename AdjAlpha,
    typename AdjBeta>
void adj_tile_gemm_acc(
    PipelineFunctions functions,
    void* device_pipeline_ptr,
    array_t<TA>& A,
    array_t<TB>& B,
    int i,
    int j,
    TileC& C,
    const Alpha& alpha,
    const Beta& beta,
    PipelineFunctions adj_functions,
    void* adj_device_pipeline_ptr,
    array_t<TA>& adj_A,
    array_t<TB>& adj_B,
    int adj_i,
    int adj_j,
    TileC& adj_C,
    const AdjAlpha& adj_alpha,
    const AdjBeta& adj_beta
)
{
    (void)adj_beta;
    adj_tile_gemm_impl<
        TileM, TileN, TileK, FullM, FullN, FullK, TransposeA, TransposeB, Accumulate, ReuseOutputWorkspace,
        PipelineBufferSize, PipelineBufferAlignment, TilePipelineStorageSize, TilePipelineStorageAlignment,
        AccumulatorStorageSize, AccumulatorStorageAlignment>(
        functions, device_pipeline_ptr, A, B, i, j, C, alpha, beta, adj_functions, adj_device_pipeline_ptr, adj_A,
        adj_B, adj_i, adj_j, adj_C, adj_alpha
    );
}

}  // namespace wp

#ifdef __clang__
#pragma clang diagnostic pop
#endif
