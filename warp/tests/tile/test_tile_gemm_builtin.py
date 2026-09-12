# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, assert_np_equal, get_cuda_test_devices, get_test_devices

TILE_M = 8
TILE_N = 4
TILE_K = 4
BLOCK_DIM = 32
PIPELINE_TILE_M = 64
PIPELINE_TILE_N = 64
PIPELINE_TILE_K = 32
PIPELINE_BLOCK_DIM = 128
BACKWARD_TILE_M = 32
BACKWARD_TILE_N = 32
BACKWARD_TILE_K = 16
BACKWARD_BLOCK_DIM = 64
SMALL_K_TILE_K = 8
SMALL_K_BLOCK_DIM = 64
OVERSIZED_TILE_M = 64
OVERSIZED_TILE_N = 64
OVERSIZED_TILE_K = 128
OVERSIZED_BLOCK_DIM = 256
SMEM_TILE_M = 128
SMEM_TILE_N = 64
SMEM_TILE_K = 32
SMEM_FULL_K = 64
SMEM_BLOCK_DIM = 128


@wp.kernel
def tile_gemm_kernel(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
    i, j = wp.tid()
    result = wp.tile_gemm(a, b, i, j, tile=(TILE_M, TILE_N, TILE_K), shape=(13, 9, 11), alpha=0.5, dtype=float)
    wp.tile_store(c, result, offset=(i * TILE_M, j * TILE_N))


@wp.kernel
def tile_gemm_acc_kernel(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
    i, j = wp.tid()
    result = wp.tile_ones(shape=(TILE_M, TILE_N), dtype=float)
    wp.tile_gemm(
        a,
        b,
        i,
        j,
        result,
        tile=(TILE_M, TILE_N, TILE_K),
        shape=(13, 9, 11),
        alpha=0.5,
        beta=0.25,
    )
    wp.tile_store(c, result, offset=(i * TILE_M, j * TILE_N))


@wp.kernel
def tile_gemm_transpose_kernel(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
    i, j = wp.tid()
    result = wp.tile_gemm(
        a,
        b,
        i,
        j,
        tile=(TILE_M, TILE_N, TILE_K),
        shape=(13, 9, 11),
        transpose_a=True,
        transpose_b=True,
    )
    wp.tile_store(c, result, offset=(i * TILE_M, j * TILE_N))


@wp.func
def pipeline_tile_gemm(a: wp.array2d[float], b: wp.array2d[float], i: int, j: int):
    return wp.tile_gemm(
        a,
        b,
        i,
        j,
        tile=(PIPELINE_TILE_M, PIPELINE_TILE_N, PIPELINE_TILE_K),
        shape=(256, 128, 128),
    )


@wp.kernel(module="unique", enable_backward=False, module_options={"enable_backward": False})
def tile_gemm_pipeline_kernel(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
    i, j = wp.tid()
    result = pipeline_tile_gemm(a, b, i, j)
    wp.tile_store(c, result, offset=(i * PIPELINE_TILE_M, j * PIPELINE_TILE_N))


@wp.kernel
def tile_gemm_pipeline_backward_kernel(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
    i, j = wp.tid()
    result = wp.tile_gemm(
        a,
        b,
        i,
        j,
        tile=(BACKWARD_TILE_M, BACKWARD_TILE_N, BACKWARD_TILE_K),
        shape=(128, 64, 64),
    )
    wp.tile_store(c, result, offset=(i * BACKWARD_TILE_M, j * BACKWARD_TILE_N))


@wp.kernel(enable_backward=False)
def tile_gemm_pipeline_fused_kernel(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
    i, j = wp.tid()
    result = pipeline_tile_gemm(a, b, i, j)
    result = wp.tile_map(wp.sin, result)
    wp.tile_store(c, result, offset=(i * PIPELINE_TILE_M, j * PIPELINE_TILE_N))


@wp.kernel(module="unique", enable_backward=False, module_options={"enable_backward": False})
def tile_gemm_pipeline_preceding_tile_kernel(
    a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float], guard_out: wp.array2d[float]
):
    i, j = wp.tid()
    guard = wp.tile_full(shape=(3, 7), value=2.5, dtype=float, storage="shared")
    result = wp.tile_gemm(
        a,
        b,
        i,
        j,
        tile=(PIPELINE_TILE_M, PIPELINE_TILE_N, PIPELINE_TILE_K),
        shape=(256, 128, 128),
    )
    wp.tile_store(c, result, offset=(i * PIPELINE_TILE_M, j * PIPELINE_TILE_N))
    wp.tile_store(guard_out, guard, offset=(i * 3, j * 7))


@wp.kernel(module="unique", enable_backward=False, module_options={"enable_backward": False})
def tile_gemm_pipeline_beta_zero_kernel(
    a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float], scratch_out: wp.array2d[float]
):
    i, j = wp.tid()
    result = wp.tile_zeros(shape=(PIPELINE_TILE_M, PIPELINE_TILE_N), dtype=float, storage="shared")
    scratch = wp.tile_ones(shape=(PIPELINE_TILE_M, PIPELINE_TILE_N), dtype=float, storage="shared")
    wp.tile_gemm(
        a,
        b,
        i,
        j,
        result,
        tile=(PIPELINE_TILE_M, PIPELINE_TILE_N, PIPELINE_TILE_K),
        shape=(256, 128, 128),
        beta=0.0,
    )
    wp.tile_store(c, result, offset=(i * PIPELINE_TILE_M, j * PIPELINE_TILE_N))
    wp.tile_store(scratch_out, scratch, offset=(i * PIPELINE_TILE_M, j * PIPELINE_TILE_N))


@wp.kernel(module="unique", enable_backward=False, module_options={"enable_backward": False})
def tile_gemm_pipeline_accumulate_kernel(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
    i, j = wp.tid()
    result = wp.tile_ones(shape=(PIPELINE_TILE_M, PIPELINE_TILE_N), dtype=float, storage="shared")
    wp.tile_gemm(
        a,
        b,
        i,
        j,
        result,
        tile=(PIPELINE_TILE_M, PIPELINE_TILE_N, PIPELINE_TILE_K),
        shape=(256, 128, 128),
        alpha=0.5,
        beta=0.25,
    )
    wp.tile_store(c, result, offset=(i * PIPELINE_TILE_M, j * PIPELINE_TILE_N))


@wp.kernel(module="unique", enable_backward=False, module_options={"enable_backward": False})
def tile_gemm_pipeline_smem_kernel(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
    i, j = wp.tid()
    result = wp.tile_gemm(
        a,
        b,
        i,
        j,
        tile=(SMEM_TILE_M, SMEM_TILE_N, SMEM_TILE_K),
        shape=(SMEM_TILE_M, SMEM_TILE_N, SMEM_FULL_K),
    )
    wp.tile_store(c, result, offset=(i * SMEM_TILE_M, j * SMEM_TILE_N))


@wp.func
def pipeline_tile_gemm_transpose_b(a: wp.array2d[float], b: wp.array2d[float], i: int, j: int):
    return wp.tile_gemm(
        a,
        b,
        i,
        j,
        tile=(PIPELINE_TILE_M, PIPELINE_TILE_N, PIPELINE_TILE_K),
        shape=(256, 128, 64),
        transpose_b=True,
    )


@wp.func
def pipeline_tile_gemm_transpose_a(a: wp.array2d[float], b: wp.array2d[float], i: int, j: int):
    return wp.tile_gemm(
        a,
        b,
        i,
        j,
        tile=(PIPELINE_TILE_M, PIPELINE_TILE_N, PIPELINE_TILE_K),
        shape=(256, 128, 64),
        transpose_a=True,
    )


@wp.func
def pipeline_tile_gemm_transpose_ab(a: wp.array2d[float], b: wp.array2d[float], i: int, j: int):
    return wp.tile_gemm(
        a,
        b,
        i,
        j,
        tile=(PIPELINE_TILE_M, PIPELINE_TILE_N, PIPELINE_TILE_K),
        shape=(256, 128, 64),
        transpose_a=True,
        transpose_b=True,
    )


@wp.kernel(enable_backward=False)
def tile_gemm_pipeline_transpose_b_kernel(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
    i, j = wp.tid()
    result = pipeline_tile_gemm_transpose_b(a, b, i, j)
    wp.tile_store(c, result, offset=(i * PIPELINE_TILE_M, j * PIPELINE_TILE_N))


@wp.kernel(enable_backward=False)
def tile_gemm_pipeline_transpose_a_kernel(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
    i, j = wp.tid()
    result = pipeline_tile_gemm_transpose_a(a, b, i, j)
    wp.tile_store(c, result, offset=(i * PIPELINE_TILE_M, j * PIPELINE_TILE_N))


@wp.kernel(enable_backward=False)
def tile_gemm_pipeline_transpose_ab_kernel(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
    i, j = wp.tid()
    result = pipeline_tile_gemm_transpose_ab(a, b, i, j)
    wp.tile_store(c, result, offset=(i * PIPELINE_TILE_M, j * PIPELINE_TILE_N))


@wp.kernel(enable_backward=False)
def tile_gemm_pipeline_mixed_kernel(a: wp.array2d[wp.float16], b: wp.array2d[wp.float16], c: wp.array2d[float]):
    i, j = wp.tid()
    result = wp.tile_zeros(shape=(PIPELINE_TILE_M, PIPELINE_TILE_N), dtype=float, storage="shared")
    wp.tile_gemm(
        a,
        b,
        i,
        j,
        result,
        tile=(PIPELINE_TILE_M, PIPELINE_TILE_N, PIPELINE_TILE_K),
        shape=(256, 128, 64),
        beta=0.0,
    )
    wp.tile_store(c, result, offset=(i * PIPELINE_TILE_M, j * PIPELINE_TILE_N))


@wp.kernel(enable_backward=False)
def tile_gemm_pipeline_mixed_return_kernel(a: wp.array2d[wp.float16], b: wp.array2d[wp.float16], c: wp.array2d[float]):
    i, j = wp.tid()
    result = wp.tile_gemm(
        a,
        b,
        i,
        j,
        tile=(PIPELINE_TILE_M, PIPELINE_TILE_N, PIPELINE_TILE_K),
        shape=(256, 128, 64),
    )
    wp.tile_store(c, result, offset=(i * PIPELINE_TILE_M, j * PIPELINE_TILE_N))


@wp.kernel(module="unique", module_options={"enable_backward": False})
def tile_gemm_pipeline_bfloat16_return_kernel(
    a: wp.array2d[wp.bfloat16], b: wp.array2d[wp.bfloat16], c: wp.array2d[float]
):
    i, j = wp.tid()
    result = wp.tile_gemm(
        a,
        b,
        i,
        j,
        tile=(PIPELINE_TILE_M, PIPELINE_TILE_N, PIPELINE_TILE_K),
        shape=(256, 128, 64),
    )
    wp.tile_store(c, result, offset=(i * PIPELINE_TILE_M, j * PIPELINE_TILE_N))


@wp.kernel(enable_backward=False)
def tile_gemm_pipeline_small_k_kernel(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
    i, j = wp.tid()
    result = wp.tile_gemm(
        a,
        b,
        i,
        j,
        tile=(PIPELINE_TILE_M, PIPELINE_TILE_N, SMALL_K_TILE_K),
        shape=(128, 64, 32),
        transpose_b=True,
    )
    wp.tile_store(c, result, offset=(i * PIPELINE_TILE_M, j * PIPELINE_TILE_N))


@wp.func
def oversized_pipeline_tile_gemm(a: wp.array2d[float], b: wp.array2d[float], i: int, j: int):
    return wp.tile_gemm(
        a,
        b,
        i,
        j,
        tile=(OVERSIZED_TILE_M, OVERSIZED_TILE_N, OVERSIZED_TILE_K),
        shape=(128, 128, 256),
    )


@wp.kernel(enable_backward=False)
def tile_gemm_oversized_pipeline_kernel(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
    i, j = wp.tid()
    result = oversized_pipeline_tile_gemm(a, b, i, j)
    wp.tile_store(c, result, offset=(i * OVERSIZED_TILE_M, j * OVERSIZED_TILE_N))


def test_tile_gemm(test, device):
    m, n, k = 13, 9, 11
    rng = np.random.default_rng(42)
    a = rng.random((m, k), dtype=np.float32)
    b = rng.random((k, n), dtype=np.float32)

    a_wp = wp.array(a, device=device, requires_grad=True)
    b_wp = wp.array(b, device=device, requires_grad=True)
    padded_shape = ((m + TILE_M - 1) // TILE_M * TILE_M, (n + TILE_N - 1) // TILE_N * TILE_N)
    c_wp = wp.empty(padded_shape, dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_gemm_kernel,
            dim=((m + TILE_M - 1) // TILE_M, (n + TILE_N - 1) // TILE_N),
            inputs=[a_wp, b_wp],
            outputs=[c_wp],
            block_dim=BLOCK_DIM,
            device=device,
        )

    expected = np.zeros(padded_shape, dtype=np.float32)
    expected[:m, :n] = 0.5 * (a @ b)
    assert_np_equal(c_wp.numpy(), expected, tol=1.0e-5)

    adj_c = np.ones_like(expected)
    tape.backward(grads={c_wp: wp.array(adj_c, device=device)})
    assert_np_equal(a_wp.grad.numpy(), 0.5 * (adj_c[:m, :n] @ b.T), tol=1.0e-5)
    assert_np_equal(b_wp.grad.numpy(), 0.5 * (a.T @ adj_c[:m, :n]), tol=1.0e-5)


def test_tile_gemm_acc(test, device):
    m, n, k = 13, 9, 11
    rng = np.random.default_rng(17)
    a = rng.random((m, k), dtype=np.float32)
    b = rng.random((k, n), dtype=np.float32)

    a_wp = wp.array(a, device=device, requires_grad=True)
    b_wp = wp.array(b, device=device, requires_grad=True)
    padded_shape = ((m + TILE_M - 1) // TILE_M * TILE_M, (n + TILE_N - 1) // TILE_N * TILE_N)
    c_wp = wp.empty(padded_shape, dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_gemm_acc_kernel,
            dim=((m + TILE_M - 1) // TILE_M, (n + TILE_N - 1) // TILE_N),
            inputs=[a_wp, b_wp],
            outputs=[c_wp],
            block_dim=BLOCK_DIM,
            device=device,
        )

    expected = np.full(padded_shape, 0.25, dtype=np.float32)
    expected[:m, :n] += 0.5 * (a @ b)
    assert_np_equal(c_wp.numpy(), expected, tol=1.0e-5)

    adj_c = np.ones_like(expected)
    tape.backward(grads={c_wp: wp.array(adj_c, device=device)})
    assert_np_equal(a_wp.grad.numpy(), 0.5 * (adj_c[:m, :n] @ b.T), tol=1.0e-5)
    assert_np_equal(b_wp.grad.numpy(), 0.5 * (a.T @ adj_c[:m, :n]), tol=1.0e-5)


def test_tile_gemm_transpose(test, device):
    m, n, k = 13, 9, 11
    rng = np.random.default_rng(29)
    a = rng.random((k, m), dtype=np.float32)
    b = rng.random((n, k), dtype=np.float32)

    a_wp = wp.array(a, device=device, requires_grad=True)
    b_wp = wp.array(b, device=device, requires_grad=True)
    c_wp = wp.zeros((m, n), dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_gemm_transpose_kernel,
            dim=((m + TILE_M - 1) // TILE_M, (n + TILE_N - 1) // TILE_N),
            inputs=[a_wp, b_wp],
            outputs=[c_wp],
            block_dim=BLOCK_DIM,
            device=device,
        )

    expected = a.T @ b.T
    assert_np_equal(c_wp.numpy(), expected, tol=1.0e-5)

    adj_c = np.ones_like(expected)
    tape.backward(grads={c_wp: wp.array(adj_c, device=device)})
    assert_np_equal(a_wp.grad.numpy(), (adj_c @ b).T, tol=1.0e-5)
    assert_np_equal(b_wp.grad.numpy(), (a @ adj_c).T, tol=1.0e-5)


def _make_tile_gemm_output_view_kernel(view, parent_shape):
    parent_rows, parent_columns = parent_shape

    @wp.kernel(module="unique")
    def output_view_kernel(
        a: wp.array2d[float],
        b: wp.array2d[float],
        initial: wp.array2d[float],
        output: wp.array2d[float],
        alpha: float,
        beta: float,
    ):
        parent = wp.tile_load(initial, shape=(parent_rows, parent_columns), storage="shared")
        if wp.static(view == "transpose"):
            result = wp.tile_transpose(parent)
        elif wp.static(view == "slice"):
            result = parent[1:33, 1:17]
        else:
            result = wp.tile_reshape(parent[0, 1:], shape=(32, 16))
        wp.tile_gemm(a, b, 0, 0, result, tile=(32, 16, 16), shape=(32, 16, 32), alpha=alpha, beta=beta)
        wp.tile_store(output, parent)

    return output_view_kernel


def test_tile_gemm_output_views(test, device):
    rng = np.random.default_rng(67)
    a = rng.standard_normal((32, 32)).astype(np.float32)
    b = rng.standard_normal((32, 16)).astype(np.float32)
    for view, parent_shape in (("transpose", (16, 32)), ("slice", (34, 18)), ("offset", (1, 513))):
        kernel = _make_tile_gemm_output_view_kernel(view, parent_shape)

        def logical_view(parent, view=view):
            if view == "transpose":
                return parent.T
            if view == "slice":
                return parent[1:33, 1:17]
            return parent[:, 1:].reshape(32, 16)

        for beta in (0.0, 0.25):
            with test.subTest(view=view, beta=beta):
                initial = rng.standard_normal(parent_shape).astype(np.float32)
                if beta == 0.0:
                    logical_view(initial).fill(np.nan)
                expected = initial.copy()
                logical_view(expected)[:] = 0.5 * (a @ b) + (beta * logical_view(initial) if beta else 0.0)
                a_wp = wp.array(a, device=device, requires_grad=True)
                b_wp = wp.array(b, device=device, requires_grad=True)
                initial_wp = wp.array(initial, device=device, requires_grad=True)
                output_wp = wp.empty(parent_shape, dtype=float, device=device, requires_grad=True)
                with wp.Tape() as tape:
                    wp.launch_tiled(
                        kernel,
                        dim=1,
                        inputs=[a_wp, b_wp, initial_wp, output_wp, 0.5, beta],
                        block_dim=BLOCK_DIM,
                        device=device,
                    )
                np.testing.assert_allclose(output_wp.numpy(), expected, rtol=1.0e-5, atol=1.0e-5)

                adj_output = rng.standard_normal(parent_shape).astype(np.float32)
                tape.backward(grads={output_wp: wp.array(adj_output, device=device)})
                adj_product = logical_view(adj_output)
                np.testing.assert_allclose(a_wp.grad.numpy(), 0.5 * (adj_product @ b.T), rtol=1.0e-5, atol=1.0e-5)
                np.testing.assert_allclose(b_wp.grad.numpy(), 0.5 * (a.T @ adj_product), rtol=1.0e-5, atol=1.0e-5)
                adj_initial = adj_output.copy()
                logical_view(adj_initial)[:] *= beta
                np.testing.assert_allclose(initial_wp.grad.numpy(), adj_initial, rtol=1.0e-5, atol=1.0e-5)
                test.assertFalse(kernel.module.load(wp.get_device(device), BLOCK_DIM).gemm_pipeline_resources)


@wp.kernel(module="unique")
def tile_gemm_mixed_backward_kernel(a: wp.array2d[wp.float16], b: wp.array2d[wp.float16], c: wp.array2d[float]):
    result = wp.tile_gemm(a, b, 0, 0, tile=(2, 2, 2), shape=(2, 2, 2))
    wp.tile_store(c, result)


def test_tile_gemm_mixed_backward(test, device):
    a = np.full((2, 2), 0.001, dtype=np.float16)
    b = a.copy()
    a_wp = wp.array(a, device=device, requires_grad=True)
    b_wp = wp.array(b, device=device, requires_grad=True)
    c_wp = wp.empty((2, 2), dtype=float, device=device, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_gemm_mixed_backward_kernel,
            dim=1,
            inputs=[a_wp, b_wp],
            outputs=[c_wp],
            block_dim=BLOCK_DIM,
            device=device,
        )
    np.testing.assert_allclose(c_wp.numpy(), a.astype(np.float32) @ b.astype(np.float32), rtol=1.0e-6)

    # The upstream gradient overflows FP16, but both input gradients are finite.
    adj_c = np.full((2, 2), 70000.0, dtype=np.float32)
    tape.backward(grads={c_wp: wp.array(adj_c, device=device)})
    np.testing.assert_array_equal(a_wp.grad.numpy(), (adj_c @ b.astype(np.float32).T).astype(np.float16))
    np.testing.assert_array_equal(b_wp.grad.numpy(), (a.astype(np.float32).T @ adj_c).astype(np.float16))


def _make_tile_gemm_float64_kernel(shape, tile, transpose_a, transpose_b):
    m, n, k = shape
    tile_m, tile_n, tile_k = tile

    @wp.kernel(module="unique")
    def float64_kernel(a: wp.array2d[wp.float64], b: wp.array2d[wp.float64], c: wp.array2d[wp.float64]):
        i, j = wp.tid()
        result = wp.tile_gemm(
            a,
            b,
            i,
            j,
            tile=(tile_m, tile_n, tile_k),
            shape=(m, n, k),
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            alpha=0.5,
        )
        wp.tile_store(c, result, offset=(i * tile_m, j * tile_n))

    return float64_kernel


def test_tile_gemm_float64(test, device):
    rng = np.random.default_rng(61)
    wp_device = wp.get_device(device)
    supports_pipeline = (
        wp_device.is_cuda
        and wp_device.arch >= 100
        and wp._src.context.runtime.toolkit_version >= (13, 1)
        and wp._src.context.runtime.core.wp_is_mathdx_enabled()
    )

    for m, n, k in ((13, 9, 11), (32, 32, 32)):
        tile_m, tile_n, tile_k = (8, 4, 4) if m == 13 else (16, 16, 16)
        for transpose_a, transpose_b in ((False, False), (False, True), (True, False), (True, True)):
            with test.subTest(shape=(m, n, k), transpose_a=transpose_a, transpose_b=transpose_b):
                float64_kernel = _make_tile_gemm_float64_kernel(
                    (m, n, k), (tile_m, tile_n, tile_k), transpose_a, transpose_b
                )
                a = rng.standard_normal((k, m) if transpose_a else (m, k))
                b = rng.standard_normal((n, k) if transpose_b else (k, n))
                logical_a = a.T if transpose_a else a
                logical_b = b.T if transpose_b else b
                a_wp = wp.array(a, device=device, requires_grad=True)
                b_wp = wp.array(b, device=device, requires_grad=True)
                c_wp = wp.empty((m, n), dtype=wp.float64, device=device, requires_grad=True)

                with wp.Tape() as tape:
                    wp.launch_tiled(
                        float64_kernel,
                        dim=((m + tile_m - 1) // tile_m, (n + tile_n - 1) // tile_n),
                        inputs=[a_wp, b_wp],
                        outputs=[c_wp],
                        block_dim=BLOCK_DIM,
                        device=device,
                    )
                np.testing.assert_allclose(c_wp.numpy(), 0.5 * (logical_a @ logical_b), rtol=1.0e-12, atol=1.0e-12)

                adj_c = rng.standard_normal((m, n))
                tape.backward(grads={c_wp: wp.array(adj_c, device=device)})
                adj_a = 0.5 * (adj_c @ logical_b.T)
                adj_b = 0.5 * (logical_a.T @ adj_c)
                np.testing.assert_allclose(
                    a_wp.grad.numpy(), adj_a.T if transpose_a else adj_a, rtol=1.0e-12, atol=1.0e-12
                )
                np.testing.assert_allclose(
                    b_wp.grad.numpy(), adj_b.T if transpose_b else adj_b, rtol=1.0e-12, atol=1.0e-12
                )

                if supports_pipeline and m == 32:
                    module_exec = float64_kernel.module.load(wp_device, BLOCK_DIM)
                    test.assertTrue(module_exec.gemm_pipeline_resources)


def test_tile_gemm_pipeline(test, device):
    m, n, k = 256, 128, 128
    rng = np.random.default_rng(23)
    a = rng.random((m, k), dtype=np.float32)
    b = rng.random((k, n), dtype=np.float32)

    a_wp = wp.array(a, device=device)
    b_wp = wp.array(b, device=device)
    c_wp = wp.empty((m, n), dtype=float, device=device)
    launch = {
        "kernel": tile_gemm_pipeline_kernel,
        "dim": (m // PIPELINE_TILE_M, n // PIPELINE_TILE_N),
        "inputs": [a_wp, b_wp],
        "outputs": [c_wp],
        "block_dim": PIPELINE_BLOCK_DIM,
        "device": device,
    }

    # The first launch creates the device pipeline. Subsequent launches, including
    # graph capture, reuse the cached pipeline bound to these arrays.
    wp.launch_tiled(**launch)
    assert_np_equal(c_wp.numpy(), a @ b, tol=2.0e-4)

    # A sliced, misaligned input cannot use the device pipeline and must run
    # through the portable path in the same compiled kernel.
    a_padded = np.zeros((m, k + 1), dtype=np.float32)
    a_padded[:, 1:] = a
    a_padded_wp = wp.array(a_padded, device=device)
    a_sliced_wp = a_padded_wp[:, 1:]
    c_sliced_wp = wp.empty((m, n), dtype=float, device=device)
    wp.launch_tiled(
        tile_gemm_pipeline_kernel,
        dim=(m // PIPELINE_TILE_M, n // PIPELINE_TILE_N),
        inputs=[a_sliced_wp, b_wp],
        outputs=[c_sliced_wp],
        block_dim=PIPELINE_BLOCK_DIM,
        device=device,
    )
    assert_np_equal(c_sliced_wp.numpy(), a @ b, tol=2.0e-4)

    wp.launch_tiled(
        tile_gemm_pipeline_fused_kernel,
        dim=(m // PIPELINE_TILE_M, n // PIPELINE_TILE_N),
        inputs=[a_wp, b_wp],
        outputs=[c_wp],
        block_dim=PIPELINE_BLOCK_DIM,
        device=device,
    )
    assert_np_equal(c_wp.numpy(), np.sin(a @ b), tol=2.0e-4)

    scratch_wp = wp.empty((m, n), dtype=float, device=device)
    wp.launch_tiled(
        tile_gemm_pipeline_beta_zero_kernel,
        dim=(m // PIPELINE_TILE_M, n // PIPELINE_TILE_N),
        inputs=[a_wp, b_wp],
        outputs=[c_wp, scratch_wp],
        block_dim=PIPELINE_BLOCK_DIM,
        device=device,
    )
    assert_np_equal(c_wp.numpy(), a @ b, tol=2.0e-4)
    assert_np_equal(scratch_wp.numpy(), np.ones((m, n), dtype=np.float32), tol=0.0)

    # The returning tile begins after a live allocation with a rounded size.
    # Its pipeline workspace must retain alignment and preserve the earlier tile.
    guard_wp = wp.empty((m // PIPELINE_TILE_M * 3, n // PIPELINE_TILE_N * 7), dtype=float, device=device)
    wp.launch_tiled(
        tile_gemm_pipeline_preceding_tile_kernel,
        dim=(m // PIPELINE_TILE_M, n // PIPELINE_TILE_N),
        inputs=[a_wp, b_wp],
        outputs=[c_wp, guard_wp],
        block_dim=PIPELINE_BLOCK_DIM,
        device=device,
    )
    np.testing.assert_allclose(c_wp.numpy(), a @ b, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_array_equal(guard_wp.numpy(), np.full(guard_wp.shape, 2.5, dtype=np.float32))

    wp.launch_tiled(
        tile_gemm_pipeline_accumulate_kernel,
        dim=(m // PIPELINE_TILE_M, n // PIPELINE_TILE_N),
        inputs=[a_wp, b_wp],
        outputs=[c_wp],
        block_dim=PIPELINE_BLOCK_DIM,
        device=device,
    )
    assert_np_equal(c_wp.numpy(), 0.5 * (a @ b) + 0.25, tol=2.0e-4)

    wp.launch_tiled(**launch)
    assert_np_equal(c_wp.numpy(), a @ b, tol=2.0e-4)

    wp_device = wp.get_device(device)
    supports_pipeline = (
        wp_device.is_cuda
        and wp_device.arch >= 100
        and wp._src.context.runtime.toolkit_version >= (13, 1)
        and wp._src.context.runtime.core.wp_is_mathdx_enabled()
    )
    if supports_pipeline:
        module_exec = tile_gemm_pipeline_kernel.module.load(wp_device, PIPELINE_BLOCK_DIM)
        hooks = module_exec.get_kernel_hooks(tile_gemm_pipeline_kernel)
        test.assertTrue(hooks.gemm_pipeline_enabled)

        beta_zero_module_exec = tile_gemm_pipeline_beta_zero_kernel.module.load(wp_device, PIPELINE_BLOCK_DIM)
        beta_zero_hooks = beta_zero_module_exec.get_kernel_hooks(tile_gemm_pipeline_beta_zero_kernel)
        test.assertTrue(beta_zero_hooks.gemm_pipeline_enabled)
        output_bytes = PIPELINE_TILE_M * PIPELINE_TILE_N * wp.types.type_size_in_bytes(wp.float32)
        test.assertEqual(beta_zero_hooks.forward_smem_bytes - hooks.forward_smem_bytes, output_bytes)

        accumulate_module_exec = tile_gemm_pipeline_accumulate_kernel.module.load(wp_device, PIPELINE_BLOCK_DIM)
        accumulate_hooks = accumulate_module_exec.get_kernel_hooks(tile_gemm_pipeline_accumulate_kernel)
        test.assertTrue(accumulate_hooks.gemm_pipeline_enabled)
        test.assertEqual(accumulate_hooks.forward_smem_bytes, beta_zero_hooks.forward_smem_bytes - output_bytes)

        with wp.ScopedCapture(wp_device, force_module_load=False) as capture:
            wp.launch_tiled(**launch)
        wp.capture_launch(capture.graph)
        assert_np_equal(c_wp.numpy(), a @ b, tol=2.0e-4)

        a_next = rng.random((m, k), dtype=np.float32)
        b_next = rng.random((k, n), dtype=np.float32)
        a_next_wp = wp.array(a_next, device=device)
        b_next_wp = wp.array(b_next, device=device)
        c_next_wp = wp.empty((m, n), dtype=float, device=device)
        command = wp.launch_tiled(**launch, record_cmd=True)
        command.set_param_by_name("a", a_next_wp)
        command.set_param_by_name("b", b_next_wp)
        command.set_param_by_name("c", c_next_wp)
        command.launch()
        assert_np_equal(c_next_wp.numpy(), a_next @ b_next, tol=2.0e-4)

        with wp.ScopedCapture(wp_device, force_module_load=False) as capture:
            command.launch()
        wp.capture_launch(capture.graph)
        assert_np_equal(c_next_wp.numpy(), a_next @ b_next, tol=2.0e-4)


def _test_tile_gemm_pipeline_transpose(test, device, kernel, transpose_a, transpose_b, seed):
    m, n, k = 256, 128, 64
    rng = np.random.default_rng(seed)
    a = rng.random((k, m) if transpose_a else (m, k), dtype=np.float32)
    b = rng.random((n, k) if transpose_b else (k, n), dtype=np.float32)

    a_wp = wp.array(a, device=device)
    b_wp = wp.array(b, device=device)
    c_wp = wp.empty((m, n), dtype=float, device=device)
    wp.launch_tiled(
        kernel,
        dim=(m // PIPELINE_TILE_M, n // PIPELINE_TILE_N),
        inputs=[a_wp, b_wp],
        outputs=[c_wp],
        block_dim=PIPELINE_BLOCK_DIM,
        device=device,
    )
    logical_a = a.T if transpose_a else a
    logical_b = b.T if transpose_b else b
    assert_np_equal(c_wp.numpy(), logical_a @ logical_b, tol=2.0e-4)

    wp_device = wp.get_device(device)
    if (
        wp_device.is_cuda
        and wp_device.arch >= 100
        and wp._src.context.runtime.toolkit_version >= (13, 1)
        and wp._src.context.runtime.core.wp_is_mathdx_enabled()
    ):
        module_exec = kernel.module.load(wp_device, PIPELINE_BLOCK_DIM)
        hooks = module_exec.get_kernel_hooks(kernel)
        test.assertTrue(hooks.gemm_pipeline_enabled)
        pipeline_symbols = {item["lto_symbol"] for item in hooks.gemm_pipeline_meta}
        test.assertTrue(any(key[0] in pipeline_symbols for key in module_exec.gemm_pipeline_resources))


def test_tile_gemm_pipeline_backward(test, device):
    m, n, k = 128, 64, 64
    rng = np.random.default_rng(59)
    a = rng.random((m, k), dtype=np.float32)
    b = rng.random((k, n), dtype=np.float32)
    a_wp = wp.array(a, device=device, requires_grad=True)
    b_wp = wp.array(b, device=device, requires_grad=True)
    c_wp = wp.empty((m, n), dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_gemm_pipeline_backward_kernel,
            dim=(m // BACKWARD_TILE_M, n // BACKWARD_TILE_N),
            inputs=[a_wp, b_wp],
            outputs=[c_wp],
            block_dim=BACKWARD_BLOCK_DIM,
            device=device,
        )
    assert_np_equal(c_wp.numpy(), a @ b, tol=2.0e-4)

    adj_c = np.ones((m, n), dtype=np.float32)
    tape.backward(grads={c_wp: wp.array(adj_c, device=device)})
    assert_np_equal(a_wp.grad.numpy(), adj_c @ b.T, tol=2.0e-4)
    assert_np_equal(b_wp.grad.numpy(), a.T @ adj_c, tol=2.0e-4)

    wp_device = wp.get_device(device)
    if (
        wp_device.is_cuda
        and wp_device.arch >= 100
        and wp._src.context.runtime.toolkit_version >= (13, 1)
        and wp._src.context.runtime.core.wp_is_mathdx_enabled()
    ):
        module_exec = tile_gemm_pipeline_backward_kernel.module.load(wp_device, BACKWARD_BLOCK_DIM)
        hooks = module_exec.get_kernel_hooks(tile_gemm_pipeline_backward_kernel)
        test.assertTrue(hooks.gemm_pipeline_enabled)


def test_tile_gemm_pipeline_transpose_a(test, device):
    _test_tile_gemm_pipeline_transpose(test, device, tile_gemm_pipeline_transpose_a_kernel, True, False, 37)


def test_tile_gemm_pipeline_transpose_b(test, device):
    _test_tile_gemm_pipeline_transpose(test, device, tile_gemm_pipeline_transpose_b_kernel, False, True, 31)


def test_tile_gemm_pipeline_transpose_ab(test, device):
    _test_tile_gemm_pipeline_transpose(test, device, tile_gemm_pipeline_transpose_ab_kernel, True, True, 43)


def test_tile_gemm_pipeline_mixed(test, device):
    m, n, k = 256, 128, 64
    rng = np.random.default_rng(41)
    a = rng.random((m, k), dtype=np.float32).astype(np.float16)
    b = rng.random((k, n), dtype=np.float32).astype(np.float16)

    a_wp = wp.array(a, device=device)
    b_wp = wp.array(b, device=device)
    c_wp = wp.empty((m, n), dtype=float, device=device)
    wp.launch_tiled(
        tile_gemm_pipeline_mixed_kernel,
        dim=(m // PIPELINE_TILE_M, n // PIPELINE_TILE_N),
        inputs=[a_wp, b_wp],
        outputs=[c_wp],
        block_dim=PIPELINE_BLOCK_DIM,
        device=device,
    )
    assert_np_equal(c_wp.numpy(), a.astype(np.float32) @ b.astype(np.float32), tol=2.0e-3)

    wp.launch_tiled(
        tile_gemm_pipeline_mixed_return_kernel,
        dim=(m // PIPELINE_TILE_M, n // PIPELINE_TILE_N),
        inputs=[a_wp, b_wp],
        outputs=[c_wp],
        block_dim=PIPELINE_BLOCK_DIM,
        device=device,
    )
    assert_np_equal(c_wp.numpy(), a.astype(np.float32) @ b.astype(np.float32), tol=2.0e-3)

    wp_device = wp.get_device(device)
    if (
        wp_device.is_cuda
        and wp_device.arch >= 100
        and wp._src.context.runtime.toolkit_version >= (13, 1)
        and wp._src.context.runtime.core.wp_is_mathdx_enabled()
    ):
        module_exec = tile_gemm_pipeline_mixed_kernel.module.load(wp_device, PIPELINE_BLOCK_DIM)
        test.assertTrue(module_exec.gemm_pipeline_resources)
        return_module_exec = tile_gemm_pipeline_mixed_return_kernel.module.load(wp_device, PIPELINE_BLOCK_DIM)
        test.assertTrue(return_module_exec.gemm_pipeline_resources)


def test_tile_gemm_pipeline_small_k(test, device):
    m, n, k = 128, 64, 32
    rng = np.random.default_rng(53)
    a = rng.random((m, k), dtype=np.float32)
    b = rng.random((n, k), dtype=np.float32)

    a_wp = wp.array(a, device=device)
    b_wp = wp.array(b, device=device)
    c_wp = wp.empty((m, n), dtype=float, device=device)
    wp.launch_tiled(
        tile_gemm_pipeline_small_k_kernel,
        dim=(m // PIPELINE_TILE_M, n // PIPELINE_TILE_N),
        inputs=[a_wp, b_wp],
        outputs=[c_wp],
        block_dim=SMALL_K_BLOCK_DIM,
        device=device,
    )
    assert_np_equal(c_wp.numpy(), a @ b.T, tol=2.0e-4)

    wp_device = wp.get_device(device)
    if (
        wp_device.is_cuda
        and wp_device.arch >= 100
        and wp._src.context.runtime.toolkit_version >= (13, 1)
        and wp._src.context.runtime.core.wp_is_mathdx_enabled()
    ):
        module_exec = tile_gemm_pipeline_small_k_kernel.module.load(wp_device, SMALL_K_BLOCK_DIM)
        test.assertTrue(module_exec.gemm_pipeline_resources)


def test_tile_gemm_pipeline_bfloat16(test, device):
    m, n, k = 256, 128, 64
    a_wp = wp.ones((m, k), dtype=wp.bfloat16, device=device)
    b_wp = wp.ones((k, n), dtype=wp.bfloat16, device=device)
    c_wp = wp.empty((m, n), dtype=float, device=device)
    wp.launch_tiled(
        tile_gemm_pipeline_bfloat16_return_kernel,
        dim=(m // PIPELINE_TILE_M, n // PIPELINE_TILE_N),
        inputs=[a_wp, b_wp],
        outputs=[c_wp],
        block_dim=PIPELINE_BLOCK_DIM,
        device=device,
    )
    assert_np_equal(c_wp.numpy(), np.full((m, n), k, dtype=np.float32), tol=0.0)

    wp_device = wp.get_device(device)
    if (
        wp_device.is_cuda
        and wp_device.arch >= 100
        and wp._src.context.runtime.toolkit_version >= (13, 1)
        and wp._src.context.runtime.core.wp_is_mathdx_enabled()
    ):
        module_exec = tile_gemm_pipeline_bfloat16_return_kernel.module.load(wp_device, PIPELINE_BLOCK_DIM)
        test.assertTrue(module_exec.gemm_pipeline_resources)


def test_tile_gemm_oversized_pipeline_fallback(test, device):
    m, n, k = 128, 128, 256
    rng = np.random.default_rng(47)
    a = rng.random((m, k), dtype=np.float32)
    b = rng.random((k, n), dtype=np.float32)

    a_wp = wp.array(a, device=device)
    b_wp = wp.array(b, device=device)
    c_wp = wp.empty((m, n), dtype=float, device=device)
    wp.launch_tiled(
        tile_gemm_oversized_pipeline_kernel,
        dim=(m // OVERSIZED_TILE_M, n // OVERSIZED_TILE_N),
        inputs=[a_wp, b_wp],
        outputs=[c_wp],
        block_dim=OVERSIZED_BLOCK_DIM,
        device=device,
    )
    assert_np_equal(c_wp.numpy(), a @ b, tol=2.0e-4)

    wp_device = wp.get_device(device)
    module_exec = tile_gemm_oversized_pipeline_kernel.module.load(wp_device, OVERSIZED_BLOCK_DIM)
    hooks = module_exec.get_kernel_hooks(tile_gemm_oversized_pipeline_kernel)
    forward_name = wp._src.codegen.cuda_kernel_forward_name(tile_gemm_oversized_pipeline_kernel)
    pipeline_smem = module_exec.meta[forward_name + "_smem_bytes"]
    fallback_smem = module_exec.meta[forward_name + "_gemm_fallback_smem_bytes"]

    # This is the regression case: the generated pipeline is too large for the
    # device, while the scalar implementation still fits and must remain usable.
    if pipeline_smem > wp_device.max_shared_memory_per_block:
        test.assertLess(fallback_smem, pipeline_smem)
        test.assertEqual(hooks.forward_smem_bytes, fallback_smem)
        test.assertFalse(hooks.gemm_pipeline_enabled)
        pipeline_symbols = {item["lto_symbol"] for item in hooks.gemm_pipeline_meta}
        test.assertFalse(any(key[0] in pipeline_symbols for key in module_exec.gemm_pipeline_resources))


def test_tile_gemm_pipeline_launch_smem(test, device):
    rng = np.random.default_rng(53)
    a = rng.random((SMEM_TILE_M, SMEM_FULL_K), dtype=np.float32)
    b = rng.random((SMEM_FULL_K, SMEM_TILE_N), dtype=np.float32)
    a_wp = wp.array(a, device=device)
    b_wp = wp.array(b, device=device)
    c_wp = wp.empty((SMEM_TILE_M, SMEM_TILE_N), dtype=float, device=device)
    launch = {
        "kernel": tile_gemm_pipeline_smem_kernel,
        "dim": (1, 1),
        "inputs": [a_wp, b_wp],
        "outputs": [c_wp],
        "block_dim": SMEM_BLOCK_DIM,
        "device": device,
    }
    wp.launch_tiled(**launch)
    assert_np_equal(c_wp.numpy(), a @ b, tol=2.0e-4)

    wp_device = wp.get_device(device)
    supports_pipeline = (
        wp_device.arch >= 100
        and wp._src.context.runtime.toolkit_version >= (13, 1)
        and wp._src.context.runtime.core.wp_is_mathdx_enabled()
    )
    if not supports_pipeline:
        return

    module_exec = tile_gemm_pipeline_smem_kernel.module.load(wp_device, SMEM_BLOCK_DIM)
    hooks = module_exec.get_kernel_hooks(tile_gemm_pipeline_smem_kernel)
    test.assertTrue(hooks.gemm_pipeline_enabled)
    test.assertLess(hooks.gemm_pipeline_smem_bytes, hooks.forward_smem_bytes)

    command = wp.launch_tiled(**launch, record_cmd=True)
    command.launch()
    test.assertEqual(command._forward_smem_bytes, hooks.gemm_pipeline_smem_bytes)

    a_padded = np.zeros((SMEM_TILE_M, SMEM_FULL_K + 1), dtype=np.float32)
    a_padded[:, 1:] = a
    a_sliced_wp = wp.array(a_padded, device=device)[:, 1:]
    c_sliced_wp = wp.empty_like(c_wp)
    command.set_param_by_name("a", a_sliced_wp)
    command.set_param_by_name("c", c_sliced_wp)
    command.launch()
    test.assertEqual(command._forward_smem_bytes, hooks.forward_smem_bytes)
    assert_np_equal(c_sliced_wp.numpy(), a @ b, tol=2.0e-4)


class TestTileGemm(unittest.TestCase):
    pass


add_function_test(TestTileGemm, "test_tile_gemm", test_tile_gemm, devices=get_test_devices())
add_function_test(TestTileGemm, "test_tile_gemm_acc", test_tile_gemm_acc, devices=get_test_devices())
add_function_test(TestTileGemm, "test_tile_gemm_transpose", test_tile_gemm_transpose, devices=get_test_devices())
add_function_test(TestTileGemm, "test_tile_gemm_output_views", test_tile_gemm_output_views, devices=get_test_devices())
add_function_test(
    TestTileGemm, "test_tile_gemm_mixed_backward", test_tile_gemm_mixed_backward, devices=get_test_devices()
)
add_function_test(TestTileGemm, "test_tile_gemm_float64", test_tile_gemm_float64, devices=get_test_devices())
add_function_test(TestTileGemm, "test_tile_gemm_pipeline", test_tile_gemm_pipeline, devices=get_test_devices())
add_function_test(
    TestTileGemm,
    "test_tile_gemm_oversized_pipeline_fallback",
    test_tile_gemm_oversized_pipeline_fallback,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestTileGemm,
    "test_tile_gemm_pipeline_launch_smem",
    test_tile_gemm_pipeline_launch_smem,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestTileGemm,
    "test_tile_gemm_pipeline_mixed",
    test_tile_gemm_pipeline_mixed,
    devices=get_test_devices(),
)
add_function_test(
    TestTileGemm,
    "test_tile_gemm_pipeline_backward",
    test_tile_gemm_pipeline_backward,
    devices=get_test_devices(),
)
add_function_test(
    TestTileGemm,
    "test_tile_gemm_pipeline_bfloat16",
    test_tile_gemm_pipeline_bfloat16,
    devices=get_test_devices(),
)
add_function_test(
    TestTileGemm,
    "test_tile_gemm_pipeline_small_k",
    test_tile_gemm_pipeline_small_k,
    devices=get_test_devices(),
)
add_function_test(
    TestTileGemm,
    "test_tile_gemm_pipeline_transpose_a",
    test_tile_gemm_pipeline_transpose_a,
    devices=get_test_devices(),
)
add_function_test(
    TestTileGemm,
    "test_tile_gemm_pipeline_transpose_b",
    test_tile_gemm_pipeline_transpose_b,
    devices=get_test_devices(),
)
add_function_test(
    TestTileGemm,
    "test_tile_gemm_pipeline_transpose_ab",
    test_tile_gemm_pipeline_transpose_ab,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
