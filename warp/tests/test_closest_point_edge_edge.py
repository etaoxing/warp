# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

epsilon = 0.00001


@wp.kernel
def closest_point_edge_edge_kernel(
    p1: wp.array(dtype=wp.vec3),
    q1: wp.array(dtype=wp.vec3),
    p2: wp.array(dtype=wp.vec3),
    q2: wp.array(dtype=wp.vec3),
    epsilon: float,
    st0: wp.array(dtype=wp.vec3),
    c1: wp.array(dtype=wp.vec3),
    c2: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    st = wp.closest_point_edge_edge(p1[tid], q1[tid], p2[tid], q2[tid], epsilon)
    s = st[0]
    t = st[1]
    st0[tid] = st
    c1[tid] = p1[tid] + (q1[tid] - p1[tid]) * s
    c2[tid] = p2[tid] + (q2[tid] - p2[tid]) * t


def closest_point_edge_edge_launch(p1, q1, p2, q2, epsilon, st0, c1, c2, device):
    n = len(p1)
    wp.launch(
        kernel=closest_point_edge_edge_kernel,
        dim=n,
        inputs=[p1, q1, p2, q2, epsilon],
        outputs=[st0, c1, c2],
        device=device,
    )


def run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device):
    p1 = wp.array(p1_h, dtype=wp.vec3, device=device)
    q1 = wp.array(q1_h, dtype=wp.vec3, device=device)
    p2 = wp.array(p2_h, dtype=wp.vec3, device=device)
    q2 = wp.array(q2_h, dtype=wp.vec3, device=device)
    st0 = wp.empty_like(p1)
    c1 = wp.empty_like(p1)
    c2 = wp.empty_like(p1)

    closest_point_edge_edge_launch(p1, q1, p2, q2, epsilon, st0, c1, c2, device)

    wp.synchronize()
    view = st0.numpy()
    return view


def test_edge_edge_middle_crossing(test, device):
    p1_h = np.array([[0, 0, 0]])
    q1_h = np.array([[1, 1, 0]])
    p2_h = np.array([[0, 1, 0]])
    q2_h = np.array([[1, 0, 0]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 0.5)  # s value
    test.assertAlmostEqual(st0[1], 0.5)  # t value


def test_edge_edge_parallel_s1_t0(test, device):
    p1_h = np.array([[0, 0, 0]])
    q1_h = np.array([[1, 1, 0]])
    p2_h = np.array([[2, 2, 0]])
    q2_h = np.array([[3, 3, 0]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 1.0)  # s value
    test.assertAlmostEqual(st0[1], 0.0)  # t value


def test_edge_edge_parallel_s0_t1(test, device):
    p1_h = np.array([[0, 0, 0]])
    q1_h = np.array([[1, 1, 0]])
    p2_h = np.array([[-2, -2, 0]])
    q2_h = np.array([[-1, -1, 0]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 0.0)  # s value
    test.assertAlmostEqual(st0[1], 1.0)  # t value


def test_edge_edge_both_degenerate_case(test, device):
    p1_h = np.array([[0, 0, 0]])
    q1_h = np.array([[0, 0, 0]])
    p2_h = np.array([[1, 1, 1]])
    q2_h = np.array([[1, 1, 1]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 0.0)  # s value
    test.assertAlmostEqual(st0[1], 0.0)  # t value


def test_edge_edge_degenerate_first_edge(test, device):
    p1_h = np.array([[0, 0, 0]])
    q1_h = np.array([[0, 0, 0]])
    p2_h = np.array([[0, 1, 0]])
    q2_h = np.array([[1, 0, 0]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 0.0)  # s value
    test.assertAlmostEqual(st0[1], 0.5)  # t value


def test_edge_edge_degenerate_second_edge(test, device):
    p1_h = np.array([[1, 0, 0]])
    q1_h = np.array([[0, 1, 0]])
    p2_h = np.array([[1, 1, 0]])
    q2_h = np.array([[1, 1, 0]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 0.5)  # s value
    test.assertAlmostEqual(st0[1], 0.0)  # t value


def test_edge_edge_parallel(test, device):
    p1_h = np.array([[0, 0, 0]])
    q1_h = np.array([[1, 0, 0]])
    p2_h = np.array([[-0.5, 1, 0]])
    q2_h = np.array([[0.5, 1, 0]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 0.0)  # s value
    test.assertAlmostEqual(st0[1], 0.5)  # t value


def test_edge_edge_perpendicular_s1_t0(test, device):
    p1_h = np.array([[0, 0, 0]])
    q1_h = np.array([[1, 1, 0]])
    p2_h = np.array([[10, 1, 0]])
    q2_h = np.array([[11, 0, 0]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 1.0)  # s value
    test.assertAlmostEqual(st0[1], 0.0)  # t value


def test_edge_edge_perpendicular_s0_t1(test, device):
    p1_h = np.array([[0, 0, 0]])
    q1_h = np.array([[1, 1, 0]])
    p2_h = np.array([[-11, -1, 0]])
    q2_h = np.array([[-5, 0, 0]])

    res = run_closest_point_edge_edge(p1_h, q1_h, p2_h, q2_h, device)
    st0 = res[0]
    test.assertAlmostEqual(st0[0], 0.0)  # s value
    test.assertAlmostEqual(st0[1], 1.0)  # t value


@wp.func
def check_edge_closest_point_sufficient_necessary(c1: wp.vec3, c2: wp.vec3, t: float, p: wp.vec3, q: wp.vec3):
    """
    This is a sufficient and necessary condition of closest point
    c1: closest point on the other edge
    c2: closest point on edge p-q
    t: c2 = (1.0-t) * p + t * q
    e1, e2: end points of the edge
    """
    eps = 1e-5
    e = p - q
    if t == 0.0:
        wp.expect_eq(wp.dot(c1 - p, p - q) > -eps, True)
        wp.expect_eq(wp.abs(wp.length(c2 - p)) < eps, True)
    elif t == 1.0:
        wp.expect_eq(wp.dot(c1 - q, q - p) > -eps, True)
        wp.expect_eq(wp.abs(wp.length(c2 - q)) < eps, True)
    else:
        # interior closest point, c1c2 must be perpendicular to e
        c1c2 = c1 - c2
        wp.expect_eq(wp.abs(wp.dot(c1c2, e)) < eps, True)


@wp.kernel
def check_edge_closest_point_sufficient_necessary_kernel(
    p1s: wp.array(dtype=wp.vec3),
    q1s: wp.array(dtype=wp.vec3),
    p2s: wp.array(dtype=wp.vec3),
    q2s: wp.array(dtype=wp.vec3),
    epsilon: float,
):
    tid = wp.tid()

    p1 = p1s[tid]
    q1 = q1s[tid]
    p2 = p2s[tid]
    q2 = q2s[tid]

    st = wp.closest_point_edge_edge(p1, q1, p2, q2, epsilon)
    s = st[0]
    t = st[1]
    c1 = p1 + (q1 - p1) * s
    c2 = p2 + (q2 - p2) * t

    check_edge_closest_point_sufficient_necessary(c1, c2, t, p2, q2)
    check_edge_closest_point_sufficient_necessary(c2, c1, s, p1, q1)


def check_edge_closest_point_random(test, device):
    num_tests = 100000
    rng = np.random.default_rng(123)
    p1 = wp.array(rng.standard_normal(size=(num_tests, 3)), dtype=wp.vec3, device=device)
    q1 = wp.array(rng.standard_normal(size=(num_tests, 3)), dtype=wp.vec3, device=device)

    p2 = wp.array(rng.standard_normal(size=(num_tests, 3)), dtype=wp.vec3, device=device)
    q2 = wp.array(rng.standard_normal(size=(num_tests, 3)), dtype=wp.vec3, device=device)

    wp.launch(
        kernel=check_edge_closest_point_sufficient_necessary_kernel,
        dim=num_tests,
        inputs=[p1, q1, p2, q2, epsilon],
        device=device,
    )

    # parallel edges
    p1 = rng.standard_normal(size=(num_tests, 3))
    q1 = rng.standard_normal(size=(num_tests, 3))

    shifts = rng.standard_normal(size=(num_tests, 3))

    p2 = p1 + shifts
    q2 = q1 + shifts

    p1 = wp.array(p1, dtype=wp.vec3, device=device)
    q1 = wp.array(q1, dtype=wp.vec3, device=device)

    p2 = wp.array(p2, dtype=wp.vec3, device=device)
    q2 = wp.array(q2, dtype=wp.vec3, device=device)

    wp.launch(
        kernel=check_edge_closest_point_sufficient_necessary_kernel,
        dim=num_tests,
        inputs=[p1, q1, p2, q2, epsilon],
        device=device,
    )


devices = get_test_devices()


class TestClosestPointEdgeEdgeMethods(unittest.TestCase):
    pass


add_function_test(
    TestClosestPointEdgeEdgeMethods,
    "test_edge_edge_middle_crossing",
    test_edge_edge_middle_crossing,
    devices=devices,
)
add_function_test(
    TestClosestPointEdgeEdgeMethods, "test_edge_edge_parallel_s1_t0", test_edge_edge_parallel_s1_t0, devices=devices
)
add_function_test(
    TestClosestPointEdgeEdgeMethods, "test_edge_edge_parallel_s0_t1", test_edge_edge_parallel_s0_t1, devices=devices
)
add_function_test(
    TestClosestPointEdgeEdgeMethods,
    "test_edge_edge_both_degenerate_case",
    test_edge_edge_both_degenerate_case,
    devices=devices,
)
add_function_test(
    TestClosestPointEdgeEdgeMethods,
    "test_edge_edge_degenerate_first_edge",
    test_edge_edge_degenerate_first_edge,
    devices=devices,
)
add_function_test(
    TestClosestPointEdgeEdgeMethods,
    "test_edge_edge_degenerate_second_edge",
    test_edge_edge_degenerate_second_edge,
    devices=devices,
)
add_function_test(TestClosestPointEdgeEdgeMethods, "test_edge_edge_parallel", test_edge_edge_parallel, devices=devices)
add_function_test(
    TestClosestPointEdgeEdgeMethods,
    "test_edge_edge_perpendicular_s1_t0",
    test_edge_edge_perpendicular_s1_t0,
    devices=devices,
)
add_function_test(
    TestClosestPointEdgeEdgeMethods,
    "test_edge_edge_perpendicular_s0_t1",
    test_edge_edge_perpendicular_s0_t1,
    devices=devices,
)
add_function_test(
    TestClosestPointEdgeEdgeMethods,
    "test_edge_closest_point_random",
    check_edge_closest_point_random,
    devices=devices,
)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
