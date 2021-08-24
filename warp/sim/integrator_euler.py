"""This module contains time-integration objects for simulating
models + state forward in time.

"""

import math
from warp.context import kernel
from warp.sim.model import JOINT_PRISMATIC
from warp.utils import quat_identity
from warp.types import spatial_transform, vec3
import numpy as np
import time

import warp as wp

from . optimizer import Optimizer


@wp.kernel
def integrate_particles(x: wp.array(dtype=wp.vec3),
                        v: wp.array(dtype=wp.vec3),
                        f_ext: wp.array(dtype=wp.vec3),
                        f_int: wp.array(dtype=wp.vec3),
                        w: wp.array(dtype=float),
                        gravity: wp.vec3,
                        dt: float,
                        x_new: wp.array(dtype=wp.vec3),
                        v_new: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]

    fe = f_ext[tid]
    fi = f_int[tid]

    inv_mass = w[tid]

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v1 = v0 + ((fe + fi) * inv_mass + gravity * wp.step(0.0 - inv_mass)) * dt
    x1 = x0 + v1 * dt

    wp.store(x_new, tid, x1)
    wp.store(v_new, tid, v1)


# semi-implicit Euler integration
@wp.kernel
def integrate_bodies(body_q: wp.array(dtype=wp.spatial_transform),
                     body_qd: wp.array(dtype=wp.spatial_vector),
                     body_f_int: wp.array(dtype=wp.spatial_vector),
                     body_f_ext: wp.array(dtype=wp.spatial_vector),
                     body_com: wp.array(dtype=wp.vec3),
                     m: wp.array(dtype=float),
                     I: wp.array(dtype=wp.mat33),
                     inv_m: wp.array(dtype=float),
                     inv_I: wp.array(dtype=wp.mat33),
                     gravity: wp.vec3,
                     dt: float,
                     body_q_new: wp.array(dtype=wp.spatial_transform),
                     body_qd_new: wp.array(dtype=wp.spatial_vector)):

    tid = wp.tid()

    # positions
    q = body_q[tid]
    qd = body_qd[tid]
    f_int = body_f_int[tid]
    f_ext = body_f_ext[tid]

    # masses
    mass = m[tid]
    inv_mass = inv_m[tid]     # 1 / mass

    inertia = I[tid]
    inv_inertia = inv_I[tid]  # inverse of 3x3 inertia matrix

    # unpack transform
    x0 = wp.spatial_transform_get_translation(q)
    r0 = wp.spatial_transform_get_rotation(q)

    # unpack spatial twist
    w0 = wp.spatial_top(qd)
    v0 = wp.spatial_bottom(qd)

    # unpack spatial wrench
    f = f_int + f_ext
    t0 = wp.spatial_top(f)
    f0 = wp.spatial_bottom(f)

    x_com = x0 + wp.rotate(r0, body_com[tid])
 
    # linear part
    v1 = v0 + (f0 * inv_mass + gravity * wp.nonzero(inv_mass)) * dt
    x1 = x_com + v1 * dt
 
    # angular part (compute in body frame)
    wb = wp.rotate_inv(r0, w0)
    tb = wp.rotate_inv(r0, t0) - wp.cross(wb, inertia*wb)   # coriolis forces

    w1 = wp.rotate(r0, wb + inv_inertia * tb * dt)
    r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)

    # angular damping, todo: expose
    w1 = w1*(1.0-0.1*dt)

    body_q_new[tid] = wp.spatial_transform(x1 - wp.rotate(r1, body_com[tid]), r1)
    body_qd_new[tid] = wp.spatial_vector(w1, v1)


@wp.kernel
def eval_springs(x: wp.array(dtype=wp.vec3),
                 v: wp.array(dtype=wp.vec3),
                 spring_indices: wp.array(dtype=int),
                 spring_rest_lengths: wp.array(dtype=float),
                 spring_stiffness: wp.array(dtype=float),
                 spring_damping: wp.array(dtype=float),
                 f: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    i = wp.load(spring_indices, tid * 2 + 0)
    j = wp.load(spring_indices, tid * 2 + 1)

    ke = wp.load(spring_stiffness, tid)
    kd = wp.load(spring_damping, tid)
    rest = wp.load(spring_rest_lengths, tid)

    xi = wp.load(x, i)
    xj = wp.load(x, j)

    vi = wp.load(v, i)
    vj = wp.load(v, j)

    xij = xi - xj
    vij = vi - vj

    l = length(xij)
    l_inv = 1.0 / l

    # normalized spring direction
    dir = xij * l_inv

    c = l - rest
    dcdt = dot(dir, vij)

    # damping based on relative velocity.
    fs = dir * (ke * c + kd * dcdt)

    wp.atomic_sub(f, i, fs)
    wp.atomic_add(f, j, fs)


@wp.kernel
def eval_triangles(x: wp.array(dtype=wp.vec3),
                   v: wp.array(dtype=wp.vec3),
                   indices: wp.array(dtype=int),
                   pose: wp.array(dtype=wp.mat22),
                   activation: wp.array(dtype=float),
                   k_mu: float,
                   k_lambda: float,
                   k_damp: float,
                   k_drag: float,
                   k_lift: float,
                   f: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    i = wp.load(indices, tid * 3 + 0)
    j = wp.load(indices, tid * 3 + 1)
    k = wp.load(indices, tid * 3 + 2)

    x0 = wp.load(x, i)        # point zero
    x1 = wp.load(x, j)        # point one
    x2 = wp.load(x, k)        # point two

    v0 = wp.load(v, i)       # vel zero
    v1 = wp.load(v, j)       # vel one
    v2 = wp.load(v, k)       # vel two

    x10 = x1 - x0     # barycentric coordinates (centered at p)
    x20 = x2 - x0

    v10 = v1 - v0
    v20 = v2 - v0

    Dm = wp.load(pose, tid)

    inv_rest_area = wp.determinant(Dm) * 2.0     # 1 / det(A) = det(A^-1)
    rest_area = 1.0 / inv_rest_area

    # scale stiffness coefficients to account for area
    k_mu = k_mu * rest_area
    k_lambda = k_lambda * rest_area
    k_damp = k_damp * rest_area

    # F = Xs*Xm^-1
    F1 = x10 * Dm[0, 0] + x20 * Dm[1, 0]
    F2 = x10 * Dm[0, 1] + x20 * Dm[1, 1]

    # dFdt = Vs*Xm^-1
    dFdt1 = v10*Dm[0, 0] + v20*Dm[1, 0]
    dFdt2 = v10*Dm[0 ,1] + v20*Dm[1, 1]

    # deviatoric PK1 + damping term
    P1 = F1*k_mu + dFdt1*k_damp
    P2 = F2*k_mu + dFdt2*k_damp

    #-----------------------------
    # St. Venant-Kirchoff

    # # Green strain, F'*F-I
    # e00 = dot(f1, f1) - 1.0
    # e10 = dot(f2, f1)
    # e01 = dot(f1, f2)
    # e11 = dot(f2, f2) - 1.0

    # E = wp.mat22(e00, e01,
    #              e10, e11)

    # # local forces (deviatoric part)
    # T = wp.mul(E, wp.transpose(Dm))

    # # spatial forces, F*T
    # fq = (f1*T[0,0] + f2*T[1,0])*k_mu*2.0
    # fr = (f1*T[0,1] + f2*T[1,1])*k_mu*2.0
    # alpha = 1.0

    #-----------------------------
    # Baraff & Witkin, note this model is not isotropic

    # c1 = length(f1) - 1.0
    # c2 = length(f2) - 1.0
    # f1 = normalize(f1)*c1*k1
    # f2 = normalize(f2)*c2*k1

    # fq = f1*Dm[0,0] + f2*Dm[0,1]
    # fr = f1*Dm[1,0] + f2*Dm[1,1]

    #-----------------------------
    # Neo-Hookean (with rest stability)

    # force = P*Dm'
    f1 = (P1 * Dm[0, 0] + P2 * Dm[0, 1])
    f2 = (P1 * Dm[1, 0] + P2 * Dm[1, 1])
    alpha = 1.0 + k_mu / k_lambda

    #-----------------------------
    # Area Preservation

    n = wp.cross(x10, x20)
    area = wp.length(n) * 0.5

    # actuation
    act = wp.load(activation, tid)

    # J-alpha
    c = area * inv_rest_area - alpha + act

    # dJdx
    n = wp.normalize(n)
    dcdq = wp.cross(x20, n) * inv_rest_area * 0.5
    dcdr = wp.cross(n, x10) * inv_rest_area * 0.5

    f_area = k_lambda * c

    #-----------------------------
    # Area Damping

    dcdt = dot(dcdq, v1) + dot(dcdr, v2) - dot(dcdq + dcdr, v0)
    f_damp = k_damp * dcdt

    f1 = f1 + dcdq * (f_area + f_damp)
    f2 = f2 + dcdr * (f_area + f_damp)
    f0 = f1 + f2

    #-----------------------------
    # Lift + Drag

    vmid = (v0 + v1 + v2) * 0.3333
    vdir = wp.normalize(vmid)

    f_drag = vmid * (k_drag * area * wp.abs(wp.dot(n, vmid)))
    f_lift = n * (k_lift * area * (1.57079 - wp.acos(wp.dot(n, vdir)))) * dot(vmid, vmid)

    # note reversed sign due to atomic_add below.. need to write the unary op -
    f0 = f0 - f_drag - f_lift
    f1 = f1 + f_drag + f_lift
    f2 = f2 + f_drag + f_lift

    # apply forces
    wp.atomic_add(f, i, f0)
    wp.atomic_sub(f, j, f1)
    wp.atomic_sub(f, k, f2)


@wp.func
def triangle_closest_point_barycentric(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)

    if (d1 <= 0.0 and d2 <= 0.0):
        return vec3(1.0, 0.0, 0.0)

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)

    if (d3 >= 0.0 and d4 <= d3):
        return vec3(0.0, 1.0, 0.0)

    vc = d1 * d4 - d3 * d2
    v = d1 / (d1 - d3)
    if (vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0):
        return vec3(1.0 - v, v, 0.0)

    cp = p - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)

    if (d6 >= 0.0 and d5 <= d6):
        return vec3(0.0, 0.0, 1.0)

    vb = d5 * d2 - d1 * d6
    w = d2 / (d2 - d6)
    if (vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0):
        return vec3(1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    if (va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0):
        return vec3(0.0, w, 1.0 - w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom

    return vec3(1.0 - v - w, v, w)

# @wp.func
# def triangle_closest_point(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
#     ab = b - a
#     ac = c - a
#     ap = p - a

#     d1 = wp.dot(ab, ap)
#     d2 = wp.dot(ac, ap)

#     if (d1 <= 0.0 and d2 <= 0.0):
#         return a

#     bp = p - b
#     d3 = wp.dot(ab, bp)
#     d4 = wp.dot(ac, bp)

#     if (d3 >= 0.0 and d4 <= d3):
#         return b

#     vc = d1 * d4 - d3 * d2
#     v = d1 / (d1 - d3)
#     if (vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0):
#         return a + ab * v

#     cp = p - c
#     d5 = dot(ab, cp)
#     d6 = dot(ac, cp)

#     if (d6 >= 0.0 and d5 <= d6):
#         return c

#     vb = d5 * d2 - d1 * d6
#     w = d2 / (d2 - d6)
#     if (vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0):
#         return a + ac * w

#     va = d3 * d6 - d5 * d4
#     w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
#     if (va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0):
#         return b + (c - b) * w

#     denom = 1.0 / (va + vb + vc)
#     v = vb * denom
#     w = vc * denom

#     return a + ab * v + ac * w


@wp.kernel
def eval_triangles_contact(
                                       # idx : wp.array(dtype=int), # list of indices for colliding particles
    num_particles: int,                # size of particles
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    pose: wp.array(dtype=wp.mat22),
    activation: wp.array(dtype=float),
    k_mu: float,
    k_lambda: float,
    k_damp: float,
    k_drag: float,
    k_lift: float,
    f: wp.array(dtype=wp.vec3)):

    tid = wp.tid()
    face_no = tid // num_particles     # which face
    particle_no = tid % num_particles  # which particle

    # index = wp.load(idx, tid)
    pos = wp.load(x, particle_no)      # at the moment, just one particle
                                       # vel0 = wp.load(v, 0)

    i = wp.load(indices, face_no * 3 + 0)
    j = wp.load(indices, face_no * 3 + 1)
    k = wp.load(indices, face_no * 3 + 2)

    if (i == particle_no or j == particle_no or k == particle_no):
        return

    p = wp.load(x, i)        # point zero
    q = wp.load(x, j)        # point one
    r = wp.load(x, k)        # point two

    # vp = wp.load(v, i) # vel zero
    # vq = wp.load(v, j) # vel one
    # vr = wp.load(v, k)  # vel two

    # qp = q-p # barycentric coordinates (centered at p)
    # rp = r-p

    bary = triangle_closest_point_barycentric(p, q, r, pos)
    closest = p * bary[0] + q * bary[1] + r * bary[2]

    diff = pos - closest
    dist = wp.dot(diff, diff)
    n = wp.normalize(diff)
    c = wp.min(dist - 0.01, 0.0)       # 0 unless within 0.01 of surface
                                       #c = wp.leaky_min(dot(n, x0)-0.01, 0.0, 0.0)
    fn = n * c * 1e5

    wp.atomic_sub(f, particle_no, fn)

    # # apply forces (could do - f / 3 here)
    wp.atomic_add(f, i, fn * bary[0])
    wp.atomic_add(f, j, fn * bary[1])
    wp.atomic_add(f, k, fn * bary[2])


@wp.kernel
def eval_triangles_body_contacts(
    num_particles: int,                          # number of particles (size of contact_point)
    x: wp.array(dtype=wp.vec3),                     # position of particles
    v: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),                     # triangle indices
    body_x: wp.array(dtype=wp.vec3),               # body body positions
    body_r: wp.array(dtype=wp.quat),
    body_v: wp.array(dtype=wp.vec3),
    body_w: wp.array(dtype=wp.vec3),
    contact_body: wp.array(dtype=int),
    contact_point: wp.array(dtype=wp.vec3),         # position of contact points relative to body
    contact_dist: wp.array(dtype=float),
    contact_mat: wp.array(dtype=int),
    materials: wp.array(dtype=float),
                                                 #   body_f : wp.array(dtype=wp.vec3),
                                                 #   body_t : wp.array(dtype=wp.vec3),
    tri_f: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    face_no = tid // num_particles     # which face
    particle_no = tid % num_particles  # which particle

    # -----------------------
    # load body body point
    c_body = wp.load(contact_body, particle_no)
    c_point = wp.load(contact_point, particle_no)
    c_dist = wp.load(contact_dist, particle_no)
    c_mat = wp.load(contact_mat, particle_no)

    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    ke = wp.load(materials, c_mat * 4 + 0)       # restitution coefficient
    kd = wp.load(materials, c_mat * 4 + 1)       # damping coefficient
    kf = wp.load(materials, c_mat * 4 + 2)       # friction coefficient
    mu = wp.load(materials, c_mat * 4 + 3)       # coulomb friction

    x0 = wp.load(body_x, c_body)      # position of colliding body
    r0 = wp.load(body_r, c_body)      # orientation of colliding body

    v0 = wp.load(body_v, c_body)
    w0 = wp.load(body_w, c_body)

    # transform point to world space
    pos = x0 + wp.rotate(r0, c_point)
    # use x0 as center, everything is offset from center of mass

    # moment arm
    r = pos - x0                       # basically just c_point in the new coordinates
    rhat = wp.normalize(r)
    pos = pos + rhat * c_dist          # add on 'thickness' of shape, e.g.: radius of sphere/capsule

    # contact point velocity
    dpdt = v0 + wp.cross(w0, r)        # this is body velocity cross offset, so it's the velocity of the contact point.

    # -----------------------
    # load triangle
    i = wp.load(indices, face_no * 3 + 0)
    j = wp.load(indices, face_no * 3 + 1)
    k = wp.load(indices, face_no * 3 + 2)

    p = wp.load(x, i)        # point zero
    q = wp.load(x, j)        # point one
    r = wp.load(x, k)        # point two

    vp = wp.load(v, i)       # vel zero
    vq = wp.load(v, j)       # vel one
    vr = wp.load(v, k)       # vel two

    bary = triangle_closest_point_barycentric(p, q, r, pos)
    closest = p * bary[0] + q * bary[1] + r * bary[2]

    diff = pos - closest               # vector from tri to point
    dist = wp.dot(diff, diff)          # squared distance
    n = wp.normalize(diff)             # points into the object
    c = wp.min(dist - 0.05, 0.0)       # 0 unless within 0.05 of surface
                                       #c = wp.leaky_min(dot(n, x0)-0.01, 0.0, 0.0)
                                       # fn = n * c * 1e6    # points towards cloth (both n and c are negative)

    # wp.atomic_sub(tri_f, particle_no, fn)

    fn = c * ke    # normal force (restitution coefficient * how far inside for ground) (negative)

    vtri = vp * bary[0] + vq * bary[1] + vr * bary[2]         # bad approximation for centroid velocity
    vrel = vtri - dpdt

    vn = dot(n, vrel)        # velocity component of body in negative normal direction
    vt = vrel - n * vn       # velocity component not in normal direction

    # contact damping
    fd = 0.0 - wp.max(vn, 0.0) * kd * wp.step(c)           # again, negative, into the ground

    # # viscous friction
    # ft = vt*kf

    # Coulomb friction (box)
    lower = mu * (fn + fd)
    upper = 0.0 - lower      # workaround because no unary ops yet

    nx = cross(n, vec3(0.0, 0.0, 1.0))         # basis vectors for tangent
    nz = cross(n, vec3(1.0, 0.0, 0.0))

    vx = wp.clamp(dot(nx * kf, vt), lower, upper)
    vz = wp.clamp(dot(nz * kf, vt), lower, upper)

    ft = (nx * vx + nz * vz) * (0.0 - wp.step(c))          # wp.vec3(vx, 0.0, vz)*wp.step(c)

    # # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    # #ft = wp.normalize(vt)*wp.min(kf*wp.length(vt), 0.0 - mu*c*ke)

    f_total = n * (fn + fd) + ft

    wp.atomic_add(tri_f, i, f_total * bary[0])
    wp.atomic_add(tri_f, j, f_total * bary[1])
    wp.atomic_add(tri_f, k, f_total * bary[2])


@wp.kernel
def eval_bending(
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), indices: wp.array(dtype=int), rest: wp.array(dtype=float), ke: float, kd: float, f: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    i = wp.load(indices, tid * 4 + 0)
    j = wp.load(indices, tid * 4 + 1)
    k = wp.load(indices, tid * 4 + 2)
    l = wp.load(indices, tid * 4 + 3)

    rest_angle = wp.load(rest, tid)

    x1 = wp.load(x, i)
    x2 = wp.load(x, j)
    x3 = wp.load(x, k)
    x4 = wp.load(x, l)

    v1 = wp.load(v, i)
    v2 = wp.load(v, j)
    v3 = wp.load(v, k)
    v4 = wp.load(v, l)

    n1 = wp.cross(x3 - x1, x4 - x1)    # normal to face 1
    n2 = wp.cross(x4 - x2, x3 - x2)    # normal to face 2

    n1_length = wp.length(n1)
    n2_length = wp.length(n2)

    if (n1_length < 1.e-6 or n2_length < 1.e-6):
        return

    rcp_n1 = 1.0 / n1_length
    rcp_n2 = 1.0 / n2_length

    cos_theta = wp.dot(n1, n2) * rcp_n1 * rcp_n2

    n1 = n1 * rcp_n1 * rcp_n1
    n2 = n2 * rcp_n2 * rcp_n2

    e = x4 - x3
    e_hat = wp.normalize(e)
    e_length = wp.length(e)

    s = wp.sign(wp.dot(wp.cross(n2, n1), e_hat))
    angle = wp.acos(cos_theta) * s

    d1 = n1 * e_length
    d2 = n2 * e_length
    d3 = n1 * wp.dot(x1 - x4, e_hat) + n2 * wp.dot(x2 - x4, e_hat)
    d4 = n1 * wp.dot(x3 - x1, e_hat) + n2 * wp.dot(x3 - x2, e_hat)

    # elastic
    f_elastic = ke * (angle - rest_angle)

    # damping
    f_damp = kd * (wp.dot(d1, v1) + wp.dot(d2, v2) + wp.dot(d3, v3) + wp.dot(d4, v4))

    # total force, proportional to edge length
    f_total = 0.0 - e_length * (f_elastic + f_damp)

    wp.atomic_add(f, i, d1 * f_total)
    wp.atomic_add(f, j, d2 * f_total)
    wp.atomic_add(f, k, d3 * f_total)
    wp.atomic_add(f, l, d4 * f_total)


@wp.kernel
def eval_tetrahedra(x: wp.array(dtype=wp.vec3),
                    v: wp.array(dtype=wp.vec3),
                    indices: wp.array(dtype=int),
                    pose: wp.array(dtype=wp.mat33),
                    activation: wp.array(dtype=float),
                    materials: wp.array(dtype=float),
                    f: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    i = wp.load(indices, tid * 4 + 0)
    j = wp.load(indices, tid * 4 + 1)
    k = wp.load(indices, tid * 4 + 2)
    l = wp.load(indices, tid * 4 + 3)

    act = wp.load(activation, tid)

    k_mu = wp.load(materials, tid * 3 + 0)
    k_lambda = wp.load(materials, tid * 3 + 1)
    k_damp = wp.load(materials, tid * 3 + 2)

    x0 = wp.load(x, i)
    x1 = wp.load(x, j)
    x2 = wp.load(x, k)
    x3 = wp.load(x, l)

    v0 = wp.load(v, i)
    v1 = wp.load(v, j)
    v2 = wp.load(v, k)
    v3 = wp.load(v, l)

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    v10 = v1 - v0
    v20 = v2 - v0
    v30 = v3 - v0

    Ds = wp.mat33(x10, x20, x30)
    Dm = wp.load(pose, tid)

    inv_rest_volume = wp.determinant(Dm) * 6.0
    rest_volume = 1.0 / inv_rest_volume

    alpha = 1.0 + k_mu / k_lambda - k_mu / (4.0 * k_lambda)

    # scale stiffness coefficients to account for area
    k_mu = k_mu * rest_volume
    k_lambda = k_lambda * rest_volume
    k_damp = k_damp * rest_volume

    # F = Xs*Xm^-1
    F = Ds * Dm
    dFdt = wp.mat33(v10, v20, v30) * Dm

    col1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    col2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    col3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    #-----------------------------
    # Neo-Hookean (with rest stability [Smith et al 2018])
         
    Ic = dot(col1, col1) + dot(col2, col2) + dot(col3, col3)

    # deviatoric part
    P = F * k_mu * (1.0 - 1.0 / (Ic + 1.0)) + dFdt * k_damp
    H = P * wp.transpose(Dm)

    f1 = wp.vec3(H[0, 0], H[1, 0], H[2, 0])
    f2 = wp.vec3(H[0, 1], H[1, 1], H[2, 1])
    f3 = wp.vec3(H[0, 2], H[1, 2], H[2, 2])

    #-----------------------------
    # C_sqrt

    # alpha = 1.0
       
    # r_s = wp.sqrt(wp.abs(dot(col1, col1) + dot(col2, col2) + dot(col3, col3) - 3.0))

    # f1 = wp.vec3(0.0, 0.0, 0.0)
    # f2 = wp.vec3(0.0, 0.0, 0.0)
    # f3 = wp.vec3(0.0, 0.0, 0.0)

    # if (r_s > 0.0):
    #     r_s_inv = 1.0/r_s

    #     C = r_s 
    #     dCdx = F*wp.transpose(Dm)*r_s_inv*wp.sign(r_s)

    #     grad1 = vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    #     grad2 = vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    #     grad3 = vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
    
    #     f1 = grad1*C*k_mu
    #     f2 = grad2*C*k_mu
    #     f3 = grad3*C*k_mu

    #-----------------------------
    # C_spherical
    
    # alpha = 1.0

    # r_s = wp.sqrt(dot(col1, col1) + dot(col2, col2) + dot(col3, col3))
    # r_s_inv = 1.0/r_s

    # C = r_s - wp.sqrt(3.0) 
    # dCdx = F*wp.transpose(Dm)*r_s_inv

    # grad1 = vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    # grad2 = vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    # grad3 = vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
 

    # f1 = grad1*C*k_mu
    # f2 = grad2*C*k_mu
    # f3 = grad3*C*k_mu

    #----------------------------
    # C_D

    # alpha = 1.0

    # r_s = wp.sqrt(dot(col1, col1) + dot(col2, col2) + dot(col3, col3))

    # C = r_s*r_s - 3.0
    # dCdx = F*wp.transpose(Dm)*2.0

    # grad1 = vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    # grad2 = vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    # grad3 = vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
 
    # f1 = grad1*C*k_mu
    # f2 = grad2*C*k_mu
    # f3 = grad3*C*k_mu

    #----------------------------
    # Hookean
     
    # alpha = 1.0

    # I = wp.mat33(wp.vec3(1.0, 0.0, 0.0),
    #              wp.vec3(0.0, 1.0, 0.0),
    #              wp.vec3(0.0, 0.0, 1.0))

    # P = (F + wp.transpose(F) + I*(0.0-2.0))*k_mu
    # H = P * wp.transpose(Dm)

    # f1 = wp.vec3(H[0, 0], H[1, 0], H[2, 0])
    # f2 = wp.vec3(H[0, 1], H[1, 1], H[2, 1])
    # f3 = wp.vec3(H[0, 2], H[1, 2], H[2, 2])



    # hydrostatic part
    J = wp.determinant(F)

    #print(J)
    s = inv_rest_volume / 6.0
    dJdx1 = wp.cross(x20, x30) * s
    dJdx2 = wp.cross(x30, x10) * s
    dJdx3 = wp.cross(x10, x20) * s

    f_volume = (J - alpha + act) * k_lambda
    f_damp = (wp.dot(dJdx1, v1) + wp.dot(dJdx2, v2) + wp.dot(dJdx3, v3)) * k_damp

    f_total = f_volume + f_damp

    f1 = f1 + dJdx1 * f_total
    f2 = f2 + dJdx2 * f_total
    f3 = f3 + dJdx3 * f_total
    f0 = (f1 + f2 + f3) * (0.0 - 1.0)

    # apply forces
    wp.atomic_sub(f, i, f0)
    wp.atomic_sub(f, j, f1)
    wp.atomic_sub(f, k, f2)
    wp.atomic_sub(f, l, f3)


@wp.kernel
def eval_contacts(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), ke: float, kd: float, kf: float, mu: float, offset: float, ground: wp.vec4, f: wp.array(dtype=wp.vec3)):

    tid = wp.tid()           

    x0 = wp.load(x, tid)
    v0 = wp.load(v, tid)

    n = wp.vec3(ground[0], ground[1], ground[2])
    c = wp.min(dot(n, x0) + ground[3] - offset, 0.0)

    vn = dot(n, v0)
    vt = v0 - n * vn

    fn = n * c * ke

    # contact damping
    fd = n * wp.min(vn, 0.0) * kd

    # viscous friction
    #ft = vt*kf

    # Coulomb friction (box)
    lower = mu * c * ke
    upper = 0.0 - lower

    vx = clamp(dot(vec3(kf, 0.0, 0.0), vt), lower, upper)
    vz = clamp(dot(vec3(0.0, 0.0, kf), vt), lower, upper)

    ft = wp.vec3(vx, 0.0, vz)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    #ft = wp.normalize(vt)*wp.min(kf*wp.length(vt), 0.0 - mu*c*ke)

    ftotal = fn + (fd + ft) * wp.step(c)

    wp.atomic_sub(f, tid, ftotal)



@wp.kernel
def eval_soft_contacts(
    particle_x: wp.array(dtype=wp.vec3), 
    particle_v: wp.array(dtype=wp.vec3), 
    body_q: wp.array(dtype=wp.spatial_transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    ke: float,
    kd: float, 
    kf: float, 
    mu: float,
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_body: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_distance: float,
    # outputs
    particle_f: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector)):

    tid = wp.tid()

    count = contact_count[0]
    if (tid >= count):
        return
        
    body_index = contact_body[tid]
    particle_index = contact_particle[tid]

    px = wp.load(particle_x, particle_index)
    pv = wp.load(particle_v, particle_index)

    X_wb = wp.spatial_transform_identity()
    if (body_index >= 0):
        X_wb = wp.load(body_q, body_index)

    # body position in world space
    bx = wp.spatial_transform_point(X_wb, contact_body_pos[tid])
    r = bx - wp.spatial_transform_point(X_wb, body_com[body_index])
    
    n = contact_normal[tid]
    c = wp.dot(n, px-bx) - contact_distance
    
    if (c > 0.0):
        return

    # body velocity
    body_v_s = wp.spatial_vector()
    if (body_index >= 0):
        body_v_s = wp.load(body_qd, body_index)
    
    body_w = wp.spatial_top(body_v_s)
    body_v = wp.spatial_bottom(body_v_s)

    # compute the body velocity at the particle position
    bv = body_v + wp.cross(body_w, r) + wp.spatial_transform_vector(X_wb, contact_body_vel[tid])

    # relative velocity
    v = pv - bv

    # decompose relative velocity
    vn = wp.dot(n, v)
    vt = v - n * vn
    
    # contact elastic
    fn = n * c * ke

    # contact damping
    fd = n * wp.min(vn, 0.0) * kd

    # viscous friction
    #ft = vt*kf

    # Coulomb friction (box)
    # lower = mu * c * ke
    # upper = 0.0 - lower

    # vx = clamp(dot(vec3(kf, 0.0, 0.0), vt), lower, upper)
    # vz = clamp(dot(vec3(0.0, 0.0, kf), vt), lower, upper)

    #ft = wp.vec3(vx, 0.0, vz)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    ft = wp.normalize(vt)*wp.min(kf*wp.length(vt), 0.0 - mu*c*ke)

    f_total = fn + (fd + ft) * wp.step(c)
    t_total = wp.cross(r, f_total)

    wp.atomic_sub(particle_f, particle_index, f_total)

    if (body_index >= 0):
        wp.atomic_add(body_f, body_index, wp.spatial_vector(t_total, f_total))



@wp.kernel
def eval_body_contacts(body_q: wp.array(dtype=wp.spatial_transform),
                       body_qd: wp.array(dtype=wp.spatial_vector),
                       body_com: wp.array(dtype=wp.vec3),
                       contact_body: wp.array(dtype=int),
                       contact_point: wp.array(dtype=wp.vec3),
                       contact_dist: wp.array(dtype=float),
                       contact_mat: wp.array(dtype=int),
                       materials: wp.array(dtype=wp.vec4),
                       body_f: wp.array(dtype=wp.spatial_vector)):

    tid = wp.tid()

    c_body = contact_body[tid]
    c_point = contact_point[tid]
    c_dist = contact_dist[tid]
    c_mat = contact_mat[tid]

    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    mat = materials[c_mat]

    ke = mat[0]       # restitution coefficient
    kd = mat[1]       # damping coefficient
    kf = mat[2]       # friction coefficient
    mu = mat[3]       # coulomb friction

    X_wb = body_q[c_body]
    v_wc = body_qd[c_body]

    # unpack spatial twist
    w = wp.spatial_top(v_wc)
    v = wp.spatial_bottom(v_wc)

    n = vec3(0.0, 1.0, 0.0)

    # transform point to world space
    cp = wp.spatial_transform_point(X_wb, c_point) - n * c_dist # add on 'thickness' of shape, e.g.: radius of sphere/capsule

    # moment arm around center of mass
    r = cp - wp.spatial_transform_point(X_wb, body_com[c_body])

    # contact point velocity
    dpdt = v + wp.cross(w, r)     

    # check ground contact
    c = wp.min(wp.dot(n, cp), 0.0) 

    vn = wp.dot(n, dpdt)     
    vt = dpdt - n * vn       

    # normal force
    fn = c * ke    

    # damping force
    fd = wp.min(vn, 0.0) * kd * wp.step(c)       # again, velocity into the ground, negative

    # viscous friction
    #ft = vt*kf

    # Coulomb friction (box)
    lower = mu * (fn + fd)   # negative
    upper = 0.0 - lower      # positive, workaround for no unary ops

    vx = wp.clamp(wp.dot(vec3(kf, 0.0, 0.0), vt), lower, upper)
    vz = wp.clamp(wp.dot(vec3(0.0, 0.0, kf), vt), lower, upper)

    ft = wp.vec3(vx, 0.0, vz) * wp.step(c)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    #ft = wp.normalize(vt)*wp.min(kf*wp.length(vt), 0.0 - mu*c*ke)

    f_total = n * (fn + fd) + ft
    t_total = wp.cross(r, f_total)

    wp.atomic_sub(body_f, c_body, wp.spatial_vector(t_total, f_total))

# # Frank & Park definition 3.20, pg 100
@wp.func
def spatial_transform_twist(t: wp.spatial_transform, x: wp.spatial_vector):

    q = spatial_transform_get_rotation(t)
    p = spatial_transform_get_translation(t)

    w = spatial_top(x)
    v = spatial_bottom(x)

    w = rotate(q, w)
    v = rotate(q, v) + cross(p, w)

    return wp.spatial_vector(w, v)


@wp.func
def spatial_transform_wrench(t: wp.spatial_transform, x: wp.spatial_vector):

    q = spatial_transform_get_rotation(t)
    p = spatial_transform_get_translation(t)

    w = spatial_top(x)
    v = spatial_bottom(x)

    v = rotate(q, v)
    w = rotate(q, w) + cross(p, v)

    return wp.spatial_vector(w, v)

@wp.func
def spatial_transform_inverse(t: wp.spatial_transform):

    p = spatial_transform_get_translation(t)
    q = spatial_transform_get_rotation(t)

    q_inv = quat_inverse(q)
    return spatial_transform(rotate(q_inv, p)*(0.0 - 1.0), q_inv)



# computes adj_t^-T*I*adj_t^-1 (tensor change of coordinates), Frank & Park, section 8.2.3, pg 290
@wp.func
def spatial_transform_inertia(t: wp.spatial_transform, I: wp.spatial_matrix):

    t_inv = spatial_transform_inverse(t)

    q = spatial_transform_get_rotation(t_inv)
    p = spatial_transform_get_translation(t_inv)

    r1 = rotate(q, vec3(1.0, 0.0, 0.0))
    r2 = rotate(q, vec3(0.0, 1.0, 0.0))
    r3 = rotate(q, vec3(0.0, 0.0, 1.0))

    R = mat33(r1, r2, r3)
    S = mul(skew(p), R)

    T = spatial_adjoint(R, S)
    
    return mul(mul(transpose(T), I), T)


# returns the twist around an axis
@wp.func
def quat_twist(axis: wp.vec3, q: wp.quat):
    
    # project imaginary part onto axis
    a = wp.vec3(q[0], q[1], q[2])
    a = wp.dot(a, axis)*axis

    return wp.normalize(wp.quat(a[0], a[1], a[2], q[3]))


@wp.kernel
def eval_body_joints(body_q: wp.array(dtype=wp.spatial_transform),
                     body_qd: wp.array(dtype=wp.spatial_vector),
                     body_com: wp.array(dtype=wp.vec3),
                     joint_type: wp.array(dtype=int),
                     joint_parent: wp.array(dtype=int),
                     joint_X_p: wp.array(dtype=wp.spatial_transform),
                     joint_X_c: wp.array(dtype=wp.spatial_transform),
                     joint_axis: wp.array(dtype=wp.vec3),
                     joint_target: wp.array(dtype=float),
                     joint_act: wp.array(dtype=float),
                     joint_target_ke: wp.array(dtype=float),
                     joint_target_kd: wp.array(dtype=float),
                     joint_limit_lower: wp.array(dtype=float),
                     joint_limit_upper: wp.array(dtype=float),
                     joint_limit_ke: wp.array(dtype=float),
                     joint_limit_kd: wp.array(dtype=float),
                     body_f: wp.array(dtype=wp.spatial_vector)):

    tid = wp.tid()

    c_child = tid
    c_parent = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wp = X_pj
    r_p = wp.vec3(0.0, 0.0, 0.0)
    w_p = wp.vec3(0.0, 0.0, 0.0)
    v_p = wp.vec3(0.0, 0.0, 0.0)

    # parent transform and moment arm
    if (c_parent >= 0):
        X_wp = body_q[c_parent]*X_wp
        r_wp = X_wp.p - wp.spatial_transform_point(body_com[c_parent])
        
        twist_p = body_qd[c_parent]

        w_p = wp.spatial_top(twist_p)
        v_p = wp.spatial_bottom(twist_p) + wp.cross(w_p, r_wp)


    # child transform and moment arm
    X_wc = body_q[c_child]*joint_X_c[tid]
    r_c = X_wc.p - wp.spatial_transform_point(body_com[c_child])
    
    twist_c = body_qd[c_child]

    w_c = wp.spatial_top(twist_c)
    v_c = wp.spatial_bottom(twist_c) + wp.cross(w_c, r_c)

    # joint properties
    axis = joint_axis[tid]
    target = joint_target[tid*3+0]                  # todo: 3-dof targets
    target_ke = joint_target_ke[tid]
    target_kd = joint_target_kd[tid]
    limit_ke = joint_limit_ke[tid]
    limit_kd = joint_limit_kd[tid]
    limit_lower = joint_limit_lower[tid*3+0]        # todo: 3-dof limits
    limit_upper = joint_limit_upper[tid*3+0]        # todo: 3-dof limits
    act = joint_act[tid]

    attach_ke = 1.e+3
    attach_kd = 1.e+2

    x_p = wp.spatial_transform_get_translation(X_wp)
    x_c = wp.spatial_transform_get_translation(X_wc)

    q_p = wp.spatial_transform_get_rotation(X_wp)
    q_c = wp.spatial_transform_get_rotation(X_wc)

    # translational error
    x_err = x_c - x_p
    v_err = v_c - v_p
    w_err = w_c - w_p

    # total force/torque on the parent
    t_total = wp.vec3(0.0, 0.0, 0.0)
    f_total = wp.vec3(0.0, 0.0, 0.0)

    # prismatic
    if (joint_type == 0):
        
        # world space joint axis
        axis_w = wp.spatial_transform_vector(X_wp, joint_axis[tid])

        # evaluate joint coordinates
        q = wp.dot(x_err, axis_w)
        qd = wp.dot(v_err, axis_w)

        # compute limit forces, damping only active when limit is violated
        if (q < limit_lower):
            limit_f = limit_ke*(limit_lower-q) - limit_kd*min(qd, 0.0)

        if (q > limit_upper):
            limit_f = limit_ke*(limit_upper-q) - limit_kd*max(qd, 0.0)

        # joint dynamics
        f_total += (target_ke*(q - target) - target_kd*qd + act + limit_f)*axis_w

        # attachment dynamics
        q_pc = wp.quat_multiply(wp.quat_inverse(q_p), q_c)

        ang_err = wp.normalize(wp.vec3(q_pc[0], q_pc[1], q_pc[2]))*wp.acos(q_pc[3])*2.0
    
        f_total += (x_err - q*axis_w)*attach_ke + (v_err - qd*axis_w)*attach_kd
        t_total += ang_err*attach_ke + w_err*attach_kd
    
    # revolute
    if (joint_type == 1):
        
        axis_w = wp.spatial_transform_vector(X_wp, joint_axis[tid])

        # swing twist decomposition
        q_pc = wp.quat_multiply(wp.quat_inverse(q_p), q_c)
        q_twist = quat_twist(axis, q_pc)

        q = wp.acos(q_twist[3])*2.0
        qd = wp.dot(w_err, axis_w)

        # joint dynamics
        t_total += (target_ke*(q - target) - target_kd*qd + act + limit_f)*axis_w
        
        # torque to remove swing
        swing = q_pc*wp.quat_inverse(quat_twist)
        swing_err = wp.normalize(wp.vec3(swing[0], swing[1], swing[2]))*wp.acos(swing[3])*2.0

        # attachment dynamics
        f_total += x_err*attach_ke + v_err*attach_kd
        t_total += swing_err*attach_ke + (w_err - qd*axis_w)*attach_kd 

    
    # ball
    if (joint_type == 2):
        pass
    
    # fixed
    if (joint_type == 3):
        pass
    
    # free
    if (joint_type == 4):
        pass

    # write forces
    if (c_parent >= 0):
        wp.atomic_add(body_f, c_parent, wp.spatial_vector(t_total + wp.cross(r_p, f_total), f_total))
        
    wp.atomic_sub(body_f, c_child, wp.spatial_vector(t_total + wp.cross(r_c, f_total), f_total))



@wp.func
def compute_muscle_force(
    i: int,
    body_X_s: wp.array(dtype=wp.spatial_transform),
    body_v_s: wp.array(dtype=wp.spatial_vector),    
    muscle_links: wp.array(dtype=int),
    muscle_points: wp.array(dtype=wp.vec3),
    muscle_activation: float,
    body_f_s: wp.array(dtype=wp.spatial_vector)):

    link_0 = wp.load(muscle_links, i)
    link_1 = wp.load(muscle_links, i+1)

    if (link_0 == link_1):
        return 0

    r_0 = wp.load(muscle_points, i)
    r_1 = wp.load(muscle_points, i+1)

    xform_0 = wp.load(body_X_s, link_0)
    xform_1 = wp.load(body_X_s, link_1)

    pos_0 = wp.spatial_transform_point(xform_0, r_0)
    pos_1 = wp.spatial_transform_point(xform_1, r_1)

    n = wp.normalize(pos_1 - pos_0)

    # todo: add passive elastic and viscosity terms
    f = n * muscle_activation

    wp.atomic_sub(body_f_s, link_0, wp.spatial_vector(wp.cross(pos_0, f), f))
    wp.atomic_add(body_f_s, link_1, wp.spatial_vector(wp.cross(pos_1, f), f))

    return 0


@wp.kernel
def eval_muscles(
    body_X_s: wp.array(dtype=wp.spatial_transform),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    muscle_start: wp.array(dtype=int),
    muscle_params: wp.array(dtype=float),
    muscle_links: wp.array(dtype=int),
    muscle_points: wp.array(dtype=wp.vec3),
    muscle_activation: wp.array(dtype=float),
    # output
    body_f_s: wp.array(dtype=wp.spatial_vector)):

    tid = wp.tid()

    m_start = wp.load(muscle_start, tid)
    m_end = wp.load(muscle_start, tid+1) - 1

    activation = wp.load(muscle_activation, tid)

    for i in range(m_start, m_end):
        compute_muscle_force(i, body_X_s, body_v_s, muscle_links, muscle_points, activation, body_f_s)
    


@wp.kernel
def eval_dense_gemm(m: int, n: int, p: int, t1: int, t2: int, A: wp.array(dtype=float), B: wp.array(dtype=float), C: wp.array(dtype=float)):
    dense_gemm(m, n, p, t1, t2, A, B, C)

@wp.kernel
def eval_dense_gemm_batched(m: wp.array(dtype=int), n: wp.array(dtype=int), p: wp.array(dtype=int), t1: int, t2: int, A_start: wp.array(dtype=int), B_start: wp.array(dtype=int), C_start: wp.array(dtype=int), A: wp.array(dtype=float), B: wp.array(dtype=float), C: wp.array(dtype=float)):
    dense_gemm_batched(m, n, p, t1, t2, A_start, B_start, C_start, A, B, C)

@wp.kernel
def eval_dense_cholesky(n: int, A: wp.array(dtype=float), regularization: float, L: wp.array(dtype=float)):
    dense_chol(n, A, regularization, L)

@wp.kernel
def eval_dense_cholesky_batched(A_start: wp.array(dtype=int), A_dim: wp.array(dtype=int), A: wp.array(dtype=float), regularization: float, L: wp.array(dtype=float)):
    dense_chol_batched(A_start, A_dim, A, regularization, L)

@wp.kernel
def eval_dense_subs(n: int, L: wp.array(dtype=float), b: wp.array(dtype=float), x: wp.array(dtype=float)):
    dense_subs(n, L, b, x)

# helper that propagates gradients back to A, treating L as a constant / temporary variable
# allows us to reuse the Cholesky decomposition from the forward pass
@wp.kernel
def eval_dense_solve(n: int, A: wp.array(dtype=float), L: wp.array(dtype=float), b: wp.array(dtype=float), x: wp.array(dtype=float)):
    dense_solve(n, A, L, b, x)

# helper that propagates gradients back to A, treating L as a constant / temporary variable
# allows us to reuse the Cholesky decomposition from the forward pass
@wp.kernel
def eval_dense_solve_batched(b_start: wp.array(dtype=int), A_start: wp.array(dtype=int), A_dim: wp.array(dtype=int), A: wp.array(dtype=float), L: wp.array(dtype=float), b: wp.array(dtype=float), x: wp.array(dtype=float)):
    dense_solve_batched(b_start, A_start, A_dim, A, L, b, x)



def compute_forces(model, state, particle_f, body_f):

    # damped springs
    if (model.spring_count):

        wp.launch(kernel=eval_springs,
                    dim=model.spring_count,
                    inputs=[
                        state.particle_q, 
                        state.particle_qd, 
                        model.spring_indices, 
                        model.spring_rest_length, 
                        model.spring_stiffness, 
                        model.spring_damping
                    ],
                    outputs=[particle_f],
                    device=model.device)

    # triangle elastic and lift/drag forces
    if (model.tri_count and model.tri_ke > 0.0):

        wp.launch(kernel=eval_triangles,
                    dim=model.tri_count,
                    inputs=[
                        state.particle_q,
                        state.particle_qd,
                        model.tri_indices,
                        model.tri_poses,
                        model.tri_activations,
                        model.tri_ke,
                        model.tri_ka,
                        model.tri_kd,
                        model.tri_drag,
                        model.tri_lift
                    ],
                    outputs=[particle_f],
                    device=model.device)

    # triangle/triangle contacts
    if (model.enable_tri_collisions and model.tri_count and model.tri_ke > 0.0):

        wp.launch(kernel=eval_triangles_contact,
                    dim=model.tri_count * model.particle_count,
                    inputs=[
                        model.particle_count,
                        state.particle_q,
                        state.particle_qd,
                        model.tri_indices,
                        model.tri_poses,
                        model.tri_activations,
                        model.tri_ke,
                        model.tri_ka,
                        model.tri_kd,
                        model.tri_drag,
                        model.tri_lift
                    ],
                    outputs=[particle_f],
                    device=model.device)

    # triangle bending
    if (model.edge_count):

        wp.launch(kernel=eval_bending,
                    dim=model.edge_count,
                    inputs=[state.particle_q, state.particle_qd, model.edge_indices, model.edge_rest_angle, model.edge_ke, model.edge_kd],
                    outputs=[particle_f],
                    device=model.device)

    # particle ground contact
    if (model.ground and model.particle_count):

        wp.launch(kernel=eval_contacts,
                    dim=model.particle_count,
                    inputs=[state.particle_q, state.particle_qd, model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_kf, model.soft_contact_mu, model.soft_contact_distance, model.ground_plane],
                    outputs=[particle_f],
                    device=model.device)

    # tetrahedral FEM
    if (model.tet_count):

        wp.launch(kernel=eval_tetrahedra,
                  dim=model.tet_count,
                  inputs=[state.particle_q, state.particle_qd, model.tet_indices, model.tet_poses, model.tet_activations, model.tet_materials],
                  outputs=[particle_f],
                  device=model.device)

    if (model.body_count and model.contact_count > 0 and model.ground):

        wp.launch(kernel=eval_body_contacts,
                  dim=model.contact_count,
                  inputs=[
                      state.body_q,
                      state.body_qd,
                      model.body_com,
                      model.contact_body0,
                      model.contact_point0,
                      model.contact_dist,
                      model.contact_material,
                      model.shape_materials
                  ],
                  outputs=[
                      body_f
                  ],
                  device=model.device)

    if (model.body_count):

        wp.launch(kernel=eval_body_joints,
                  dim=model.body_count,
                  inputs=[
                      state.body_q,
                      state.body_qd,
                      model.body_com,
                      model.joint_type,
                      model.joint_parent,
                      model.joint_X_p,
                      model.joint_X_c,
                      model.joint_axis,
                      model.joint_target,
                      model.joint_act,
                      model.joint_target_ke,
                      model.joint_target_kd,
                      model.joint_limit_lower,
                      model.joint_limit_upper,
                      model.joint_limit_ke,
                      model.joint_limit_kd
                  ],
                  outputs=[
                      body_f
                  ],
                  device=model.device)

    # particle shape contact
    if (model.particle_count and model.shape_count):
        
        wp.launch(kernel=eval_soft_contacts,
                    dim=model.soft_contact_max,
                    inputs=[
                        state.particle_q, 
                        state.particle_qd,
                        state.body_q,
                        state.body_qd,
                        model.body_com,
                        model.soft_contact_ke,
                        model.soft_contact_kd, 
                        model.soft_contact_kf, 
                        model.soft_contact_mu,
                        model.soft_contact_count,
                        model.soft_contact_particle,
                        model.soft_contact_body,
                        model.soft_contact_body_pos,
                        model.soft_contact_body_vel,
                        model.soft_contact_normal,
                        model.soft_contact_distance],
                        # outputs
                    outputs=[
                        particle_f,
                        body_f],
                    device=model.device)


    # evaluate muscle actuation
    if (model.muscle_count):
        
        wp.launch(
            kernel=eval_muscles,
            dim=model.muscle_count,
            inputs=[
                state.body_q,
                state.body_qd,
                model.muscle_start,
                model.muscle_params,
                model.muscle_links,
                model.muscle_points,
                model.muscle_activation
            ],
            outputs=[
                body_f
            ],
            device=model.device,
            preserve_output=True)


    # if (model.articulation_count):
        
    #     # evaluate joint torques
    #     wp.launch(
    #         kernel=eval_body_tau,
    #         dim=model.articulation_count,
    #         inputs=[
    #             model.articulation_joint_start,
    #             model.joint_type,
    #             model.joint_parent,
    #             model.joint_q_start,
    #             model.joint_qd_start,
    #             state.joint_q,
    #             state.joint_qd,
    #             state.joint_act,
    #             model.joint_target,
    #             model.joint_target_ke,
    #             model.joint_target_kd,
    #             model.joint_limit_lower,
    #             model.joint_limit_upper,
    #             model.joint_limit_ke,
    #             model.joint_limit_kd,
    #             model.joint_axis,
    #             state.joint_S_s,
    #             state.body_f_s
    #         ],
    #         outputs=[
    #             state.body_ft_s,
    #             state.joint_tau
    #         ],
    #         device=model.device,
    #         preserve_output=True)





class SemiImplicitIntegrator:
    """A semi-implicit integrator using symplectic Euler

    After constructing `Model` and `State` objects this time-integrator
    may be used to advance the simulation state forward in time.

    Semi-implicit time integration is a variational integrator that 
    preserves energy, however it not unconditionally stable, and requires a time-step
    small enough to support the required stiffness and damping forces.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

    Example:

        >>> integrator = wp.SemiImplicitIntegrator()
        >>>
        >>> # simulation loop
        >>> for i in range(100):
        >>>     state = integrator.forward(model, state, dt)

    """

    def __init__(self):
        pass


    def simulate(self, model, state_in, state_out, dt):

        with wp.ScopedTimer("simulate", False):

            particle_f = None
            body_f = None

            # alloc particle force buffer
            if (model.particle_count):
                particle_f = state_out.particle_f
                #particle_f.zero_()

            if (model.body_count):
                body_f = state_out.body_f
                #body_f.zero()

            compute_forces(model, state_in, particle_f, body_f)

            #-------------------------------------
            # integrate bodies

            if (model.body_count):

                wp.launch(
                    kernel=integrate_bodies,
                    dim=model.body_count,
                    inputs=[
                        state_in.body_q,
                        state_in.body_qd,
                        state_in.body_f,
                        body_f,
                        model.body_com,
                        model.body_mass,
                        model.body_inertia,
                        model.body_inv_mass,
                        model.body_inv_inertia,
                        model.gravity,
                        dt,
                    ],
                    outputs=[
                        state_out.body_q,
                        state_out.body_qd
                    ],
                    device=model.device)

            #----------------------------
            # integrate particles

            if (model.particle_count):

                wp.launch(
                    kernel=integrate_particles,
                    dim=model.particle_count,
                    inputs=[
                        state_in.particle_q, 
                        state_in.particle_qd,
                        state_in.particle_f,
                        particle_f, 
                        model.particle_inv_mass, 
                        model.gravity, 
                        dt
                    ],
                    outputs=[
                        state_out.particle_q, 
                        state_out.particle_qd
                        ],
                    device=model.device)

            return state_out


@wp.kernel
def compute_particle_residual(particle_qd_0: wp.array(dtype=wp.vec3),
                            particle_qd_1: wp.array(dtype=wp.vec3),
                            particle_f: wp.array(dtype=wp.vec3),
                            particle_m: wp.array(dtype=float),
                            gravity: wp.vec3,
                            dt: float,
                            residual: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    m = wp.load(particle_m, tid)
    v1 = wp.load(particle_qd_1, tid)
    v0 = wp.load(particle_qd_0, tid)
    f = wp.load(particle_f, tid)

    err = wp.vec3()

    if (m > 0.0):
        #err = (v1-v0)*m - f*dt - gravity*dt*m   
        #invm = 1.0/(m + 1.e+3*dt*dt*16.0)
        #err = (v1-v0)*m - f*dt - gravity*dt*m
        #err = err*invm
        err = (v1-v0)*m - f*dt - gravity*dt*m

    wp.store(residual, tid, err)
 

@wp.kernel
def update_particle_position(
    particle_q_0: wp.array(dtype=wp.vec3),
    particle_q_1: wp.array(dtype=wp.vec3),
    particle_qd_1: wp.array(dtype=wp.vec3),
    x: wp.array(dtype=wp.vec3),
    dt: float):

    tid = wp.tid()

    qd_1 = wp.load(x, tid)
    
    q_0 = wp.load(particle_q_0, tid)
    q_1 = q_0 + qd_1*dt

    wp.store(particle_q_1, tid, q_1)
    wp.store(particle_qd_1, tid, qd_1)



def compute_residual(model, state_in, state_out, particle_f, residual, dt):

    wp.launch(
        kernel=compute_particle_residual,
        dim=model.particle_count,
        inputs=[
            state_in.particle_qd,
            state_out.particle_qd,
            particle_f,
            model.particle_mass,
            model.gravity,
            dt,
            residual.astype(dtype=wp.vec3)
        ], 
        device=model.device)

def init_state(model, state_in, state_out, dt):

    wp.launch(
        kernel=integrate_particles,
        dim=model.particle_count,
        inputs=[
            state_in.particle_q, 
            state_in.particle_qd, 
            state_in.particle_f, 
            model.particle_inv_mass, 
            model.gravity, 
            dt
        ],
        outputs=[
            state_out.particle_q, 
            state_out.particle_qd
            ],
        device=model.device)


# compute the final positions given output velocity (x)
def update_state(model, state_in, state_out, x, dt):

    wp.launch(
        kernel=update_particle_position,
        dim=model.particle_count,
        inputs=[
            state_in.particle_q,
            state_out.particle_q,
            state_out.particle_qd,
            x,
            dt
        ],
        device=model.device)



class VariationalImplicitIntegrator:

    def __init__(self, model, solver="gd", alpha=0.1, max_iters=32, report=False):

        self.solver = solver
        self.alpha = alpha
        self.max_iters = max_iters
        self.report = report

        self.opt = Optimizer(model.particle_count*3, mode=self.solver, device=model.device)

        # allocate temporary space for evaluating particle forces
        self.particle_f = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)

    def simulate(self, model, state_in, state_out, dt): 

        if (state_in is state_out):
            raise RuntimeError("Implicit integrators require state objects to not alias each other")


        with wp.ScopedTimer("simulate", False):

            # alloc particle force buffer
            if (model.particle_count):
                
                def residual_func(x, dfdx):

                    self.particle_f.zero_()

                    update_state(model, state_in, state_out, x.astype(wp.vec3), dt)
                    compute_forces(model, state_out, self.particle_f, None)
                    compute_residual(model, state_in, state_out, self.particle_f, dfdx, dt)


                # initialize oututput state using the input velocity to create 'predicted state'
                init_state(model, state_in, state_out, dt)

                # our optimization variable
                x = state_out.particle_qd.astype(dtype=float)

                self.opt.solve(
                    x=x, 
                    grad_func=residual_func,
                    max_iters=self.max_iters,
                    alpha=self.alpha,
                    report=self.report)

                # final update to output state with solved velocity
                update_state(model, state_in, state_out, x.astype(wp.vec3), dt)
  

            return state_out