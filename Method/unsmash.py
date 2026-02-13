"""
Method/unsmash.py

Plain-Python reference implementation of unSMASH (uncoupled Spheres
Multi-state Mapping Approach to Surface Hopping).

Reference:
  Lawrence, Mannouch & Richardson,
  "A Size-Consistent Multi-State Mapping Approach to Surface Hopping"

This mirrors MethodJIT/unsmash.py exactly (same algorithm, same RNG
consumption order) so that the two can be compared bitwise in tests.
"""

import numpy as np
import time


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


# ================================================================== #
#  Bloch-sphere <-> effective two-state coefficients                  #
# ================================================================== #

def sphere_to_coeffs(Sx, Sy, Sz):
    cn_real = np.sqrt(max((1.0 + Sz) * 0.5, 0.0))
    if cn_real < 1e-14:
        cn = 0.0 + 0.0j
        cb = 1.0 + 0.0j
    else:
        cn = cn_real + 0.0j
        cb = (Sx + 1j * Sy) / (2.0 * cn_real)
    return cn, cb


def coeffs_to_sphere(cn, cb):
    Sz = (cn * cn.conjugate()).real - (cb * cb.conjugate()).real
    prod = cn.conjugate() * cb
    Sx = 2.0 * prod.real
    Sy = 2.0 * prod.imag
    return Sx, Sy, Sz


def z_rotate_sphere(Sx, Sy, Sz, E_acst, E_b, dt):
    phi = (E_acst - E_b) * dt * 0.5
    cp = np.cos(phi)
    sp = np.sin(phi)
    Sx_new = Sx * cp + Sy * sp
    Sy_new = -Sx * sp + Sy * cp
    return Sx_new, Sy_new, Sz


# ================================================================== #
#  Force on active surface                                            #
# ================================================================== #

def Force(dHij, dH0, acst, U, NStates, ndof):
    F = np.empty(ndof)
    for d in range(ndof):
        F[d] = -dH0[d]
        s = 0.0 + 0.0j
        for i in range(NStates):
            for j in range(NStates):
                s += U[i, acst].conjugate() * dHij[i, j, d] * U[j, acst]
        F[d] -= s.real
    return F


# ================================================================== #
#  Hop detection                                                      #
# ================================================================== #

def checkHop(S, acst, NStates):
    hop_needed = False
    target = acst
    min_sz = 0.0
    for b in range(NStates):
        if b == acst:
            continue
        sz = S[b, 2]
        if sz < min_sz:
            min_sz = sz
            target = b
            hop_needed = True
    return hop_needed, target


# ================================================================== #
#  Hop direction + momentum rescaling                                 #
# ================================================================== #

def attempt_hop(P, E, U, dHij, acst, newacst, M_mass, NStates, ndof):
    if acst == newacst:
        return P.copy(), False

    a, b = acst, newacst
    dE = E[b] - E[a]

    if abs(dE) < 1e-30:
        return P.copy(), False

    sqrtM = np.sqrt(M_mass)

    dk = np.zeros(ndof)
    for d in range(ndof):
        dH_ab = 0.0 + 0.0j
        for i in range(NStates):
            for j in range(NStates):
                dH_ab += U[i, a].conjugate() * dHij[i, j, d] * U[j, b]
        dk[d] = dH_ab.real / dE / sqrtM

    dk_dot = 0.0
    P_dk = 0.0
    for d in range(ndof):
        dk_dot += dk[d] * dk[d]
        P_dk += (P[d] / sqrtM) * dk[d]

    if dk_dot < 1e-30:
        return P.copy(), False

    P_proj_norm2 = P_dk * P_dk / dk_dot

    P_new = np.empty(ndof)
    if P_proj_norm2 < 2.0 * dE:
        # Rejected -- reverse projected component
        for d in range(ndof):
            proj = P_dk * dk[d] / dk_dot
            orth = P[d] / sqrtM - proj
            P_new[d] = (orth - proj) * sqrtM
        return P_new, False
    else:
        # Accepted -- rescale
        scale = np.sqrt(P_proj_norm2 - 2.0 * dE) / np.sqrt(P_proj_norm2)
        for d in range(ndof):
            proj = P_dk * dk[d] / dk_dot
            orth = P[d] / sqrtM - proj
            P_new[d] = (orth + scale * proj) * sqrtM
        return P_new, True


# ================================================================== #
#  Sphere relabelling (Eqs. 10-12)                                    #
# ================================================================== #

def relabel_spheres(S, old_acst, new_acst, NStates):
    ni = old_acst
    nf = new_acst
    sx_hop = S[nf, 0]
    sy_hop = S[nf, 1]
    sz_hop = S[nf, 2]
    S[ni, 0] = sx_hop
    S[ni, 1] = -sy_hop
    S[ni, 2] = -sz_hop
    S[nf, 0] = 0.0
    S[nf, 1] = 0.0
    S[nf, 2] = 0.0


# ================================================================== #
#  g-factor weights (Appendix A)                                      #
# ================================================================== #

def compute_g_factors(S, acst, U, initState, NStates):
    j = initState
    n = acst

    rho_P = 1.0
    for mu in range(NStates):
        if mu == n:
            continue
        rho_P *= 2.0 * abs(S[mu, 2])

    jn = U[j, n]
    jn_abs2 = (jn * jn.conjugate()).real

    gP = rho_P * jn_abs2
    gC = 2.0 * jn_abs2

    for a in range(NStates):
        if a == n:
            continue
        aj = U[j, a].conjugate()
        z = jn * aj
        Sx_na = S[a, 0]
        Sy_na = S[a, 1]
        gP += 2.0 * z.real * Sx_na - 2.0 * z.imag * Sy_na
        gC += 3.0 * z.real * Sx_na - 3.0 * z.imag * Sy_na

    return gP, gC


# ================================================================== #
#  Population estimator                                               #
# ================================================================== #

def pop_unsmash(S, acst, U, gP, gC, NStates):
    N = float(NStates)
    nt = acst
    rho = np.zeros((NStates, NStates), dtype=np.complex128)

    for jp in range(NStates):
        for jpp in range(NStates):
            rho[jp, jpp] += N * gP * U[jp, nt] * U[jpp, nt].conjugate()

    for b in range(NStates):
        if b == nt:
            continue
        sigma_nb = (S[b, 0] - 1j * S[b, 1]) * 0.5
        sigma_nb_conj = (S[b, 0] + 1j * S[b, 1]) * 0.5
        for jp in range(NStates):
            for jpp in range(NStates):
                rho[jp, jpp] += N * gC * (
                    U[jp, nt] * sigma_nb * U[jpp, b].conjugate()
                    + U[jp, b] * sigma_nb_conj * U[jpp, nt].conjugate()
                )
    return rho


# ================================================================== #
#  VelVer with uncoupled Bloch-sphere dynamics                        #
# ================================================================== #

def VelVer(R, P, S, E, U, F1, acst, dt, M_mass, NStates, ndof,
           Hel_func, dHel_func, dHel0_func):
    """One VelVer step evolving N-1 Bloch spheres independently."""
    cD_all = np.zeros((NStates, NStates), dtype=np.complex128)

    for b in range(NStates):
        if b == acst:
            continue
        Sx, Sy, Sz = z_rotate_sphere(S[b, 0], S[b, 1], S[b, 2],
                                     E[acst], E[b], dt)
        cn, cb_ = sphere_to_coeffs(Sx, Sy, Sz)
        for i in range(NStates):
            cD_all[b, i] = U[i, acst] * cn + U[i, b] * cb_

    v = np.empty(ndof)
    for d in range(ndof):
        v[d] = P[d] / M_mass
        R[d] += v[d] * dt + 0.5 * F1[d] * dt * dt / M_mass

    Hij = Hel_func(R) + 0j
    dHij = dHel_func(R)
    dH0 = dHel0_func(R)
    E_new, U_new = np.linalg.eigh(Hij)

    F2 = Force(dHij, dH0, acst, U_new, NStates, ndof)
    for d in range(ndof):
        v[d] += 0.5 * (F1[d] + F2[d]) * dt / M_mass
        P[d] = v[d] * M_mass

    for b in range(NStates):
        if b == acst:
            continue
        cn_new = 0.0 + 0.0j
        cb_new = 0.0 + 0.0j
        for i in range(NStates):
            cn_new += U_new[i, acst].conjugate() * cD_all[b, i]
            cb_new += U_new[i, b].conjugate() * cD_all[b, i]

        norm2 = (cn_new * cn_new.conjugate()).real + \
                (cb_new * cb_new.conjugate()).real
        if norm2 > 1e-30:
            inv_norm = 1.0 / np.sqrt(norm2)
            cn_new = cn_new * inv_norm
            cb_new = cb_new * inv_norm
        else:
            cn_new = 1.0 + 0.0j
            cb_new = 0.0 + 0.0j

        Sx, Sy, Sz = coeffs_to_sphere(cn_new, cb_new)
        Sx, Sy, Sz = z_rotate_sphere(Sx, Sy, Sz, E_new[acst], E_new[b], dt)
        S[b, 0] = Sx
        S[b, 1] = Sy
        S[b, 2] = Sz

    return R, P, S, Hij, dHij, dH0, E_new, U_new, F2


# ================================================================== #
#  Main trajectory loop                                               #
# ================================================================== #

def runTraj(parameters):
    """
    Run an ensemble of unSMASH trajectories.

    RNG consumption order per trajectory (matches JIT pre-generation):
      1. ndof interleaved normals: R[d], P[d] for d=0..ndof-1
      2. NStates * 2 uniforms: sphere_rand[b, 0..1] for b=0..NStates-1
         (the acst-slot draws are consumed but unused)
      3. 1 uniform: acst_rand -> initial active state

    Returns rho_ensemble (NStates, NStates, n_skip) complex, un-normalised.
    """
    try:
        np.random.seed(parameters.SEED)
    except AttributeError:
        pass

    NSteps    = parameters.NSteps
    NTraj     = parameters.NTraj
    NStates   = parameters.NStates
    initState = parameters.initState
    nskip     = parameters.nskip
    dtN       = parameters.dtN
    M_mass    = parameters.M
    maxhop    = getattr(parameters, 'maxhop', 10)

    Hel_func   = parameters.Hel
    dHel_func  = parameters.dHel
    dHel0_func = parameters.dHel0
    initR_func = parameters.initR

    if NSteps % nskip == 0:
        pl = 0
    else:
        pl = 1
    n_skip = NSteps // nskip + pl

    rho_ensemble = np.zeros((NStates, NStates, n_skip), dtype=complex)

    for itraj in range(NTraj):
        t0 = time.time()

        # 1. Nuclear ICs (interleaved R[d], P[d])
        R, P = initR_func()

        # 2. Sphere randoms: NStates * 2 uniforms (all slots, including acst)
        sphere_rand = np.empty((NStates, 2))
        for b in range(NStates):
            sphere_rand[b, 0] = np.random.random()
            sphere_rand[b, 1] = np.random.random()

        # 3. Active state random
        acst_rand = np.random.random()
        acst = int(acst_rand * NStates)
        if acst >= NStates:
            acst = NStates - 1

        # Initial QM
        Hij = Hel_func(R) + 0j
        dHij = dHel_func(R)
        dH0 = dHel0_func(R)
        E, U = np.linalg.eigh(Hij)

        # Initialise Bloch spheres on upper hemisphere
        ndof = len(R)
        S = np.zeros((NStates, 3))
        for b in range(NStates):
            if b == acst:
                continue
            u = sphere_rand[b, 0]
            v = sphere_rand[b, 1]
            Sz = u
            phi = 2.0 * np.pi * v
            sin_theta = np.sqrt(max(1.0 - Sz * Sz, 0.0))
            S[b, 0] = sin_theta * np.cos(phi)
            S[b, 1] = sin_theta * np.sin(phi)
            S[b, 2] = Sz

        # g-factor weights
        gP, gC = compute_g_factors(S, acst, U, initState, NStates)

        # Initial force
        F1 = Force(dHij, dH0, acst, U, NStates, ndof)

        iskip = 0
        for step in range(NSteps):
            # Estimator
            if step % nskip == 0:
                rho_ensemble[:, :, iskip] += pop_unsmash(
                    S, acst, U, gP, gC, NStates)
                iskip += 1

            # Full VelVer step
            R_new = R.copy()
            P_new = P.copy()
            S_new = S.copy()
            (R_new, P_new, S_new, Hij_new, dHij_new, dH0_new,
             E_new, U_new, F2_new) = VelVer(
                R_new, P_new, S_new, E.copy(), U.copy(), F1.copy(),
                acst, dtN, M_mass, NStates, ndof,
                Hel_func, dHel_func, dHel0_func)

            # Check hop
            hop_needed, new_acst = checkHop(S_new, acst, NStates)

            if hop_needed:
                P_attempt, feasible = attempt_hop(
                    P_new, E_new, U_new, dHij_new,
                    acst, new_acst, M_mass, NStates, ndof)

                if feasible:
                    R0 = R.copy()
                    P0 = P.copy()
                    S0 = S.copy()
                    E0 = E.copy()
                    U0 = U.copy()
                    F1_0 = F1.copy()

                    tL = 0.0
                    tR = dtN

                    Rm = R0.copy()
                    Pm = P0.copy()
                    Sm = S0.copy()
                    Em = E0.copy()
                    Um = U0.copy()
                    F1m = F1_0.copy()
                    dHijm = dHij_new.copy()
                    dH0m = dH0_new.copy()

                    last_tm = 0.0
                    for _bisect in range(maxhop):
                        tm = 0.5 * (tL + tR)
                        last_tm = tm
                        Rt = R0.copy()
                        Pt = P0.copy()
                        St = S0.copy()
                        (Rt, Pt, St, _Ht, dHt, dH0t,
                         Et, Ut, F2t) = VelVer(
                            Rt, Pt, St, E0.copy(), U0.copy(),
                            F1_0.copy(), acst, tm, M_mass,
                            NStates, ndof,
                            Hel_func, dHel_func, dHel0_func)

                        hop_tm, _ = checkHop(St, acst, NStates)
                        if not hop_tm:
                            tL = tm
                        else:
                            tR = tm

                        Rm[:] = Rt
                        Pm[:] = Pt
                        Sm[:] = St
                        Em[:] = Et
                        Um[:, :] = Ut
                        F1m[:] = F2t
                        dHijm[:, :, :] = dHt
                        dH0m[:] = dH0t

                    P_hop, accepted2 = attempt_hop(
                        Pm, Em, Um, dHijm,
                        acst, new_acst, M_mass, NStates, ndof)
                    Pm[:] = P_hop
                    if accepted2:
                        relabel_spheres(Sm, acst, new_acst, NStates)
                        acst = new_acst

                    remaining = dtN - last_tm
                    (Rm, Pm, Sm, _Hf, dHf, dH0f,
                     Ef, Uf, F2f) = VelVer(
                        Rm, Pm, Sm, Em, Um, F1m, acst, remaining,
                        M_mass, NStates, ndof,
                        Hel_func, dHel_func, dHel0_func)

                    R[:] = Rm
                    P[:] = Pm
                    S[:] = Sm
                    E = Ef
                    U = Uf
                    F1 = F2f
                    dHij = dHf
                    dH0 = dH0f
                else:
                    # Hop not feasible -- reversed momentum
                    R[:] = R_new
                    P[:] = P_attempt
                    S[:] = S_new
                    E = E_new
                    U = U_new
                    F1 = F2_new
                    dHij = dHij_new
                    dH0 = dH0_new
            else:
                R[:] = R_new
                P[:] = P_new
                S[:] = S_new
                E = E_new
                U = U_new
                F1 = F2_new
                dHij = dHij_new
                dH0 = dH0_new

        time_taken = time.time() - t0
        print(f"Time taken: {time_taken:.2f} seconds")

    return rho_ensemble
