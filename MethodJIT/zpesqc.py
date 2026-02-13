"""
JIT-compiled ZPE-corrected SQC (zpesqc) method.
Reference: Method/zpesqc.py
Identical to SQC but uses per-state gamma0 in force calculation.
"""
import numpy as np
from numba import njit, prange

_jit = dict(cache=True, fastmath=True, boundscheck=False, error_model='numpy')


@njit(**_jit)
def umap_sqc(qF, pF, dt, VMat, NStates):
    """Second-order symplectic propagator for single-set mapping variables."""
    qFin = qF.copy()
    pFin = pF.copy()
    VxqF = np.zeros(NStates)
    for k in range(NStates):
        s = 0.0
        for l in range(NStates):
            s += VMat[k, l] * qFin[l]
        VxqF[k] = s
    for k in range(NStates):
        pF[k] -= 0.5 * dt * VxqF[k]
    for k in range(NStates):
        s = 0.0
        for l in range(NStates):
            s += VMat[k, l] * pFin[l]
        qF[k] += dt * s
    dt2h = dt * dt * 0.5
    for k in range(NStates):
        s = 0.0
        for l in range(NStates):
            s += VMat[k, l] * VxqF[l]
        qF[k] -= dt2h * s
    for k in range(NStates):
        s = 0.0
        for l in range(NStates):
            s += VMat[k, l] * qF[l]
        pF[k] -= 0.5 * dt * s


@njit(**_jit)
def force_zpesqc(qF, pF, gamma0, dHij, dH0, NStates, ndof):
    """Force with per-state gamma0 (ZPE corrected)."""
    F = np.empty(ndof)
    for d in range(ndof):
        F[d] = -dH0[d]
    for i in range(NStates):
        qi2 = qF[i] * qF[i] + pF[i] * pF[i] - 2.0 * gamma0[i]
        for d in range(ndof):
            F[d] -= 0.5 * dHij[i, i, d] * qi2
        for j in range(i + 1, NStates):
            cross = qF[i] * qF[j] + pF[i] * pF[j]
            for d in range(ndof):
                F[d] -= dHij[i, j, d] * cross
    return F


@njit(**_jit)
def pop_square(qF, pF, gamma, NStates):
    rho = np.zeros((NStates, NStates), dtype=np.complex128)
    eta = np.empty(NStates)
    for i in range(NStates):
        eta[i] = 0.5 * (qF[i] * qF[i] + pF[i] * pF[i])
    for i in range(NStates):
        rho[i, i] = 1.0 + 0.0j
    for i in range(NStates):
        for j in range(NStates):
            val = eta[j] - (1.0 if i == j else 0.0)
            if val < 0.0 or val > 2.0 * gamma:
                rho[i, i] = 0.0 + 0.0j
    return rho


@njit(**_jit)
def pop_triangle(qF, pF, NStates):
    rho = np.zeros((NStates, NStates), dtype=np.complex128)
    eta = np.empty(NStates)
    for i in range(NStates):
        eta[i] = 0.5 * (qF[i] * qF[i] + pF[i] * pF[i])
    for i in range(NStates):
        rho[i, i] = 1.0 + 0.0j
    for i in range(NStates):
        for j in range(NStates):
            if (i == j and eta[j] < 1.0) or (i != j and eta[j] >= 1.0):
                rho[i, i] = 0.0 + 0.0j
    return rho


_kernel_cache = {}

def make_zpesqc_kernel(hel_fn, dhel_fn, dhel0_fn):
    key = (hel_fn, dhel_fn, dhel0_fn)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @njit(parallel=True, **_jit)
    def run_traj(NTraj, NSteps, NStates, nskip, EStep, dtN, M_mass,
                 initState, use_square,
                 R_rand, P_rand, sigR, sigP,
                 map_rand,   # (NTraj, 2, NStates)
                 c, epsilon, Delta, omega):
        n_skip = NSteps // nskip
        dtE = dtN / EStep
        half_e_floor = EStep // 2
        half_e_ceil = (EStep + 1) // 2
        ndof = len(sigR)
        pi = np.pi
        gamma_sq = (np.sqrt(3.0) - 1.0) / 2.0
        gamma_tri = 1.0 / 3.0
        gamma = gamma_sq if use_square else gamma_tri
        rho_all = np.zeros((NTraj, NStates, NStates, n_skip), dtype=np.complex128)

        for itraj in prange(NTraj):
            R = np.empty(ndof)
            P = np.empty(ndof)
            for d in range(ndof):
                R[d] = R_rand[itraj, d] * sigR[d]
                P[d] = P_rand[itraj, d] * sigP[d]

            eta = np.empty(NStates)
            theta = np.empty(NStates)
            if use_square:
                for s in range(NStates):
                    eta[s] = 2.0 * gamma_sq * map_rand[itraj, 0, s]
                    theta[s] = 2.0 * pi * map_rand[itraj, 1, s]
            else:
                for s in range(NStates):
                    eta[s] = map_rand[itraj, 0, s]
                    theta[s] = 2.0 * pi * map_rand[itraj, 1, s]

            # compute gamma0 before adding 1 to initState
            gamma0 = np.empty(NStates)
            for s in range(NStates):
                gamma0[s] = eta[s]

            eta[initState] += 1.0
            qF = np.empty(NStates)
            pF = np.empty(NStates)
            for s in range(NStates):
                qF[s] = np.sqrt(2.0 * eta[s]) * np.cos(theta[s])
                pF[s] = -np.sqrt(2.0 * eta[s]) * np.sin(theta[s])

            Hij = hel_fn(R, c, epsilon, Delta)
            dHij = dhel_fn(R, c)
            dH0 = dhel0_fn(R, omega)

            iskip = 0
            for step in range(NSteps):
                if step % nskip == 0:
                    if use_square:
                        rho = pop_square(qF, pF, gamma_sq, NStates)
                    else:
                        rho = pop_triangle(qF, pF, NStates)
                    for a in range(NStates):
                        for b in range(NStates):
                            rho_all[itraj, a, b, iskip] = rho[a, b]
                    iskip += 1

                # VelVer
                for _t in range(half_e_floor):
                    umap_sqc(qF, pF, dtE, Hij, NStates)

                F1 = force_zpesqc(qF, pF, gamma0, dHij, dH0, NStates, ndof)
                v = np.empty(ndof)
                for d in range(ndof):
                    v[d] = P[d] / M_mass
                    R[d] += v[d] * dtN + 0.5 * F1[d] * dtN * dtN / M_mass

                Hij = hel_fn(R, c, epsilon, Delta)
                dHij = dhel_fn(R, c)
                dH0 = dhel0_fn(R, omega)
                F2 = force_zpesqc(qF, pF, gamma0, dHij, dH0, NStates, ndof)
                for d in range(ndof):
                    v[d] += 0.5 * (F1[d] + F2[d]) * dtN / M_mass
                    P[d] = v[d] * M_mass

                Hij = hel_fn(R, c, epsilon, Delta)
                for _t in range(half_e_ceil):
                    umap_sqc(qF, pF, dtE, Hij, NStates)

        rho_ensemble = np.zeros((NStates, NStates, n_skip), dtype=np.complex128)
        for itraj in range(NTraj):
            for a in range(NStates):
                for b in range(NStates):
                    for t in range(n_skip):
                        rho_ensemble[a, b, t] += rho_all[itraj, a, b, t]
        return rho_ensemble
    _kernel_cache[key] = run_traj
    return run_traj
