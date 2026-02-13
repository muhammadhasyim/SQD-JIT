"""
JIT-compiled Mean-Field Ehrenfest (MFE) method.
Reference: Method/mfe.py
"""
import numpy as np
from numba import njit, prange

_jit = dict(cache=True, fastmath=True, boundscheck=False, error_model='numpy')


@njit(**_jit)
def propagate_ci(ci, Vij, dt, NStates):
    """RK4 propagation of electronic coefficients."""
    c = ci.copy()
    # k1
    k1 = np.empty(NStates, dtype=np.complex128)
    for a in range(NStates):
        s = 0.0 + 0.0j
        for b in range(NStates):
            s += Vij[a, b] * c[b]
        k1[a] = -1j * s
    # k2
    ct = np.empty(NStates, dtype=np.complex128)
    for a in range(NStates):
        ct[a] = c[a] + (dt / 2.0) * k1[a]
    k2 = np.empty(NStates, dtype=np.complex128)
    for a in range(NStates):
        s = 0.0 + 0.0j
        for b in range(NStates):
            s += Vij[a, b] * ct[b]
        k2[a] = -1j * s
    # k3
    for a in range(NStates):
        ct[a] = c[a] + (dt / 2.0) * k2[a]
    k3 = np.empty(NStates, dtype=np.complex128)
    for a in range(NStates):
        s = 0.0 + 0.0j
        for b in range(NStates):
            s += Vij[a, b] * ct[b]
        k3[a] = -1j * s
    # k4
    for a in range(NStates):
        ct[a] = c[a] + dt * k3[a]
    k4 = np.empty(NStates, dtype=np.complex128)
    for a in range(NStates):
        s = 0.0 + 0.0j
        for b in range(NStates):
            s += Vij[a, b] * ct[b]
        k4[a] = -1j * s
    # update
    for a in range(NStates):
        c[a] = c[a] + (dt / 6.0) * (k1[a] + 2.0 * k2[a] + 2.0 * k3[a] + k4[a])
    return c


@njit(**_jit)
def force_mfe(ci, dHij, dH0, NStates, ndof):
    F = np.empty(ndof)
    for d in range(ndof):
        F[d] = -dH0[d]
    for i in range(NStates):
        rho_ii = (ci[i] * ci[i].conjugate()).real
        for d in range(ndof):
            F[d] -= dHij[i, i, d] * rho_ii
        for j in range(i + 1, NStates):
            rho_ij = (ci[i].conjugate() * ci[j]).real
            for d in range(ndof):
                F[d] -= 2.0 * dHij[i, j, d] * rho_ij
    return F


_kernel_cache = {}

def make_mfe_kernel(hel_fn, dhel_fn, dhel0_fn):
    key = (hel_fn, dhel_fn, dhel0_fn)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @njit(parallel=True, **_jit)
    def run_traj(NTraj, NSteps, NStates, nskip, EStep, dtN, M_mass,
                 initState, R_rand, P_rand, sigR, sigP,
                 c, epsilon, Delta, omega):
        n_skip = NSteps // nskip
        dtE = dtN / EStep
        half_e_floor = EStep // 2
        half_e_ceil = (EStep + 1) // 2
        ndof = len(sigR)
        rho_all = np.zeros((NTraj, NStates, NStates, n_skip), dtype=np.complex128)

        for itraj in prange(NTraj):
            R = np.empty(ndof)
            P = np.empty(ndof)
            for d in range(ndof):
                R[d] = R_rand[itraj, d] * sigR[d]
                P[d] = P_rand[itraj, d] * sigP[d]

            ci = np.zeros(NStates, dtype=np.complex128)
            ci[initState] = 1.0 + 0.0j

            Hij = hel_fn(R, c, epsilon, Delta) 
            # make complex
            HijC = np.empty((NStates, NStates), dtype=np.complex128)
            for a in range(NStates):
                for b in range(NStates):
                    HijC[a, b] = Hij[a, b] + 0.0j
            dHij = dhel_fn(R, c)
            dH0 = dhel0_fn(R, omega)
            F1 = force_mfe(ci, dHij, dH0, NStates, ndof)

            iskip = 0
            for step in range(NSteps):
                if step % nskip == 0:
                    # pop = outer(ci*, ci)
                    for a in range(NStates):
                        for b in range(NStates):
                            rho_all[itraj, a, b, iskip] = ci[a].conjugate() * ci[b]
                    iskip += 1

                # --- VelVer ---
                ci_w = ci.copy()
                # half electronic
                for _t in range(half_e_floor):
                    ci_w = propagate_ci(ci_w, HijC, dtE, NStates)
                # normalize
                norm = 0.0
                for a in range(NStates):
                    norm += (ci_w[a].conjugate() * ci_w[a]).real
                for a in range(NStates):
                    ci_w[a] = ci_w[a] / norm
                ci = ci_w.copy()

                # nuclear
                v = np.empty(ndof)
                for d in range(ndof):
                    v[d] = P[d] / M_mass
                for d in range(ndof):
                    R[d] += v[d] * dtN + 0.5 * F1[d] * dtN * dtN / M_mass

                Hij = hel_fn(R, c, epsilon, Delta)
                for a in range(NStates):
                    for b in range(NStates):
                        HijC[a, b] = Hij[a, b] + 0.0j
                dHij = dhel_fn(R, c)
                dH0 = dhel0_fn(R, omega)
                F2 = force_mfe(ci, dHij, dH0, NStates, ndof)
                for d in range(ndof):
                    v[d] += 0.5 * (F1[d] + F2[d]) * dtN / M_mass
                    P[d] = v[d] * M_mass
                F1 = F2

                # half electronic
                for _t in range(half_e_ceil):
                    ci_w = propagate_ci(ci_w, HijC, dtE, NStates)
                norm = 0.0
                for a in range(NStates):
                    norm += (ci_w[a].conjugate() * ci_w[a]).real
                for a in range(NStates):
                    ci_w[a] = ci_w[a] / norm
                ci = ci_w.copy()

        rho_ensemble = np.zeros((NStates, NStates, n_skip), dtype=np.complex128)
        for itraj in range(NTraj):
            for a in range(NStates):
                for b in range(NStates):
                    for t in range(n_skip):
                        rho_ensemble[a, b, t] += rho_all[itraj, a, b, t]
        return rho_ensemble
    _kernel_cache[key] = run_traj
    return run_traj
