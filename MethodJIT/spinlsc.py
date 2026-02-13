"""
JIT-compiled spin-LSC (Linearized Semiclassical) method.
Reference: Method/spinlsc.py
Uses complex forward mapping variable zF only (no backward).
"""
import numpy as np
from numba import njit, prange

_jit = dict(cache=True, fastmath=True, boundscheck=False, error_model='numpy')


@njit(**_jit)
def umap_spinlsc(zF, dt, VMat, NStates):
    """Split-step symplectic propagator for complex mapping variable z."""
    Zreal = np.empty(NStates)
    Zimag = np.empty(NStates)
    for s in range(NStates):
        Zreal[s] = zF[s].real
        Zimag[s] = zF[s].imag

    # Propagate Imaginary first by dt/2
    for k in range(NStates):
        s = 0.0
        for l in range(NStates):
            s += VMat[k, l] * Zreal[l]
        Zimag[k] -= 0.5 * s * dt

    # Propagate Real by full dt
    for k in range(NStates):
        s = 0.0
        for l in range(NStates):
            s += VMat[k, l] * Zimag[l]
        Zreal[k] += s * dt

    # Propagate Imaginary final by dt/2
    for k in range(NStates):
        s = 0.0
        for l in range(NStates):
            s += VMat[k, l] * Zreal[l]
        Zimag[k] -= 0.5 * s * dt

    out = np.empty(NStates, dtype=np.complex128)
    for s in range(NStates):
        out[s] = Zreal[s] + 1j * Zimag[s]
    return out


@njit(**_jit)
def force_spinlsc(zF, gw, dHij, dH0, NStates, ndof):
    """Force using spin-LSC density matrix estimator."""
    # eta = Re(outer(zF*, zF) - gw * I)
    F = np.empty(ndof)
    for d in range(ndof):
        F[d] = -dH0[d]
    for i in range(NStates):
        eta_ii = (zF[i].conjugate() * zF[i]).real - gw
        for d in range(ndof):
            F[d] -= 0.5 * dHij[i, i, d] * eta_ii
        for j in range(i + 1, NStates):
            eta_ij = (zF[i].conjugate() * zF[j]).real
            for d in range(ndof):
                F[d] -= 2.0 * 0.5 * dHij[i, j, d] * eta_ij
    return F


@njit(**_jit)
def pop_spinlsc(zF, gw, NStates):
    """Population estimator: rho = 0.5*(outer(zF*, zF) - gw*I)."""
    rho = np.zeros((NStates, NStates), dtype=np.complex128)
    for i in range(NStates):
        for j in range(NStates):
            rho[i, j] = 0.5 * (zF[i].conjugate() * zF[j])
            if i == j:
                rho[i, j] -= 0.5 * gw
    return rho


_kernel_cache = {}

def make_spinlsc_kernel(hel_fn, dhel_fn, dhel0_fn):
    key = (hel_fn, dhel_fn, dhel0_fn)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @njit(parallel=True, **_jit)
    def run_traj(NTraj, NSteps, NStates, nskip, EStep, dtN, M_mass,
                 initState,
                 R_rand, P_rand, sigR, sigP,
                 phi_rand,   # (NTraj, NStates) uniform [0,1]
                 c, epsilon, Delta, omega):
        n_skip = NSteps // nskip
        dtE = dtN / EStep
        half_e_floor = EStep // 2
        half_e_ceil = (EStep + 1) // 2
        ndof = len(sigR)
        pi = np.pi
        gw = (2.0 / NStates) * (np.sqrt(NStates + 1.0) - 1.0)
        rho_all = np.zeros((NTraj, NStates, NStates, n_skip), dtype=np.complex128)

        for itraj in prange(NTraj):
            R = np.empty(ndof)
            P = np.empty(ndof)
            for d in range(ndof):
                R[d] = R_rand[itraj, d] * sigR[d]
                P[d] = P_rand[itraj, d] * sigP[d]

            # init mapping
            zF = np.empty(NStates, dtype=np.complex128)
            for s in range(NStates):
                rF = np.sqrt(gw)
                if s == initState:
                    rF = np.sqrt(2.0 + gw)
                phi = phi_rand[itraj, s] * 2.0 * pi
                zF[s] = rF * (np.cos(phi) + 1j * np.sin(phi))

            Hij = hel_fn(R, c, epsilon, Delta)
            dHij = dhel_fn(R, c)
            dH0 = dhel0_fn(R, omega)

            iskip = 0
            for step in range(NSteps):
                if step % nskip == 0:
                    rho = pop_spinlsc(zF, gw, NStates)
                    for a in range(NStates):
                        for b in range(NStates):
                            rho_all[itraj, a, b, iskip] = rho[a, b]
                    iskip += 1

                # --- VelVer ---
                zF_w = zF.copy()
                # half mapping
                for _t in range(half_e_floor):
                    zF_w = umap_spinlsc(zF_w, dtE, Hij, NStates)
                zF = zF_w.copy()

                # nuclear
                F1 = force_spinlsc(zF, gw, dHij, dH0, NStates, ndof)
                v = np.empty(ndof)
                for d in range(ndof):
                    v[d] = P[d] / M_mass
                    R[d] += v[d] * dtN + 0.5 * F1[d] * dtN * dtN / M_mass

                Hij = hel_fn(R, c, epsilon, Delta)
                dHij = dhel_fn(R, c)
                dH0 = dhel0_fn(R, omega)
                F2 = force_spinlsc(zF, gw, dHij, dH0, NStates, ndof)
                for d in range(ndof):
                    v[d] += 0.5 * (F1[d] + F2[d]) * dtN / M_mass
                    P[d] = v[d] * M_mass

                # second half mapping
                Hij = hel_fn(R, c, epsilon, Delta)
                for _t in range(half_e_ceil):
                    zF_w = umap_spinlsc(zF_w, dtE, Hij, NStates)
                zF = zF_w.copy()

        rho_ensemble = np.zeros((NStates, NStates, n_skip), dtype=np.complex128)
        for itraj in range(NTraj):
            for a in range(NStates):
                for b in range(NStates):
                    for t in range(n_skip):
                        rho_ensemble[a, b, t] += rho_all[itraj, a, b, t]
        return rho_ensemble
    _kernel_cache[key] = run_traj
    return run_traj
