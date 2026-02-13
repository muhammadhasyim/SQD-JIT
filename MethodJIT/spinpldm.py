"""
JIT-compiled spin-PLDM (Partially Linearized Density Matrix) method.
Reference: Method/spinpldm.py
Uses complex zF/zB, Ugam via eigh, forward-backward combinations.
"""
import numpy as np
from numba import njit, prange

_jit = dict(cache=True, fastmath=True, boundscheck=False, error_model='numpy')


@njit(**_jit)
def umap_spin(z, dt, VMat, NStates):
    """Split-step symplectic propagator for complex mapping variable."""
    Zreal = np.empty(NStates)
    Zimag = np.empty(NStates)
    for s in range(NStates):
        Zreal[s] = z[s].real
        Zimag[s] = z[s].imag

    for k in range(NStates):
        s = 0.0
        for l in range(NStates):
            s += VMat[k, l] * Zreal[l]
        Zimag[k] -= 0.5 * s * dt

    for k in range(NStates):
        s = 0.0
        for l in range(NStates):
            s += VMat[k, l] * Zimag[l]
        Zreal[k] += s * dt

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
def force_spinpldm(zF, zB, gw, dHij, dH0, NStates, ndof):
    """Force using combined forward-backward density matrix."""
    F = np.empty(ndof)
    for d in range(ndof):
        F[d] = -dH0[d]
    for i in range(NStates):
        eta_ii = 0.5 * ((zF[i].conjugate() * zF[i]).real +
                         (zB[i].conjugate() * zB[i]).real - 2.0 * gw)
        for d in range(ndof):
            F[d] -= 0.5 * dHij[i, i, d] * eta_ii
        for j in range(i + 1, NStates):
            eta_ij = 0.5 * ((zF[i].conjugate() * zF[j]).real +
                             (zB[i].conjugate() * zB[j]).real)
            for d in range(ndof):
                F[d] -= 2.0 * 0.5 * dHij[i, j, d] * eta_ij
    return F


@njit(**_jit)
def matmul_complex(A, B, N):
    """Complex matrix multiply C = A @ B for NxN matrices."""
    C = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            s = 0.0 + 0.0j
            for k in range(N):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


_kernel_cache = {}

def make_spinpldm_kernel(hel_fn, dhel_fn, dhel0_fn):
    key = (hel_fn, dhel_fn, dhel0_fn)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @njit(parallel=True, **_jit)
    def run_traj(NTraj, NSteps, NStates, nskip, EStep, dtN, M_mass,
                 initState,
                 R_rand, P_rand, sigR, sigP,
                 phi_rand,     # (NTraj, nFB, 2, NStates)
                 FB_arr,       # (nFB, 2) int array
                 W_arr,        # (nFB,) float weights
                 c, epsilon, Delta, omega):
        n_skip = NSteps // nskip
        dtE = dtN / EStep
        half_e_floor = EStep // 2
        half_e_ceil = (EStep + 1) // 2
        ndof = len(sigR)
        pi = np.pi
        nFB = len(W_arr)
        gw = (2.0 / NStates) * (np.sqrt(NStates + 1.0) - 1.0)
        total_work = NTraj * nFB
        rho_all = np.zeros((total_work, NStates, NStates, n_skip), dtype=np.complex128)

        for work_idx in prange(total_work):
            itraj = work_idx // nFB
            ifb = work_idx % nFB
            F_state = FB_arr[ifb, 0]
            B_state = FB_arr[ifb, 1]
            W = W_arr[ifb]

            R = np.empty(ndof)
            P = np.empty(ndof)
            for d in range(ndof):
                R[d] = R_rand[itraj, d] * sigR[d]
                P[d] = P_rand[itraj, d] * sigP[d]

            # init mapping zF
            zF = np.empty(NStates, dtype=np.complex128)
            for s in range(NStates):
                rF = np.sqrt(gw)
                if s == F_state:
                    rF = np.sqrt(2.0 + gw)
                phi = phi_rand[itraj, ifb, 0, s] * 2.0 * pi
                zF[s] = rF * (np.cos(phi) + 1j * np.sin(phi))
            # init mapping zB
            zB = np.empty(NStates, dtype=np.complex128)
            for s in range(NStates):
                rB = np.sqrt(gw)
                if s == B_state:
                    rB = np.sqrt(2.0 + gw)
                phi = phi_rand[itraj, ifb, 1, s] * 2.0 * pi
                zB[s] = rB * (np.cos(phi) + 1j * np.sin(phi))

            zF0 = zF[initState]
            zB0 = zB[initState]

            # Ugam = identity
            Ugam = np.zeros((NStates, NStates), dtype=np.complex128)
            for s in range(NStates):
                Ugam[s, s] = 1.0 + 0.0j

            Hij = hel_fn(R, c, epsilon, Delta)
            dHij = dhel_fn(R, c)
            dH0 = dhel0_fn(R, omega)

            iskip = 0
            for step in range(NSteps):
                if step % nskip == 0:
                    # pop: rho = 0.25 * outer(rhoF, rhoB)
                    gamma_col = np.empty(NStates, dtype=np.complex128)
                    for s in range(NStates):
                        gamma_col[s] = gw * Ugam[s, initState]
                    rhoF = np.empty(NStates, dtype=np.complex128)
                    rhoB = np.empty(NStates, dtype=np.complex128)
                    for s in range(NStates):
                        rhoF[s] = zB[s].conjugate() * zB0 - gamma_col[s].conjugate()
                        rhoB[s] = zF[s] * zF0.conjugate() - gamma_col[s]
                    for a in range(NStates):
                        for b in range(NStates):
                            rho_all[work_idx, a, b, iskip] = 0.25 * W * rhoF[a] * rhoB[b]
                    iskip += 1

                # --- VelVer ---
                zF_w = zF.copy()
                zB_w = zB.copy()
                # half mapping
                for _t in range(half_e_floor):
                    zF_w = umap_spin(zF_w, dtE, Hij, NStates)
                    zB_w = umap_spin(zB_w, dtE, Hij, NStates)
                zF = zF_w.copy()
                zB = zB_w.copy()

                # nuclear
                F1 = force_spinpldm(zF, zB, gw, dHij, dH0, NStates, ndof)
                v = np.empty(ndof)
                for d in range(ndof):
                    v[d] = P[d] / M_mass
                    R[d] += v[d] * dtN + 0.5 * F1[d] * dtN * dtN / M_mass

                Hij = hel_fn(R, c, epsilon, Delta)
                dHij = dhel_fn(R, c)
                dH0 = dhel0_fn(R, omega)
                F2 = force_spinpldm(zF, zB, gw, dHij, dH0, NStates, ndof)
                for d in range(ndof):
                    v[d] += 0.5 * (F1[d] + F2[d]) * dtN / M_mass
                    P[d] = v[d] * M_mass

                # second half mapping
                Hij = hel_fn(R, c, epsilon, Delta)
                for _t in range(half_e_ceil):
                    zF_w = umap_spin(zF_w, dtE, Hij, NStates)
                    zB_w = umap_spin(zB_w, dtE, Hij, NStates)
                zF = zF_w.copy()
                zB = zB_w.copy()

                # Ugam propagation via eigh
                E = np.linalg.eigvalsh(Hij)
                E_full, U = np.linalg.eigh(Hij)
                # Udt = U @ diag(exp(-i*E*dt)) @ U.T
                Udt = np.zeros((NStates, NStates), dtype=np.complex128)
                for a in range(NStates):
                    for b in range(NStates):
                        s = 0.0 + 0.0j
                        for k in range(NStates):
                            s += U[a, k] * np.exp(-1j * E_full[k] * dtN) * U[b, k]
                        Udt[a, b] = s
                Ugam = matmul_complex(Udt, Ugam, NStates)

        rho_ensemble = np.zeros((NStates, NStates, n_skip), dtype=np.complex128)
        for widx in range(total_work):
            for a in range(NStates):
                for b in range(NStates):
                    for t in range(n_skip):
                        rho_ensemble[a, b, t] += rho_all[widx, a, b, t]
        return rho_ensemble
    _kernel_cache[key] = run_traj
    return run_traj
