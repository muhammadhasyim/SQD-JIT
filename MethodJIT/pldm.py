"""
JIT-compiled Partial Linearized Density Matrix (PLDM) method.

All functions are pure @njit with explicit array arguments --
no Bunch class, no globals, no dynamic dispatch.

The trajectory ensemble is parallelised with numba.prange.
A factory function ``make_pldm_kernel`` captures model-specific
@njit Hamiltonian functions via closure, returning a fully
compiled ``run_traj`` kernel.

References
----------
Original: Method/pldm.py
"""

import numpy as np
from numba import njit, prange


# --------------- decorator shorthand ----------------------------------------
_jit = dict(cache=True, fastmath=True, boundscheck=False, error_model='numpy')
# ----------------------------------------------------------------------------


# ---- Mapping-variable propagator -------------------------------------------

@njit(**_jit)
def umap(qF, qB, pF, pB, dt, VMat):
    """
    Second-order symplectic propagator for the PLDM mapping variables.

    Operates **in-place** on qF, qB, pF, pB to avoid heap allocation.
    """
    NStates = len(qF)

    # --- store input copies (on-stack for small NStates) --------------------
    qFin = qF.copy()
    qBin = qB.copy()
    pFin = pF.copy()
    pBin = pB.copy()

    # VMat @ q_in  (manual loop avoids tiny-array dispatch overhead)
    VxqB = np.zeros(NStates)
    VxqF = np.zeros(NStates)
    for k in range(NStates):
        s_b = 0.0
        s_f = 0.0
        for l in range(NStates):
            s_b += VMat[k, l] * qBin[l]
            s_f += VMat[k, l] * qFin[l]
        VxqB[k] = s_b
        VxqF[k] = s_f

    # momentum half-kick  (first-order from input positions)
    for k in range(NStates):
        pB[k] -= 0.5 * dt * VxqB[k]
        pF[k] -= 0.5 * dt * VxqF[k]

    # position update  (first-order from input momenta)
    for k in range(NStates):
        acc_b = 0.0
        acc_f = 0.0
        for l in range(NStates):
            acc_b += VMat[k, l] * pBin[l]
            acc_f += VMat[k, l] * pFin[l]
        qB[k] += dt * acc_b
        qF[k] += dt * acc_f

    # position second-order correction
    dt2h = dt * dt * 0.5
    for k in range(NStates):
        acc_b = 0.0
        acc_f = 0.0
        for l in range(NStates):
            acc_b += VMat[k, l] * VxqB[l]
            acc_f += VMat[k, l] * VxqF[l]
        qB[k] -= dt2h * acc_b
        qF[k] -= dt2h * acc_f

    # momentum half-kick  (first-order from *output* positions)
    for k in range(NStates):
        acc_b = 0.0
        acc_f = 0.0
        for l in range(NStates):
            acc_b += VMat[k, l] * qB[l]
            acc_f += VMat[k, l] * qF[l]
        pB[k] -= 0.5 * dt * acc_b
        pF[k] -= 0.5 * dt * acc_f


# ---- Force -----------------------------------------------------------------

@njit(**_jit)
def force_pldm(qF, pF, qB, pB, dHij, dH0):
    """
    Nuclear force for PLDM.

    F = -dH0 - 0.25 * sum_i dH[i,i,:] * (qF_i^2 + pF_i^2 + qB_i^2 + pB_i^2)
               - 0.5 * sum_{i<j} dH[i,j,:] * (qFi*qFj + pFi*pFj + qBi*qBj + pBi*pBj)
    """
    ndof = len(dH0)
    NStates = len(qF)
    F = np.empty(ndof)
    for d in range(ndof):
        F[d] = -dH0[d]

    for i in range(NStates):
        qi2 = qF[i] * qF[i] + pF[i] * pF[i] + qB[i] * qB[i] + pB[i] * pB[i]
        for d in range(ndof):
            F[d] -= 0.25 * dHij[i, i, d] * qi2
        for j in range(i + 1, NStates):
            cross = (qF[i] * qF[j] + pF[i] * pF[j]
                     + qB[i] * qB[j] + pB[i] * pB[j])
            for d in range(ndof):
                F[d] -= 0.5 * dHij[i, j, d] * cross
    return F


# ---- Population estimator --------------------------------------------------

@njit(**_jit)
def pop_pldm(qF, pF, qB, pB, rho0, NStates):
    """
    Density-matrix estimator: rho = outer(qF + i*pF, qB - i*pB) * rho0.
    """
    rho = np.empty((NStates, NStates), dtype=np.complex128)
    for a in range(NStates):
        za = qF[a] + 1j * pF[a]
        for b in range(NStates):
            zb = qB[b] - 1j * pB[b]
            rho[a, b] = za * zb * rho0
    return rho


# ---- Factory: creates a model-specialised JIT kernel -----------------------

_kernel_cache = {}

def make_pldm_kernel(hel_fn, dhel_fn, dhel0_fn):
    """
    Factory that returns a JIT-compiled ``run_traj`` function.

    Results are memoized: calling with the same functions returns the
    already-compiled kernel without recompilation.

    Parameters
    ----------
    hel_fn   : @njit function  R -> (NStates, NStates) Hamiltonian
    dhel_fn  : @njit function  R -> (NStates, NStates, ndof) gradient
    dhel0_fn : @njit function  R -> (ndof,) state-independent gradient

    Returns
    -------
    run_traj : @njit(parallel=True) function
    """
    key = (hel_fn, dhel_fn, dhel0_fn)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @njit(parallel=True, **_jit)
    def run_traj(NTraj, NSteps, NStates, nskip, EStep, dtN, M_mass,
                 initState, stype_focused,
                 R_rand, P_rand, sigR, sigP,
                 map_rand,  # (NTraj, 4, NStates) for sampled mode, unused for focused
                 c, epsilon, Delta, omega):
        """
        Run an ensemble of PLDM trajectories.

        Random numbers are pre-generated for reproducibility.
        Each trajectory writes to its own slice of rho_all (no race).
        """
        n_skip = NSteps // nskip
        dtE = dtN / EStep
        half_e_floor = EStep // 2
        half_e_ceil = (EStep + 1) // 2

        # Per-trajectory storage (no race -- each itraj owns its slice)
        rho_all = np.zeros((NTraj, NStates, NStates, n_skip),
                           dtype=np.complex128)

        for itraj in prange(NTraj):
            # ---- initialise nuclear DOF from pre-generated randoms ----
            ndof = len(sigR)
            R = np.empty(ndof)
            P = np.empty(ndof)
            for d in range(ndof):
                R[d] = R_rand[itraj, d] * sigR[d]
                P[d] = P_rand[itraj, d] * sigP[d]

            # ---- initialise mapping variables -------------------------
            qF = np.zeros(NStates)
            qB = np.zeros(NStates)
            pF = np.zeros(NStates)
            pB = np.zeros(NStates)

            if stype_focused:
                qF[initState] = 1.0
                qB[initState] = 1.0
                pF[initState] = 1.0
                pB[initState] = -1.0
            else:
                for s in range(NStates):
                    qF[s] = map_rand[itraj, 0, s]
                    qB[s] = map_rand[itraj, 1, s]
                    pF[s] = map_rand[itraj, 2, s]
                    pB[s] = map_rand[itraj, 3, s]

            # initial rho0 factor
            rho0 = 0.25 * ((qF[initState] - 1j * pF[initState])
                           * (qB[initState] + 1j * pB[initState]))

            # ---- initial QM -------------------------------------------
            Hij  = hel_fn(R, c, epsilon, Delta)
            dHij = dhel_fn(R, c)
            dH0  = dhel0_fn(R, omega)

            # ---- time propagation -------------------------------------
            iskip = 0
            for step in range(NSteps):
                # -- estimator --
                if step % nskip == 0:
                    rho = pop_pldm(qF, pF, qB, pB, rho0, NStates)
                    for a in range(NStates):
                        for b in range(NStates):
                            rho_all[itraj, a, b, iskip] = rho[a, b]
                    iskip += 1

                # -- VelVer integrator (inlined) -------------------------
                # half-step mapping
                for _t in range(half_e_floor):
                    umap(qF, qB, pF, pB, dtE, Hij)

                # nuclear half: force -> position update -> QM -> force -> velocity
                v = np.empty(ndof)
                for d in range(ndof):
                    v[d] = P[d] / M_mass

                F1 = force_pldm(qF, pF, qB, pB, dHij, dH0)

                for d in range(ndof):
                    R[d] += v[d] * dtN + 0.5 * F1[d] * dtN * dtN / M_mass

                # recompute QM at new R
                Hij  = hel_fn(R, c, epsilon, Delta)
                dHij = dhel_fn(R, c)
                dH0  = dhel0_fn(R, omega)

                F2 = force_pldm(qF, pF, qB, pB, dHij, dH0)

                for d in range(ndof):
                    v[d] += 0.5 * (F1[d] + F2[d]) * dtN / M_mass
                    P[d] = v[d] * M_mass

                # second half-step mapping (recompute Hij in case needed)
                Hij = hel_fn(R, c, epsilon, Delta)
                for _t in range(half_e_ceil):
                    umap(qF, qB, pF, pB, dtE, Hij)

        # ---- reduce across trajectories --------------------------------
        rho_ensemble = np.zeros((NStates, NStates, n_skip),
                                dtype=np.complex128)
        for itraj in range(NTraj):
            for a in range(NStates):
                for b in range(NStates):
                    for t in range(n_skip):
                        rho_ensemble[a, b, t] += rho_all[itraj, a, b, t]

        return rho_ensemble

    _kernel_cache[key] = run_traj
    return run_traj


# ---- Convenience wrapper (mirrors original API) ----------------------------

def runTraj(parameters):
    """
    Drop-in replacement for Method.pldm.runTraj(parameters).

    Imports model JIT functions, builds the kernel, pre-generates
    random numbers, and runs the ensemble.
    """
    from ModelJIT.spinBoson import hel, dhel, dhel0, get_model_params

    # Build JIT kernel (compiled once, cached to disk)
    kernel = make_pldm_kernel(hel, dhel, dhel0)

    # Extract parameters
    NTraj     = parameters.NTraj
    NSteps    = parameters.NSteps
    NStates   = parameters.NStates
    initState = parameters.initState
    nskip     = parameters.nskip
    dtN       = parameters.dtN
    EStep     = int(dtN / parameters.dtE)
    M_mass    = float(parameters.M) if np.isscalar(parameters.M) else float(parameters.M[0])

    stype     = getattr(parameters, 'stype', 'focused')
    stype_focused = (stype == 'focused' or stype == '_')

    ndof = parameters.ndof
    c = parameters.c
    omega = parameters.omega if hasattr(parameters, 'omega') else parameters.ω
    epsilon = parameters.epsilon if hasattr(parameters, 'epsilon') else parameters.ε
    Delta = parameters.Delta if hasattr(parameters, 'Delta') else parameters.Δ
    beta = parameters.beta if hasattr(parameters, 'beta') else parameters.β

    # Thermal widths
    sigP = np.sqrt(omega / (2.0 * np.tanh(0.5 * beta * omega)))
    sigR = sigP / omega

    # Pre-generate random numbers (reproducible)
    seed = getattr(parameters, 'SEED', None)
    if seed is not None:
        np.random.seed(seed)

    R_rand = np.random.randn(NTraj, ndof)
    P_rand = np.random.randn(NTraj, ndof)

    # mapping randoms (only used for sampled mode)
    if not stype_focused:
        map_rand = np.random.randn(NTraj, 4, NStates)
    else:
        map_rand = np.empty((NTraj, 4, NStates))  # unused placeholder

    rho = kernel(NTraj, NSteps, NStates, nskip, EStep, dtN, M_mass,
                 initState, stype_focused,
                 R_rand, P_rand, sigR, sigP,
                 map_rand,
                 c, epsilon, Delta, omega)
    return rho
