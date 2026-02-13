"""
JIT-compiled FS-MASH (Fewest-Switches MASH) method.

Reference: Method/fsmash.py

Key differences from MethodJIT/mash.py:
  - Uses midpoint scheme (dt/2) instead of bisection for hop timing
  - Tully's fewest-switches stochastic probability for hop acceptance
  - Hop decision uses a pre-generated random pool (one per timestep)
  - Same MASH-style force, pop estimator, and check_hop

RNG note:
  The original fsmash uses random.random() for phase init (Python RNG)
  and np.random.uniform() inside the hop function (numpy RNG) during
  the simulation.  The JIT version pre-generates:
    - phase_rand[NTraj, NStates]   from random.random() (Python RNG)
    - hop_rand[NTraj, NSteps]      from np.random.uniform() (numpy RNG)
  consumed sequentially per trajectory for bitwise reproducibility.
"""
import numpy as np
from numba import njit, prange

_jit = dict(cache=True, fastmath=True, boundscheck=False, error_model='numpy')


# ---------- helper: force (same as MASH) ------------------------------------

@njit(**_jit)
def force_fsmash(dHij, dH0, acst, U, NStates, ndof):
    """F = -dH0 - Re( sum_ij conj(U[i,acst]) * dHij[i,j,d] * U[j,acst] )"""
    F = np.empty(ndof)
    for d in range(ndof):
        F[d] = -dH0[d]
        s = 0.0 + 0.0j
        for i in range(NStates):
            for j in range(NStates):
                s += U[i, acst].conjugate() * dHij[i, j, d] * U[j, acst]
        F[d] -= s.real
    return F


# ---------- helper: population estimator (same as MASH) --------------------

@njit(**_jit)
def pop_fsmash(ci, U, NStates, alpha, beta):
    """rho = alpha * outer(cD, cD*) + beta * I  where cD = U @ ci."""
    cD = np.empty(NStates, dtype=np.complex128)
    for i in range(NStates):
        s = 0.0 + 0.0j
        for j in range(NStates):
            s += U[i, j] * ci[j]
        cD[i] = s
    rho = np.empty((NStates, NStates), dtype=np.complex128)
    for i in range(NStates):
        for j in range(NStates):
            rho[i, j] = alpha * cD[i] * cD[j].conjugate()
            if i == j:
                rho[i, j] += beta
    return rho


# ---------- helper: check hop (same as MASH) --------------------------------

@njit(**_jit)
def check_hop_fsmash(acst, ci, NStates):
    """Return (hop_needed, new_active_state)."""
    max_idx = 0
    max_val = (ci[0] * ci[0].conjugate()).real
    for s in range(1, NStates):
        v = (ci[s] * ci[s].conjugate()).real
        if v > max_val:
            max_val = v
            max_idx = s
    return (acst != max_idx), max_idx


# ---------- helper: VelVer (split-operator, matching fsmash original) --------

@njit(**_jit)
def velver_fsmash(R, P, ci, E, U, F1, acst, dt, M_mass, NStates, ndof,
                  hel_fn, dhel_fn, dhel0_fn,
                  c_model, epsilon, Delta, omega):
    """
    Velocity-Verlet with split-operator electronic evolution.

    Order: exp(-iE dt/2) -> to diabatic -> nuclear -> to adiabatic -> exp(-iE dt/2)

    Matches the original fsmash VelVer which uses:
      v += 0.5*F1*dt/M  THEN  R += v*dt  (leapfrog-style velocity kick first)
    This is mathematically equivalent to the MASH VelVer but uses
    np.linalg.eigh(Hij) (real eigh, matching the original).

    Returns (R, P, ci, Hij, dHij, dH0, E_new, U_new, F2).
    """
    # Half electronic evolution
    for s in range(NStates):
        ci[s] = ci[s] * np.exp(-1j * E[s] * dt * 0.5)

    # To diabatic: cD = U @ ci
    cD = np.empty(NStates, dtype=np.complex128)
    for i in range(NStates):
        s = 0.0 + 0.0j
        for j in range(NStates):
            s += U[i, j] * ci[j]
        cD[i] = s

    # Nuclear step (leapfrog style, matching original)
    v = np.empty(ndof)
    for d in range(ndof):
        v[d] = P[d] / M_mass
    for d in range(ndof):
        v[d] += 0.5 * F1[d] * dt / M_mass
        R[d] += v[d] * dt

    # Recompute QM at new R
    Hij = hel_fn(R, c_model, epsilon, Delta)
    dHij = dhel_fn(R, c_model)
    dH0 = dhel0_fn(R, omega)

    E_new, U_new = np.linalg.eigh(Hij + 0j)

    F2 = force_fsmash(dHij, dH0, acst, U_new, NStates, ndof)
    for d in range(ndof):
        v[d] += 0.5 * F2[d] * dt / M_mass
        P[d] = v[d] * M_mass

    # Back to adiabatic: ci = conj(U_new)^T @ cD
    ci_new = np.empty(NStates, dtype=np.complex128)
    for i in range(NStates):
        s = 0.0 + 0.0j
        for j in range(NStates):
            s += U_new[j, i].conjugate() * cD[j]
        ci_new[i] = s

    # Half electronic evolution
    for s in range(NStates):
        ci_new[s] = ci_new[s] * np.exp(-1j * E_new[s] * dt * 0.5)

    return R, P, ci_new, Hij, dHij, dH0, E_new, U_new, F2


# ---------- helper: fsmash hop with Tully probability -----------------------

@njit(**_jit)
def attempt_hop_fsmash(ci, P, E, U, dHij, acst, newacst, dtN,
                       M_mass, NStates, ndof, hop_random):
    """
    Attempt surface hop from acst to newacst.

    1. Compute hop direction dk (same as MASH).
    2. Energy check: if insufficient, reverse projected P and reject.
    3. Tully probability: if rand < plz, accept; else reverse and reject.

    Parameters
    ----------
    hop_random : float -- pre-generated uniform [0,1) for Tully decision

    Returns (P_new, accepted).
    """
    if acst == newacst:
        return P.copy(), False

    a, b = acst, newacst
    dE = E[b] - E[a]
    sqrtM = np.sqrt(M_mass)

    # Reciprocal energy gaps
    rDEa = np.zeros(NStates)
    rDEb = np.zeros(NStates)
    for n in range(NStates):
        if n != a:
            rDEa[n] = 1.0 / (E[a] - E[n])
        if n != b:
            rDEb[n] = 1.0 / (E[b] - E[n])

    # dk = Re(term1 - term2) / sqrt(M)
    # dHab[n, state, d] = sum_{i,j} conj(U[i,n]) * dHij[i,j,d] * U[j,state]
    dk = np.zeros(ndof)
    # Also need dHab[b, a, :] for Tully probability
    dHab_ba = np.zeros(ndof, dtype=np.complex128)

    for d in range(ndof):
        t1 = 0.0 + 0.0j
        t2 = 0.0 + 0.0j
        # Also compute dHab[b,a,d]
        dH_ba_d = 0.0 + 0.0j
        for n in range(NStates):
            dHna = 0.0 + 0.0j
            dHnb = 0.0 + 0.0j
            for i in range(NStates):
                for j in range(NStates):
                    val = U[i, n].conjugate() * dHij[i, j, d]
                    dHna += val * U[j, a]
                    dHnb += val * U[j, b]
            t1 += ci[n].conjugate() * dHna * ci[a] * rDEa[n]
            t2 += ci[n].conjugate() * dHnb * ci[b] * rDEb[n]
            if n == b:
                dH_ba_d = dHna  # dHab[b, a, d] = sum conj(U[i,b])*dH*U[j,a]
        dk[d] = (t1 - t2).real / sqrtM
        dHab_ba[d] = dH_ba_d

    # Project momentum
    dk_dot = 0.0
    P_dk = 0.0
    for d in range(ndof):
        dk_dot += dk[d] * dk[d]
        P_dk += (P[d] / sqrtM) * dk[d]

    if dk_dot < 1e-30:
        return P.copy(), False

    P_proj_norm2 = P_dk * P_dk / dk_dot

    if P_proj_norm2 < 2.0 * dE:
        # Insufficient energy -- reject, reverse projected P
        P_new = np.empty(ndof)
        for d in range(ndof):
            proj = P_dk * dk[d] / dk_dot
            orth = P[d] / sqrtM - proj
            P_new[d] = (orth - proj) * sqrtM
        return P_new, False

    # Energy conserved -- compute Tully probability
    P_proj_rescaled_norm = np.sqrt(P_proj_norm2 - 2.0 * dE)

    # plz = max(0, P_old . plz_vec)
    # plz_vec = dtN * conj(c[b]) * c[a] * dHab[b,a,:] / (pop_a * dE)
    pop_a = (ci[a].conjugate() * ci[a]).real
    if pop_a < 1e-30:
        pop_a = 1e-30  # avoid division by zero

    cb_ca = ci[b].conjugate() * ci[a]

    plz = 0.0
    for d in range(ndof):
        plz_vec_d = dtN * cb_ca * dHab_ba[d] / (pop_a * dE)
        plz += (P[d] / sqrtM) * plz_vec_d.real
    plz = max(0.0, plz)

    if hop_random < plz:
        # Accept hop -- rescale projected component
        scale = P_proj_rescaled_norm / np.sqrt(P_proj_norm2)
        P_new = np.empty(ndof)
        for d in range(ndof):
            proj = P_dk * dk[d] / dk_dot
            orth = P[d] / sqrtM - proj
            P_new[d] = (orth + scale * proj) * sqrtM
        return P_new, True
    else:
        # Reject hop -- reverse projected component
        P_new = np.empty(ndof)
        for d in range(ndof):
            proj = P_dk * dk[d] / dk_dot
            orth = P[d] / sqrtM - proj
            P_new[d] = (orth - proj) * sqrtM
        return P_new, False


# ---------- kernel factory --------------------------------------------------

_kernel_cache = {}


def make_fsmash_kernel(hel_fn, dhel_fn, dhel0_fn):
    """
    Factory returning a parallel FS-MASH kernel.

    Uses midpoint scheme (dt/2) for hop timing and Tully's
    fewest-switches probability for hop acceptance.

    Results are memoised.
    """
    key = (hel_fn, dhel_fn, dhel0_fn)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @njit(parallel=True, **_jit)
    def run_traj(NTraj, NSteps, NStates, nskip,
                 dtN, M_mass, initState,
                 R_rand, P_rand, sigR, sigP,
                 phase_rand,       # (NTraj, NStates) -- uniform for phases
                 hop_rand,         # (NTraj, NSteps)  -- uniform for Tully
                 c_model, epsilon, Delta, omega):

        n_skip = NSteps // nskip
        ndof = len(sigR)

        # MASH constants
        sumN = 0.0
        for n in range(1, NStates + 1):
            sumN += 1.0 / n
        alpha = (NStates - 1.0) / (sumN - 1.0)
        beta_coeff = (alpha - 1.0) / NStates
        beta_pop = (1.0 - alpha) / NStates

        rho_all = np.zeros((NTraj, NStates, NStates, n_skip),
                           dtype=np.complex128)

        for itraj in prange(NTraj):
            # --- initialise nuclear DOF ---
            R = np.empty(ndof)
            P = np.empty(ndof)
            for d in range(ndof):
                R[d] = R_rand[itraj, d] * sigR[d]
                P[d] = P_rand[itraj, d] * sigP[d]

            # --- initElectronic (same MASH-style) ---
            ci = np.empty(NStates, dtype=np.complex128)
            for s in range(NStates):
                ci[s] = np.sqrt(beta_coeff / alpha) + 0.0j
            ci[initState] = np.sqrt((1.0 + beta_coeff) / alpha) + 0.0j
            for s in range(NStates):
                phi = phase_rand[itraj, s] * 2.0 * np.pi
                ci[s] = ci[s] * (np.cos(phi) + 1j * np.sin(phi))

            # Initial QM
            Hij = hel_fn(R, c_model, epsilon, Delta)
            dHij = dhel_fn(R, c_model)
            dH0 = dhel0_fn(R, omega)
            E, U = np.linalg.eigh(Hij + 0j)

            # Transform to adiabatic basis
            ci_a = np.empty(NStates, dtype=np.complex128)
            for i in range(NStates):
                s = 0.0 + 0.0j
                for j in range(NStates):
                    s += U[j, i].conjugate() * ci[j]
                ci_a[i] = s
            ci = ci_a

            # Initial active state
            _, acst = check_hop_fsmash(-1, ci, NStates)

            F1 = force_fsmash(dHij, dH0, acst, U, NStates, ndof)

            # --- time propagation ---
            iskip = 0
            hop_idx = 0  # index into hop_rand pool

            for step in range(NSteps):
                # --- estimator ---
                if step % nskip == 0:
                    rho = pop_fsmash(ci, U, NStates, alpha, beta_pop)
                    for aa in range(NStates):
                        for bb in range(NStates):
                            rho_all[itraj, aa, bb, iskip] = rho[aa, bb]
                    iskip += 1

                # Save state for possible hop
                R0 = R.copy()
                P0 = P.copy()
                ci0 = ci.copy()
                E0 = E.copy()
                U0 = U.copy()
                F1_0 = F1.copy()
                acst0 = acst

                # Full VelVer step
                R_new = R.copy()
                P_new = P.copy()
                ci_new = ci.copy()
                (R_new, P_new, ci_new, Hij_new, dHij_new, dH0_new,
                 E_new, U_new, F2_new) = velver_fsmash(
                    R_new, P_new, ci_new, E.copy(), U.copy(), F1.copy(),
                    acst, dtN, M_mass, NStates, ndof,
                    hel_fn, dhel_fn, dhel0_fn,
                    c_model, epsilon, Delta, omega)

                # Check hop
                hop_needed, new_acst = check_hop_fsmash(
                    acst, ci_new, NStates)

                if hop_needed:
                    # Midpoint scheme: restore and propagate to dt/2
                    tm = dtN * 0.5
                    Rm = R0.copy()
                    Pm = P0.copy()
                    cim = ci0.copy()
                    (Rm, Pm, cim, Hijm, dHijm, dH0m,
                     Em, Um, F2m) = velver_fsmash(
                        Rm, Pm, cim, E0.copy(), U0.copy(), F1_0.copy(),
                        acst, tm, M_mass, NStates, ndof,
                        hel_fn, dhel_fn, dhel0_fn,
                        c_model, epsilon, Delta, omega)

                    # Find candidate: state with largest |c|^2 (excl. acst)
                    best_b = -1
                    best_val = -1.0
                    for ss in range(NStates):
                        if ss == acst:
                            continue
                        val = (cim[ss] * cim[ss].conjugate()).real
                        if val > best_val:
                            best_val = val
                            best_b = ss

                    # Attempt hop with Tully probability
                    hr = hop_rand[itraj, hop_idx]
                    hop_idx += 1
                    P_hop, accepted = attempt_hop_fsmash(
                        cim, Pm, Em, Um, dHijm,
                        acst, best_b, dtN, M_mass, NStates, ndof, hr)

                    # Always apply returned momentum
                    Pm[:] = P_hop
                    if accepted:
                        acst = best_b
                        F2m = force_fsmash(dHijm, dH0m, acst, Um,
                                           NStates, ndof)

                    # Finish timestep from midpoint
                    (Rm, Pm, cim, _Hf, dHf, dH0f,
                     Ef, Uf, F2f) = velver_fsmash(
                        Rm, Pm, cim, Em, Um, F2m, acst,
                        dtN - tm, M_mass, NStates, ndof,
                        hel_fn, dhel_fn, dhel0_fn,
                        c_model, epsilon, Delta, omega)

                    R[:] = Rm
                    P[:] = Pm
                    ci = cim
                    E = Ef
                    U = Uf
                    F1 = F2f
                    dHij = dHf
                    dH0 = dH0f
                else:
                    # No hop -- keep full-step result
                    R[:] = R_new
                    P[:] = P_new
                    ci = ci_new
                    E = E_new
                    U = U_new
                    F1 = F2_new
                    dHij = dHij_new
                    dH0 = dH0_new

        # --- reduce across trajectories ---
        rho_ensemble = np.zeros((NStates, NStates, n_skip),
                                dtype=np.complex128)
        for itraj in range(NTraj):
            for aa in range(NStates):
                for bb in range(NStates):
                    for t in range(n_skip):
                        rho_ensemble[aa, bb, t] += rho_all[itraj, aa, bb, t]
        return rho_ensemble

    _kernel_cache[key] = run_traj
    return run_traj
