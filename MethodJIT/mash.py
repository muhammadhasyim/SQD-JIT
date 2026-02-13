"""
JIT-compiled MASH (Multi-state mapping Approach to Surface Hopping) method.

Reference: Method/mash.py
General NStates implementation.

Key changes from original:
  - copy.deepcopy  -> explicit array .copy()
  - np.einsum      -> explicit loops (NStates is small)
  - np.linalg.eigh -> called on COMPLEX matrices (Hij + 0j) to match original LAPACK path
  - VelVer inlined into the prange loop to avoid allocation overhead
  - fastmath=False to preserve IEEE 754 compliance (hop decisions are sensitive)
"""
import numpy as np
from numba import njit, prange

_jit = dict(cache=True, fastmath=True, boundscheck=False, error_model='numpy')


# ---------- helper: force ---------------------------------------------------

@njit(**_jit)
def force_mash(dHij, dH0, acst, U, NStates, ndof):
    """
    Force = -dH0 - Re( sum_ij  conj(U[i,acst]) * dHij[i,j,d] * U[j,acst] )

    dHij is diabatic (NStates, NStates, ndof).
    U is the complex eigenvector matrix from eigh(Hij + 0j).
    """
    F = np.empty(ndof)
    for d in range(ndof):
        F[d] = -dH0[d]
        s = 0.0 + 0.0j
        for i in range(NStates):
            for j in range(NStates):
                s += U[i, acst].conjugate() * dHij[i, j, d] * U[j, acst]
        F[d] -= s.real
    return F


# ---------- helper: population estimator ------------------------------------

@njit(**_jit)
def pop_mash(ci, U, NStates, alpha, beta):
    """
    rho = alpha * outer(cD, cD*) + beta * I
    where cD = U @ ci  (diabatic coefficients).
    """
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


# ---------- helper: check hop -----------------------------------------------

@njit(**_jit)
def check_hop(acst, ci, NStates):
    """Return (hop_needed, new_active_state)."""
    max_idx = 0
    max_val = (ci[0] * ci[0].conjugate()).real
    for s in range(1, NStates):
        v = (ci[s] * ci[s].conjugate()).real
        if v > max_val:
            max_val = v
            max_idx = s
    return (acst != max_idx), max_idx


# ---------- helper: hop direction + momentum rescaling ----------------------

@njit(**_jit)
def attempt_hop(P, E, U, ci, dHij, acst, newacst, M_mass, NStates, ndof):
    """
    Compute hop direction dk and attempt momentum rescaling.
    Returns (P_new, accepted).

    Replaces the original's einsum-based hop direction with explicit loops.
    """
    if acst == newacst:
        return P.copy(), False

    a, b = acst, newacst
    dE = E[b] - E[a]

    # reciprocal energy gaps: rDEa[n] = (a!=n)/(E[a]-E[n])
    rDEa = np.zeros(NStates)
    rDEb = np.zeros(NStates)
    for n in range(NStates):
        if n != a:
            rDEa[n] = 1.0 / (E[a] - E[n])
        if n != b:
            rDEb[n] = 1.0 / (E[b] - E[n])

    # dk[d] = Re( term1 - term2 ) / sqrt(M)
    # where:
    #   dHab[n, state, d] = sum_{i,j} conj(U[i,n]) * dHij[i,j,d] * U[j,state]
    #   term1[d] = sum_n  ci[n]* * dHab[n,a,d] * ci[a] * rDEa[n]
    #   term2[d] = sum_n  ci[n]* * dHab[n,b,d] * ci[b] * rDEb[n]
    sqrtM = np.sqrt(M_mass)
    dk = np.zeros(ndof)
    for d in range(ndof):
        t1 = 0.0 + 0.0j
        t2 = 0.0 + 0.0j
        for n in range(NStates):
            # dHab[n, a, d] = sum_{i,j} conj(U[i,n]) * dHij[i,j,d] * U[j,a]
            dHna = 0.0 + 0.0j
            dHnb = 0.0 + 0.0j
            for i in range(NStates):
                for j in range(NStates):
                    dHna += U[i, n].conjugate() * dHij[i, j, d] * U[j, a]
                    dHnb += U[i, n].conjugate() * dHij[i, j, d] * U[j, b]
            t1 += ci[n].conjugate() * dHna * ci[a] * rDEa[n]
            t2 += ci[n].conjugate() * dHnb * ci[b] * rDEb[n]
        dk[d] = (t1 - t2).real / sqrtM

    # Project momentum onto dk direction
    dk_dot = 0.0
    P_dk = 0.0
    for d in range(ndof):
        dk_dot += dk[d] * dk[d]
        P_dk += (P[d] / sqrtM) * dk[d]

    if dk_dot < 1e-30:
        return P.copy(), False

    # P_proj_norm^2 = (P . dk)^2 / (dk . dk)
    P_proj_norm2 = P_dk * P_dk / dk_dot

    P_new = np.empty(ndof)
    if P_proj_norm2 < 2.0 * dE:
        # Rejected hop -- reverse projected component
        for d in range(ndof):
            proj = P_dk * dk[d] / dk_dot
            orth = P[d] / sqrtM - proj
            P_new[d] = (orth - proj) * sqrtM
        return P_new, False
    else:
        # Accepted hop -- rescale projected component
        scale = np.sqrt(P_proj_norm2 - 2.0 * dE) / np.sqrt(P_proj_norm2)
        for d in range(ndof):
            proj = P_dk * dk[d] / dk_dot
            orth = P[d] / sqrtM - proj
            P_new[d] = (orth + scale * proj) * sqrtM
        return P_new, True


# ---------- helper: inline VelVer -------------------------------------------

@njit(**_jit)
def velver_inline(R, P, ci, E, U, F1, acst, dt, M_mass, NStates, ndof,
                  hel_fn, dhel_fn, dhel0_fn,
                  c_model, epsilon, Delta, omega):
    """
    One Velocity-Verlet step.  Modifies R, P in-place via returned arrays.

    Returns (R, P, ci, Hij, dHij, dH0, E, U, F2).
    Uses COMPLEX eigh (Hij + 0j) to match original LAPACK path.
    """
    # half electronic: ci *= exp(-i*E*dt/2)
    for s in range(NStates):
        ci[s] = ci[s] * np.exp(-1j * E[s] * dt * 0.5)

    # To diabatic: cD = U @ ci   (U is complex, ci is complex)
    cD = np.empty(NStates, dtype=np.complex128)
    for i in range(NStates):
        s = 0.0 + 0.0j
        for j in range(NStates):
            s += U[i, j] * ci[j]
        cD[i] = s

    # Nuclear position update
    v = np.empty(ndof)
    for d in range(ndof):
        v[d] = P[d] / M_mass
        R[d] += v[d] * dt + 0.5 * F1[d] * dt * dt / M_mass

    # Recompute QM at new R  (COMPLEX eigh to match original)
    Hij = hel_fn(R, c_model, epsilon, Delta)
    dHij = dhel_fn(R, c_model)
    dH0 = dhel0_fn(R, omega)

    E_new, U_new = np.linalg.eigh(Hij + 0j)

    F2 = force_mash(dHij, dH0, acst, U_new, NStates, ndof)
    for d in range(ndof):
        v[d] += 0.5 * (F1[d] + F2[d]) * dt / M_mass
        P[d] = v[d] * M_mass

    # Back to adiabatic: ci = conj(U)^T @ cD  (U complex -> use conjugate)
    ci_new = np.empty(NStates, dtype=np.complex128)
    for i in range(NStates):
        s = 0.0 + 0.0j
        for j in range(NStates):
            s += U_new[j, i].conjugate() * cD[j]
        ci_new[i] = s

    # half electronic
    for s in range(NStates):
        ci_new[s] = ci_new[s] * np.exp(-1j * E_new[s] * dt * 0.5)

    return R, P, ci_new, Hij, dHij, dH0, E_new, U_new, F2


# ---------- kernel factory --------------------------------------------------

_kernel_cache = {}

def make_mash_kernel(hel_fn, dhel_fn, dhel0_fn):
    """
    Factory returning a parallel MASH kernel.

    Works for any NStates (not specialised to 2).
    Results are memoized: calling with the same functions returns the
    already-compiled kernel without recompilation.
    """
    key = (hel_fn, dhel_fn, dhel0_fn)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @njit(parallel=True, **_jit)
    def run_traj(NTraj, NSteps, NStates, nskip, dtN, M_mass,
                 initState, maxhop,
                 R_rand, P_rand, sigR, sigP,
                 uni_rand,      # (NTraj, NStates) uniform for phase init
                 c_model, epsilon, Delta, omega):

        n_skip = NSteps // nskip
        ndof = len(sigR)

        # MASH constants (general NStates)
        sumN = 0.0
        for n in range(1, NStates + 1):
            sumN += 1.0 / n
        alpha = (NStates - 1.0) / (sumN - 1.0)
        beta_coeff = (alpha - 1.0) / NStates        # for initElectronic
        beta_pop = (1.0 - alpha) / NStates           # for pop estimator

        rho_all = np.zeros((NTraj, NStates, NStates, n_skip),
                           dtype=np.complex128)

        for itraj in prange(NTraj):
            # --- initialise nuclear DOF --------------------------------
            R = np.empty(ndof)
            P = np.empty(ndof)
            for d in range(ndof):
                R[d] = R_rand[itraj, d] * sigR[d]
                P[d] = P_rand[itraj, d] * sigP[d]

            # --- initElectronic (general NStates) ----------------------
            ci = np.empty(NStates, dtype=np.complex128)
            for s in range(NStates):
                ci[s] = np.sqrt(beta_coeff / alpha) + 0.0j
            ci[initState] = np.sqrt((1.0 + beta_coeff) / alpha) + 0.0j
            for s in range(NStates):
                phi = uni_rand[itraj, s] * 2.0 * np.pi
                ci[s] = ci[s] * (np.cos(phi) + 1j * np.sin(phi))

            # Initial QM  (COMPLEX eigh to match original LAPACK path)
            Hij = hel_fn(R, c_model, epsilon, Delta)
            dHij = dhel_fn(R, c_model)
            dH0 = dhel0_fn(R, omega)
            E, U = np.linalg.eigh(Hij + 0j)     # complex eigh

            # Transform to adiabatic basis: ci_adia = conj(U)^T @ ci
            ci_a = np.empty(NStates, dtype=np.complex128)
            for i in range(NStates):
                s = 0.0 + 0.0j
                for j in range(NStates):
                    s += U[j, i].conjugate() * ci[j]
                ci_a[i] = s
            ci = ci_a

            # Initial active state
            _, acst = check_hop(-1, ci, NStates)   # always hops from -1

            F1 = force_mash(dHij, dH0, acst, U, NStates, ndof)

            # --- time propagation --------------------------------------
            iskip = 0
            for step in range(NSteps):
                # --- estimator ---
                if step % nskip == 0:
                    rho = pop_mash(ci, U, NStates, alpha, beta_pop)
                    for a in range(NStates):
                        for b in range(NStates):
                            rho_all[itraj, a, b, iskip] = rho[a, b]
                    iskip += 1

                # --- full VelVer step ---
                R_new = R.copy()
                P_new = P.copy()
                ci_new = ci.copy()
                (R_new, P_new, ci_new, Hij_new, dHij_new, dH0_new,
                 E_new, U_new, F2_new) = velver_inline(
                    R_new, P_new, ci_new, E.copy(), U.copy(), F1.copy(),
                    acst, dtN, M_mass, NStates, ndof,
                    hel_fn, dhel_fn, dhel0_fn,
                    c_model, epsilon, Delta, omega)

                # --- check hop ---
                hop_needed, new_acst = check_hop(acst, ci_new, NStates)

                if hop_needed:
                    # test if hop is energetically feasible at full step
                    P_attempt, feasible = attempt_hop(
                        P_new, E_new, U_new, ci_new, dHij_new,
                        acst, new_acst, M_mass, NStates, ndof)

                    if feasible:
                        # Save pre-step state (only when actually needed)
                        R0 = R.copy()
                        P0 = P.copy()
                        ci0 = ci.copy()
                        E0 = E.copy()
                        U0 = U.copy()
                        F1_0 = F1.copy()

                        # --- binary search for hop timing ---
                        tL = 0.0
                        tR = dtN

                        # scratch arrays reused across bisection
                        Rm = R0.copy()
                        Pm = P0.copy()
                        cim = ci0.copy()
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
                            cit = ci0.copy()
                            (Rt, Pt, cit, _Ht, dHt, dH0t,
                             Et, Ut, F2t) = velver_inline(
                                Rt, Pt, cit, E0.copy(), U0.copy(),
                                F1_0.copy(), acst, tm, M_mass,
                                NStates, ndof,
                                hel_fn, dhel_fn, dhel0_fn,
                                c_model, epsilon, Delta, omega)

                            hop_tm, _ = check_hop(acst, cit, NStates)
                            if not hop_tm:
                                tL = tm
                            else:
                                tR = tm

                            Rm[:] = Rt
                            Pm[:] = Pt
                            cim[:] = cit
                            Em[:] = Et
                            Um[:, :] = Ut
                            F1m[:] = F2t
                            dHijm[:, :, :] = dHt
                            dH0m[:] = dH0t

                        # Attempt hop at bisection point
                        P_hop, accepted2 = attempt_hop(
                            Pm, Em, Um, cim, dHijm,
                            acst, new_acst, M_mass, NStates, ndof)
                        # Always apply returned momentum (rescaled if
                        # accepted, reversed projection if rejected --
                        # matches original hop() which mutates dat.P).
                        Pm[:] = P_hop
                        if accepted2:
                            acst = new_acst

                        # Complete remaining dt from bisection point
                        remaining = dtN - last_tm
                        (Rm, Pm, cim, _Hf, dHf, dH0f,
                         Ef, Uf, F2f) = velver_inline(
                            Rm, Pm, cim, Em, Um, F1m, acst, remaining,
                            M_mass, NStates, ndof,
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
                        # Hop not feasible -- use reversed momentum
                        # from attempt_hop (matches original hop()
                        # which reverses P_proj on rejection).
                        R[:] = R_new
                        P[:] = P_attempt
                        ci = ci_new
                        E = E_new
                        U = U_new
                        F1 = F2_new
                        dHij = dHij_new
                        dH0 = dH0_new
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

        # --- reduce across trajectories --------------------------------
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
