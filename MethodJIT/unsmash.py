"""
JIT-compiled unSMASH (uncoupled Spheres Multi-state MASH) method.

Reference:
  Lawrence, Mannouch & Richardson,
  "A Size-Consistent Multi-State Mapping Approach to Surface Hopping"

Key differences from MethodJIT/mash.py:
  - Electronic variables are N-1 independent Bloch spheres (one per non-active
    adiabatic state), each on the unit sphere.
  - Each sphere evolves as an isolated two-state subsystem (Eq. 6) -- this is
    what makes the method size-consistent.
  - Hop detection: S_z crosses zero (not argmax of |c|^2).
  - Hop direction: NACV between active and target state (simpler than MASH).
  - Sphere relabelling after a successful hop (Eqs. 10-12).
  - Density-matrix estimator uses g_P / g_C weights for diabatic observables
    (Appendix A).

Data layout:
  Spheres are stored as S[NStates, 3] where S[b, :] = (Sx, Sy, Sz) for the
  pair (active_state, b).  S[active_state, :] is unused.
"""
import numpy as np
from numba import njit, prange

_jit = dict(cache=True, fastmath=True, boundscheck=False, error_model='numpy')


# ========================================================================== #
#  Helper: force on the active surface  (identical to MASH)                  #
# ========================================================================== #

@njit(**_jit)
def force_unsmash(dHij, dH0, acst, U, NStates, ndof):
    """
    F = -dH0 - Re( sum_ij conj(U[i,acst]) * dHij[i,j,d] * U[j,acst] )
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


# ========================================================================== #
#  Bloch-sphere  <-->  effective two-state coefficients                      #
# ========================================================================== #

@njit(**_jit)
def sphere_to_coeffs(Sx, Sy, Sz):
    """
    Convert normalised Bloch-sphere coords to effective 2-state coefficients.

    Convention: c_n (active-state coeff) is real and positive.
        c_n = sqrt((1 + Sz) / 2)
        c_b = (Sx + i Sy) / (2 c_n)

    Returns (c_n, c_b) as complex128.
    """
    cn_real = np.sqrt(max((1.0 + Sz) * 0.5, 0.0))
    if cn_real < 1e-14:
        # S_z ~ -1: active state has ~zero population (edge case near hop)
        cn = 0.0 + 0.0j
        cb = 1.0 + 0.0j
    else:
        cn = cn_real + 0.0j
        cb = (Sx + 1j * Sy) / (2.0 * cn_real)
    return cn, cb


@njit(**_jit)
def coeffs_to_sphere(cn, cb):
    """
    Convert effective 2-state coefficients back to Bloch-sphere coords.

    Sz = |cn|^2 - |cb|^2
    Sx = 2 Re(cn* cb)
    Sy = 2 Im(cn* cb)
    """
    Sz = (cn * cn.conjugate()).real - (cb * cb.conjugate()).real
    prod = cn.conjugate() * cb
    Sx = 2.0 * prod.real
    Sy = 2.0 * prod.imag
    return Sx, Sy, Sz


# ========================================================================== #
#  Half-step electronic evolution  (z-rotation of Bloch sphere)              #
# ========================================================================== #

@njit(**_jit)
def z_rotate_sphere(Sx, Sy, Sz, E_acst, E_b, dt):
    """
    Half-step adiabatic phase evolution as a rotation about the z-axis.

    phi = (E_acst - E_b) * dt / 2     (hbar = 1 in atomic units)

    Sx' =  Sx cos(phi) + Sy sin(phi)
    Sy' = -Sx sin(phi) + Sy cos(phi)
    Sz' =  Sz
    """
    phi = (E_acst - E_b) * dt * 0.5
    cp = np.cos(phi)
    sp = np.sin(phi)
    Sx_new = Sx * cp + Sy * sp
    Sy_new = -Sx * sp + Sy * cp
    return Sx_new, Sy_new, Sz


# ========================================================================== #
#  Hop detection: any S_z < 0?                                               #
# ========================================================================== #

@njit(**_jit)
def check_hop_unsmash(S, acst, NStates):
    """
    Return (hop_needed, target_state).

    A hop from *acst* to *b* is triggered when S_z^{(acst, b)} < 0.
    If multiple spheres have crossed, pick the one with most-negative S_z
    (largest population on the target surface).
    """
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


# ========================================================================== #
#  Hop direction (NACV) + momentum rescaling                                 #
# ========================================================================== #

@njit(**_jit)
def attempt_hop_unsmash(P, E, U, dHij, acst, newacst, M_mass, NStates, ndof):
    """
    Hop direction = NACV d^{(n,b)} between active state n and target b.

    d_k^{(n,b)} = <n|dH/dR_k|b> / (E_b - E_n)

    Mass-weighted: dk_tilde = d / sqrt(M)

    Energy check (Eq. 7): E_kin^(d) > V_b - V_n
    Rejected hop  (Eq. 8): reverse momentum along dk
    Accepted hop  (Eq. 9): rescale momentum along dk to conserve energy

    Returns (P_new, accepted).
    """
    if acst == newacst:
        return P.copy(), False

    a, b = acst, newacst
    dE = E[b] - E[a]  # energy gap

    if abs(dE) < 1e-30:
        return P.copy(), False

    sqrtM = np.sqrt(M_mass)

    # dk[d] = <a|dH/dR_d|b> / (E_b - E_a) / sqrt(M)
    dk = np.zeros(ndof)
    for d in range(ndof):
        dH_ab = 0.0 + 0.0j
        for i in range(NStates):
            for j in range(NStates):
                dH_ab += U[i, a].conjugate() * dHij[i, j, d] * U[j, b]
        dk[d] = dH_ab.real / dE / sqrtM

    # Project momentum onto dk direction
    dk_dot = 0.0
    P_dk = 0.0
    for d in range(ndof):
        dk_dot += dk[d] * dk[d]
        P_dk += (P[d] / sqrtM) * dk[d]

    if dk_dot < 1e-30:
        return P.copy(), False

    # E_kin^(d) = (p_tilde . dk)^2 / (2 dk.dk)
    P_proj_norm2 = P_dk * P_dk / dk_dot

    P_new = np.empty(ndof)
    if P_proj_norm2 < 2.0 * dE:
        # Rejected hop -- reverse projected component  (Eq. 8)
        for d in range(ndof):
            proj = P_dk * dk[d] / dk_dot
            orth = P[d] / sqrtM - proj
            P_new[d] = (orth - proj) * sqrtM
        return P_new, False
    else:
        # Accepted hop -- rescale projected component  (Eq. 9)
        scale = np.sqrt(P_proj_norm2 - 2.0 * dE) / np.sqrt(P_proj_norm2)
        for d in range(ndof):
            proj = P_dk * dk[d] / dk_dot
            orth = P[d] / sqrtM - proj
            P_new[d] = (orth + scale * proj) * sqrtM
        return P_new, True


# ========================================================================== #
#  Sphere relabelling after a successful hop  (Eqs. 10-12)                   #
# ========================================================================== #

@njit(**_jit)
def relabel_spheres(S, old_acst, new_acst, NStates):
    """
    After hop from old_acst (n_i) to new_acst (n_f = b):

    For mu != n_i  (and mu != n_f):
        S_new[mu] = S_old[mu]         -- keep the sphere as-is
    For mu = n_i:
        S_new[n_i] = (Sx^{(n_i,n_f)},  -Sy^{(n_i,n_f)},  -Sz^{(n_i,n_f)})
                   = (S_old[n_f, 0],    -S_old[n_f, 1],    -S_old[n_f, 2])
    S_new[n_f] is unused (new active state slot).

    Modifies S in-place.
    """
    ni = old_acst
    nf = new_acst

    # Save the sphere that was between old_acst and new_acst
    sx_hop = S[nf, 0]
    sy_hop = S[nf, 1]
    sz_hop = S[nf, 2]

    # The sphere for the new pair (n_f, n_i) uses the identity (Eq. 12):
    #   S^{(b,a)} = (Sx^{(a,b)}, -Sy^{(a,b)}, -Sz^{(a,b)})
    S[ni, 0] = sx_hop
    S[ni, 1] = -sy_hop
    S[ni, 2] = -sz_hop

    # All other spheres (mu != ni, mu != nf) stay the same -- no action needed.
    # The nf slot is now the active state, mark as unused:
    S[nf, 0] = 0.0
    S[nf, 1] = 0.0
    S[nf, 2] = 0.0


# ========================================================================== #
#  VelVer with uncoupled Bloch-sphere dynamics                               #
# ========================================================================== #

@njit(**_jit)
def velver_unsmash(R, P, S, E, U, F1, acst, dt, M_mass, NStates, ndof,
                   hel_fn, dhel_fn, dhel0_fn,
                   c_model, epsilon, Delta, omega):
    """
    One Velocity-Verlet step evolving N-1 Bloch spheres independently.

    Split-operator scheme per sphere (acst, b):
      1. Half z-rotation  (adiabatic phase)
      2. To diabatic      (effective 2-state -> full diabatic via U_old)
      3. Nuclear step      (position + recompute QM)
      4. Back to adiabatic (full diabatic -> effective 2-state via U_new)
      5. Renormalise to unit sphere
      6. Half z-rotation  (new adiabatic phase)

    Returns (R, P, S, Hij, dHij, dH0, E_new, U_new, F2).
    """
    # --- diabatic storage for each sphere's effective wavefunction ----------
    # cD_all[b, i] stores the NStates-dim diabatic vector for sphere b
    cD_all = np.zeros((NStates, NStates), dtype=np.complex128)

    for b in range(NStates):
        if b == acst:
            continue

        # 1. Half z-rotation
        Sx, Sy, Sz = z_rotate_sphere(S[b, 0], S[b, 1], S[b, 2],
                                     E[acst], E[b], dt)

        # 2. Convert to effective coefficients
        cn, cb_ = sphere_to_coeffs(Sx, Sy, Sz)

        # 3. To diabatic: cD[i] = U_old[i, acst]*cn + U_old[i, b]*cb_
        for i in range(NStates):
            cD_all[b, i] = U[i, acst] * cn + U[i, b] * cb_

    # --- nuclear position update -------------------------------------------
    v = np.empty(ndof)
    for d in range(ndof):
        v[d] = P[d] / M_mass
        R[d] += v[d] * dt + 0.5 * F1[d] * dt * dt / M_mass

    # --- recompute QM at new R ---------------------------------------------
    Hij = hel_fn(R, c_model, epsilon, Delta)
    dHij = dhel_fn(R, c_model)
    dH0 = dhel0_fn(R, omega)
    E_new, U_new = np.linalg.eigh(Hij + 0j)

    # --- nuclear momentum update -------------------------------------------
    F2 = force_unsmash(dHij, dH0, acst, U_new, NStates, ndof)
    for d in range(ndof):
        v[d] += 0.5 * (F1[d] + F2[d]) * dt / M_mass
        P[d] = v[d] * M_mass

    # --- back to adiabatic and second half-step for each sphere ------------
    for b in range(NStates):
        if b == acst:
            continue

        # Back to adiabatic: cn = U_new^dag . cD  projected onto (acst, b)
        cn_new = 0.0 + 0.0j
        cb_new = 0.0 + 0.0j
        for i in range(NStates):
            cn_new += U_new[i, acst].conjugate() * cD_all[b, i]
            cb_new += U_new[i, b].conjugate() * cD_all[b, i]

        # Renormalise to unit sphere
        norm2 = (cn_new * cn_new.conjugate()).real + \
                (cb_new * cb_new.conjugate()).real
        if norm2 > 1e-30:
            inv_norm = 1.0 / np.sqrt(norm2)
            cn_new = cn_new * inv_norm
            cb_new = cb_new * inv_norm
        else:
            cn_new = 1.0 + 0.0j
            cb_new = 0.0 + 0.0j

        # Convert to sphere
        Sx, Sy, Sz = coeffs_to_sphere(cn_new, cb_new)

        # 6. Half z-rotation with new E
        Sx, Sy, Sz = z_rotate_sphere(Sx, Sy, Sz, E_new[acst], E_new[b], dt)

        S[b, 0] = Sx
        S[b, 1] = Sy
        S[b, 2] = Sz

    return R, P, S, Hij, dHij, dH0, E_new, U_new, F2


# ========================================================================== #
#  Initial weight factors for diabatic initialisation  (Appendix A)          #
# ========================================================================== #

@njit(**_jit)
def compute_g_factors(S, acst, U, initState, NStates):
    """
    Compute g_j^P and g_j^C for a diabatic initial state |j> = |initState>.

    g_j^P  = rho_P |<j|n>|^2
           + sum_{a!=n} 2 Re(<j|n><a|j>) Sx^{(n,a)}
           - sum_{a!=n} 2 Im(<j|n><a|j>) Sy^{(n,a)}

    g_j^C  = 2 |<j|n>|^2
           + sum_{a!=n} 3 Re(<j|n><a|j>) Sx^{(n,a)}
           - sum_{a!=n} 3 Im(<j|n><a|j>) Sy^{(n,a)}

    where <j|a> = U[j, a]  (j-th row, a-th column of the eigenvector matrix),
    and  rho_P = prod_{mu != n} 2 |Sz^{(n,mu)}|.
    """
    j = initState
    n = acst

    # rho_P = prod_{mu != n} 2 |S_z^{(n,mu)}|
    rho_P = 1.0
    for mu in range(NStates):
        if mu == n:
            continue
        rho_P *= 2.0 * abs(S[mu, 2])

    # <j|n> and <a|j> use columns of U  (U diagonalises H: H = U E U^dag)
    # In our convention U[i, a] = <diabat i | adiabat a>
    jn = U[j, n]  # <j|n>   (complex)
    jn_abs2 = (jn * jn.conjugate()).real

    gP = rho_P * jn_abs2
    gC = 2.0 * jn_abs2

    for a in range(NStates):
        if a == n:
            continue
        aj = U[j, a].conjugate()    # <a|j> = conj(U[j,a])
        z = jn * aj                 # <j|n><a|j>
        Sx_na = S[a, 0]
        Sy_na = S[a, 1]
        gP += 2.0 * z.real * Sx_na - 2.0 * z.imag * Sy_na
        gC += 3.0 * z.real * Sx_na - 3.0 * z.imag * Sy_na

    return gP, gC


# ========================================================================== #
#  Population / density-matrix estimator  (Eqs. 21-24 + Appendix A)          #
# ========================================================================== #

@njit(**_jit)
def pop_unsmash(S, acst, U, gP, gC, NStates):
    """
    Diabatic density matrix estimator for a single trajectory.

    rho[j,j'] = N * gP * U[j,n_t] conj(U[j',n_t])
              + N * gC * sum_{b!=n_t} [
                    U[j,n_t] sigma_{n_t,b} conj(U[j',b])
                  + U[j,b]  conj(sigma_{n_t,b}) conj(U[j',n_t])
                ]

    sigma_{n,b} = (Sx^{(n,b)} - i Sy^{(n,b)}) / 2   (Eq. 18)

    The factor N (= NStates) compensates the 1/N sampling probability of
    each initial active state.
    """
    N = float(NStates)
    nt = acst
    rho = np.zeros((NStates, NStates), dtype=np.complex128)

    # Population part
    for jp in range(NStates):
        for jpp in range(NStates):
            rho[jp, jpp] += N * gP * U[jp, nt] * U[jpp, nt].conjugate()

    # Coherence part
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


# ========================================================================== #
#  Kernel factory                                                            #
# ========================================================================== #

_kernel_cache = {}


def make_unsmash_kernel(hel_fn, dhel_fn, dhel0_fn):
    """
    Factory returning a parallel unSMASH kernel.

    Works for any NStates >= 2; rigorously recovers the original two-state
    MASH for NStates == 2.

    Results are memoised: calling with the same functions returns the
    already-compiled kernel without recompilation.
    """
    key = (hel_fn, dhel_fn, dhel0_fn)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @njit(parallel=True, **_jit)
    def run_traj(NTraj, NSteps, NStates, nskip, dtN, M_mass,
                 initState, maxhop,
                 R_rand, P_rand, sigR, sigP,
                 sphere_rand,      # (NTraj, NStates, 2): Sz=u, phi=2pi*v
                 acst_rand,        # (NTraj,) uniform [0,1] -> initial acst
                 c_model, epsilon, Delta, omega):

        n_skip = NSteps // nskip
        ndof = len(sigR)

        rho_all = np.zeros((NTraj, NStates, NStates, n_skip),
                           dtype=np.complex128)

        for itraj in prange(NTraj):
            # ============================================================== #
            #  1. Initialise nuclear DOF                                     #
            # ============================================================== #
            R = np.empty(ndof)
            P = np.empty(ndof)
            for d in range(ndof):
                R[d] = R_rand[itraj, d] * sigR[d]
                P[d] = P_rand[itraj, d] * sigP[d]

            # ============================================================== #
            #  2. Initial QM                                                 #
            # ============================================================== #
            Hij = hel_fn(R, c_model, epsilon, Delta)
            dHij = dhel_fn(R, c_model)
            dH0 = dhel0_fn(R, omega)
            E, U = np.linalg.eigh(Hij + 0j)

            # ============================================================== #
            #  3. Choose initial active state (uniform)                      #
            # ============================================================== #
            acst = int(acst_rand[itraj] * NStates)
            if acst >= NStates:
                acst = NStates - 1

            # ============================================================== #
            #  4. Sample Bloch spheres UNIFORMLY on upper hemisphere         #
            #     The measure (Eq. 16) gives dS = dSz dphi/(2pi)            #
            #     so S_z ~ U[0,1] and phi ~ U[0,2pi).                       #
            #     The rho_P weighting is included in g_j^P (Appendix A).    #
            # ============================================================== #
            S = np.zeros((NStates, 3))
            for b in range(NStates):
                if b == acst:
                    continue
                u = sphere_rand[itraj, b, 0]
                v = sphere_rand[itraj, b, 1]
                Sz = u                               # uniform on [0,1]
                phi = 2.0 * np.pi * v
                sin_theta = np.sqrt(max(1.0 - Sz * Sz, 0.0))
                S[b, 0] = sin_theta * np.cos(phi)    # Sx
                S[b, 1] = sin_theta * np.sin(phi)    # Sy
                S[b, 2] = Sz                          # Sz

            # ============================================================== #
            #  5. Compute initial g_P, g_C weights  (diabatic init)          #
            # ============================================================== #
            gP, gC = compute_g_factors(S, acst, U, initState, NStates)

            # ============================================================== #
            #  6. Initial force on active surface                            #
            # ============================================================== #
            F1 = force_unsmash(dHij, dH0, acst, U, NStates, ndof)

            # ============================================================== #
            #  7. Time propagation                                           #
            # ============================================================== #
            iskip = 0
            for step in range(NSteps):

                # --- estimator ------------------------------------------- #
                if step % nskip == 0:
                    rho = pop_unsmash(S, acst, U, gP, gC, NStates)
                    for a in range(NStates):
                        for bb in range(NStates):
                            rho_all[itraj, a, bb, iskip] = rho[a, bb]
                    iskip += 1

                # --- full VelVer step ------------------------------------ #
                R_new = R.copy()
                P_new = P.copy()
                S_new = S.copy()

                (R_new, P_new, S_new, Hij_new, dHij_new, dH0_new,
                 E_new, U_new, F2_new) = velver_unsmash(
                    R_new, P_new, S_new, E.copy(), U.copy(), F1.copy(),
                    acst, dtN, M_mass, NStates, ndof,
                    hel_fn, dhel_fn, dhel0_fn,
                    c_model, epsilon, Delta, omega)

                # --- check hop ------------------------------------------- #
                hop_needed, new_acst = check_hop_unsmash(
                    S_new, acst, NStates)

                if hop_needed:
                    # test if hop is energetically feasible at full step
                    P_attempt, feasible = attempt_hop_unsmash(
                        P_new, E_new, U_new, dHij_new,
                        acst, new_acst, M_mass, NStates, ndof)

                    if feasible:
                        # Save pre-step state
                        R0 = R.copy()
                        P0 = P.copy()
                        S0 = S.copy()
                        E0 = E.copy()
                        U0 = U.copy()
                        F1_0 = F1.copy()

                        # --- binary search for hop timing --------------- #
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
                             Et, Ut, F2t) = velver_unsmash(
                                Rt, Pt, St, E0.copy(), U0.copy(),
                                F1_0.copy(), acst, tm, M_mass,
                                NStates, ndof,
                                hel_fn, dhel_fn, dhel0_fn,
                                c_model, epsilon, Delta, omega)

                            hop_tm, _ = check_hop_unsmash(
                                St, acst, NStates)
                            if not hop_tm:
                                tL = tm
                            else:
                                tR = tm

                            Rm[:] = Rt
                            Pm[:] = Pt
                            for ss in range(NStates):
                                for cc in range(3):
                                    Sm[ss, cc] = St[ss, cc]
                            Em[:] = Et
                            Um[:, :] = Ut
                            F1m[:] = F2t
                            dHijm[:, :, :] = dHt
                            dH0m[:] = dH0t

                        # Attempt hop at bisection point
                        P_hop, accepted2 = attempt_hop_unsmash(
                            Pm, Em, Um, dHijm,
                            acst, new_acst, M_mass, NStates, ndof)
                        # Always apply returned momentum (rescaled if
                        # accepted, reversed projection if rejected --
                        # matches original hop() which mutates P).
                        Pm[:] = P_hop
                        if accepted2:
                            # Relabel spheres  (Eqs. 10-12)
                            relabel_spheres(Sm, acst, new_acst, NStates)
                            acst = new_acst

                        # Complete remaining dt from bisection point
                        remaining = dtN - last_tm
                        (Rm, Pm, Sm, _Hf, dHf, dH0f,
                         Ef, Uf, F2f) = velver_unsmash(
                            Rm, Pm, Sm, Em, Um, F1m, acst, remaining,
                            M_mass, NStates, ndof,
                            hel_fn, dhel_fn, dhel0_fn,
                            c_model, epsilon, Delta, omega)

                        R[:] = Rm
                        P[:] = Pm
                        for ss in range(NStates):
                            for cc in range(3):
                                S[ss, cc] = Sm[ss, cc]
                        E = Ef
                        U = Uf
                        F1 = F2f
                        dHij = dHf
                        dH0 = dH0f
                    else:
                        # Hop not feasible -- use reversed momentum
                        # from attempt_hop (matches original which
                        # reverses P_proj on rejection).
                        R[:] = R_new
                        P[:] = P_attempt
                        for ss in range(NStates):
                            for cc in range(3):
                                S[ss, cc] = S_new[ss, cc]
                        E = E_new
                        U = U_new
                        F1 = F2_new
                        dHij = dHij_new
                        dH0 = dH0_new
                else:
                    # No hop -- keep full-step result
                    R[:] = R_new
                    P[:] = P_new
                    for ss in range(NStates):
                        for cc in range(3):
                            S[ss, cc] = S_new[ss, cc]
                    E = E_new
                    U = U_new
                    F1 = F2_new
                    dHij = dHij_new
                    dH0 = dH0_new

        # --- reduce across trajectories ---------------------------------- #
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
