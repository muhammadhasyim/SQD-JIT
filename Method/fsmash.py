"""
Method/fsmash.py

Fewest-Switches MASH (Multi-state Mapping Approach to Surface Hopping).

This variant of MASH uses Tully's fewest-switches stochastic hop
probability on top of the standard MASH energy-conservation check.
Hop timing is determined by a midpoint scheme (half-timestep) rather
than bisection search.

The electronic wavefunction is propagated in the adiabatic basis using
exp(-i E dt) unitary evolution, and forces are computed on the active
adiabatic surface.

Adapted from mash_qph.py by Muhammad R. Hasyim (Eric Koessler,
Johan Runeson).

References
----------
* Runeson, Richardson, J. Chem. Phys. 159, 094115 (2023).
"""

import numpy as np
import random
import time


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


# ================================================================ #
#  Mapping variable initialisation
# ================================================================ #

def initElectronic(NStates, initState, Hij, Upolaron=None):
    """
    Initialise MASH mapping coefficients c in the adiabatic basis.

    Parameters
    ----------
    NStates   : int
    initState : int -- index in the diabatic basis
    Hij       : (NStates, NStates) -- electronic Hamiltonian at R
    Upolaron  : (NStates, NStates) or None -- basis rotation matrix

    Returns
    -------
    c : (NStates,) complex128 -- coefficients in adiabatic basis
    """
    sumN = np.sum(np.array([1.0 / n for n in range(1, NStates + 1)]))
    alpha = (NStates - 1) / (sumN - 1)
    beta = (alpha - 1) / NStates

    c = np.sqrt(beta / alpha) * np.ones(NStates, dtype=np.complex128)
    c[initState] = np.sqrt((1.0 + beta) / alpha)
    for n in range(NStates):
        uni = random.random()
        c[n] = c[n] * np.exp(1j * 2.0 * np.pi * uni)

    # Transform to MH / polaron basis if applicable
    if Upolaron is not None:
        c = np.conj(Upolaron).T @ c

    # Transform to adiabatic basis
    E, U = np.linalg.eigh(Hij)
    c = np.conj(U).T @ c
    return c


# ================================================================ #
#  Force and propagation
# ================================================================ #

def Force(acst, R, U, dHel_func, dHel0_func):
    """
    Classical force on each nuclear DOF using the active adiabatic state.

    Parameters
    ----------
    acst       : int -- active adiabatic state
    R          : (ndof,) -- nuclear positions
    U          : (NStates, NStates) -- adiabatic eigenvectors
    dHel_func  : callable -- parameters.dHel
    dHel0_func : callable -- parameters.dHel0
    """
    F = -dHel0_func(R)
    dF = -dHel_func(R)

    # <a|dH|a> = sum_ij U[i,a]* dH[i,j,k] U[j,a]
    F -= np.einsum('i, ijk, j -> k',
                   U[:, acst].conjugate(), dF + 0j, U[:, acst]).real
    return F


def Umap(c, dt, E):
    """Unitary evolution of mapping coefficients in adiabatic basis."""
    return np.exp(-1j * dt * E) * c


def VelVer(R, P, c, acst, F1, M, E, U, dt, Hel_func, dHel_func, dHel0_func):
    """
    Velocity-Verlet integrator with split-operator electronic evolution.

    Electronic propagation: exp(-iE dt/2) -- nuclear step -- exp(-iE dt/2).
    The wavefunction is transformed to the diabatic basis during the
    nuclear step and back to the (new) adiabatic basis afterwards.

    Returns
    -------
    R, P, c, F2, acst, E, U -- updated trajectory variables
    """
    v = P / M

    # Half electronic evolution (adiabatic)
    c = Umap(c, dt / 2.0, E)
    # Adiabatic -> diabatic
    c = U.astype(np.complex128) @ c

    v += 0.5 * F1 * dt / M
    R = R + v * dt

    # New Hamiltonian at updated R
    E, U = np.linalg.eigh(Hel_func(R))
    F2 = Force(acst, R, U, dHel_func, dHel0_func)

    v += 0.5 * F2 * dt / M

    # Diabatic -> new adiabatic
    c = np.conj(U.astype(np.complex128)).T @ c
    # Half electronic evolution (adiabatic)
    c = Umap(c, dt / 2.0, E)

    return R, v * M, c, F2, acst, E, U


# ================================================================ #
#  Population estimator
# ================================================================ #

def pop(c, NStates):
    """
    MASH density-matrix estimator with zero-point-energy correction.

    Parameters
    ----------
    c       : (NStates,) -- mapping coefficients (any basis)
    NStates : int

    Returns
    -------
    rho : (NStates, NStates) complex
    """
    sumN = np.sum(np.array([1.0 / n for n in range(1, NStates + 1)]))
    alpha = (NStates - 1) / (sumN - 1)
    beta = (1.0 - alpha) / NStates
    return alpha * np.outer(c, np.conj(c)) + beta * np.identity(NStates)


# ================================================================ #
#  Hop detection and hopping
# ================================================================ #

def checkHop(acst, c):
    """Check whether the active state should change."""
    n_max = np.argmax(np.abs(c))
    if acst != n_max:
        return True, acst, n_max
    return False, acst, acst


def hop(c, P, R, M, Ead, U, a, b, dtN, dHel_func):
    """
    Attempt a surface hop from adiabatic state *a* to *b*.

    Uses energy-conservation check followed by Tully's fewest-switches
    stochastic probability.  Momentum is rescaled along the hopping
    direction.

    Parameters
    ----------
    c     : (NStates,) complex  -- adiabatic mapping coefficients
    P     : (ndof,) float       -- nuclear momenta
    R     : (ndof,) float       -- nuclear positions
    M     : float               -- nuclear mass
    Ead   : (NStates,) float    -- adiabatic energies
    U     : (NStates, NStates)  -- adiabatic eigenvectors
    a, b  : int                 -- previous and proposed active states
    dtN   : float               -- nuclear timestep
    dHel_func : callable        -- parameters.dHel

    Returns
    -------
    P        : (ndof,) float -- (possibly rescaled) momenta
    accepted : bool
    """
    if a == b:
        return P, False

    P_scaled = P / np.sqrt(M)
    dE = np.real(Ead[b] - Ead[a])

    dHij = dHel_func(R)

    # Non-adiabatic coupling direction
    j = np.arange(len(Ead))
    dEa = Ead[a] - Ead
    dEb = Ead[b] - Ead
    dEa[a] = 1.0   # avoid division by zero
    dEb[b] = 1.0
    rdEa = (a != j) / dEa
    rdEb = (b != j) / dEb

    # Transform dHij to adiabatic basis
    dHab = np.einsum('ia, ijk, jb -> abk', U.conjugate(), dHij + 0j, U)

    term1 = np.einsum('n, nj, n -> j',
                       c.conjugate(), dHab[:, a, :] * c[a], rdEa)
    term2 = np.einsum('n, nj, n -> j',
                       c.conjugate(), dHab[:, b, :] * c[b], rdEb)

    dk = (term1 - term2).real / np.sqrt(M)

    # Project momentum
    dk_norm2 = np.dot(dk, dk)
    if dk_norm2 < 1e-30:
        return P, False
    P_proj = np.dot(P_scaled, dk) * dk / dk_norm2
    P_orth = P_scaled - P_proj
    P_proj_norm = np.sqrt(np.dot(P_proj, P_proj))

    if P_proj_norm**2 < 2.0 * dE:
        # Insufficient energy -- reject hop, reverse projected momentum
        P_new = (P_orth - P_proj) * np.sqrt(M)
        return P_new.real, False

    # Energy conserved -- now apply Tully probability
    P_proj_old = np.copy(P_proj)
    P_old = np.copy(P_scaled)
    P_proj_rescaled = np.sqrt(P_proj_norm**2 - 2.0 * dE) / P_proj_norm * P_proj

    # Tully's fewest-switches probability
    pop_a = c.conjugate()[a] * c[a]
    plz_vec = dtN * (c.conjugate()[b] * c[a]) * dHab[b, a, :] / (pop_a * dE)
    plz = max(0.0, np.dot(P_old, plz_vec.real))

    if np.random.uniform(0, 1) < plz:
        # Accept hop
        P_new = (P_orth + P_proj_rescaled) * np.sqrt(M)
        return P_new.real, True
    else:
        # Reject hop -- reverse projected momentum
        P_new = (P_orth - P_proj_old) * np.sqrt(M)
        return P_new.real, False


# ================================================================ #
#  Main trajectory loop
# ================================================================ #

def runTraj(parameters):
    """
    Run an ensemble of FS-MASH trajectories.

    Parameters
    ----------
    parameters : object
        Must provide: NSteps, NTraj, NStates, initState, nskip, dtN, M,
        Hel(R), dHel(R), dHel0(R), initR().
        Optionally: Upolaron, SEED.

    Returns
    -------
    rho_ensemble : (NStates, NStates, n_frames) complex
        Un-normalised density matrix summed over trajectories.
        Divide by NTraj for average.
    """
    # ------- Seed --------------------
    try:
        np.random.seed(parameters.SEED)
    except AttributeError:
        pass

    # ------- Parameters ---------------
    NSteps    = parameters.NSteps
    NTraj     = parameters.NTraj
    NStates   = parameters.NStates
    initState = parameters.initState
    nskip     = parameters.nskip
    dtN       = parameters.dtN
    M         = parameters.M

    Hel_func   = parameters.Hel
    dHel_func  = parameters.dHel
    dHel0_func = parameters.dHel0
    initR_func = parameters.initR
    Upolaron   = getattr(parameters, 'Upolaron', None)
    # ----------------------------------

    if NSteps % nskip == 0:
        pl = 0
    else:
        pl = 1

    rho_ensemble = np.zeros((NStates, NStates, NSteps // nskip + pl),
                            dtype=complex)

    for itraj in range(NTraj):
        t0 = time.time()

        # Initialise nuclear DOFs
        R, P = initR_func()

        # Initial electronic Hamiltonian and adiabatic basis
        Hij = Hel_func(R) + 0j
        E, U = np.linalg.eigh(Hij)

        # Initialise mapping coefficients (adiabatic basis)
        c = initElectronic(NStates, initState, Hij, Upolaron)
        acst = np.argmax(np.abs(c))

        # Initial force on active surface
        F1 = Force(acst, R, U, dHel_func, dHel0_func)

        iskip = 0
        for t in range(NSteps):
            # ------- Estimators -----------------------------------
            if t % nskip == 0:
                cD = U @ c  # transform to diabatic/working basis
                rho_ensemble[:, :, iskip] += pop(cD, NStates)
                iskip += 1
            # -------------------------------------------------------

            dt = dtN

            # Save state for possible hop
            R0  = R.copy()
            P0  = P.copy()
            c0  = c.copy()
            F10 = F1.copy()
            acst0 = acst
            E0  = E.copy()
            U0  = U.copy()

            # Full Verlet step
            R, P, c, F1, acst, E, U = VelVer(
                R, P, c, acst, F1, M, E, U, dt,
                Hel_func, dHel_func, dHel0_func)

            # Check if active state changed
            hop_needed, _, new_acst = checkHop(acst, c)
            if hop_needed:
                # Restore and propagate to midpoint
                tm = dt / 2.0
                R, P, c, F1, acst, E, U = (
                    R0.copy(), P0.copy(), c0.copy(), F10.copy(),
                    acst0, E0.copy(), U0.copy()
                )
                R, P, c, F1, acst, E, U = VelVer(
                    R, P, c, acst, F1, M, E, U, tm,
                    Hel_func, dHel_func, dHel0_func)

                # Find candidate state
                p_abs = np.abs(c)**2
                p_abs[acst] = 0.0
                b = np.argmax(p_abs)

                # Attempt hop with Tully probability
                P, accepted = hop(
                    c, P, R, M, E, U, acst, b, dtN, dHel_func)

                if accepted:
                    acst = b
                    F1 = Force(acst, R, U, dHel_func, dHel0_func)

                # Finish the timestep
                R, P, c, F1, acst, E, U = VelVer(
                    R, P, c, acst, F1, M, E, U, dt - tm,
                    Hel_func, dHel_func, dHel0_func)

        time_taken = time.time() - t0
        print(f"Time taken: {time_taken:.2f} seconds")

    return rho_ensemble
