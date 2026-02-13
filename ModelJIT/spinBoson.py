"""
JIT-compiled spin-boson model functions.

All functions are pure @njit with explicit parameter passing --
no class attributes, no globals.  The ``get_model_params`` helper
(plain Python) returns every scalar / array the JIT kernels need.

References
----------
Original: Model/spinBoson.py
"""

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Plain-Python helpers (run once at setup, NOT JIT-compiled)
# ---------------------------------------------------------------------------

def model(M=3):
    """Return (epsilon, xi, beta, omega_c, Delta, ndof) for model index M."""
    epsilon = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 5.0])
    xi      = np.array([0.09, 0.09, 0.1, 0.1, 2.0, 4.0])
    beta    = np.array([0.1, 5.0, 0.25, 5.0, 1.0, 0.1])
    omega_c = np.array([2.5, 2.5, 1.0, 2.5, 1.0, 2.0])
    Delta   = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    N       = np.array([100, 100, 100, 100, 400, 400], dtype=np.int64)
    return epsilon[M], xi[M], beta[M], omega_c[M], Delta[M], int(N[M])


def bath_param(xi, omega_c, ndof):
    """Discretised Ohmic spectral density -> coupling c, frequencies omega."""
    omega_max = 4.0
    omega_0 = omega_c * (1.0 - np.exp(-omega_max)) / ndof
    c = np.zeros(ndof)
    omega = np.zeros(ndof)
    for d in range(ndof):
        omega[d] = -omega_c * np.log(1.0 - (d + 1) * omega_0 / omega_c)
        c[d] = np.sqrt(xi * omega_0) * omega[d]
    return c, omega


def get_model_params(M=3):
    """
    Return a dict of all model parameters needed by JIT kernels.

    This is called ONCE before the simulation; the returned arrays are
    passed directly into the JIT-compiled ``run_traj`` function.
    """
    epsilon, xi, beta, omega_c, Delta, ndof = model(M)
    c, omega = bath_param(xi, omega_c, ndof)

    # Thermal widths for initial-condition sampling
    sigP = np.sqrt(omega / (2.0 * np.tanh(0.5 * beta * omega)))
    sigR = sigP / omega

    return dict(
        NStates=2,
        NSteps=200,
        NTraj=200,
        dtN=0.01,
        EStep=20,       # int(dtN / dtE) where dtE = dtN/20
        M_mass=1.0,     # nuclear mass (scalar, all DOFs equal)
        initState=0,
        nskip=10,
        ndof=ndof,
        epsilon=epsilon,
        Delta=Delta,
        beta=beta,
        c=c,
        omega=omega,
        sigR=sigR,
        sigP=sigP,
    )


# ---------------------------------------------------------------------------
# JIT-compiled model functions  (called inside prange kernels)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, boundscheck=False, error_model='numpy')
def hel(R, c, epsilon, Delta):
    """
    Electronic Hamiltonian  H_el(R)  for the spin-boson model.

    Parameters
    ----------
    R       : (ndof,) float64  -- nuclear positions
    c       : (ndof,) float64  -- system-bath couplings
    epsilon : float64           -- bias
    Delta   : float64           -- tunnelling splitting

    Returns
    -------
    Vij : (2, 2) float64
    """
    Vij = np.zeros((2, 2))
    coupling = 0.0
    for d in range(len(R)):
        coupling += c[d] * R[d]
    Vij[0, 0] =  coupling + epsilon
    Vij[1, 1] = -coupling - epsilon
    Vij[0, 1] = Delta
    Vij[1, 0] = Delta
    return Vij


@njit(cache=True, fastmath=True, boundscheck=False, error_model='numpy')
def dhel(R, c):
    """
    Gradient of H_el w.r.t. nuclear coordinates (state-dependent part).

    Returns
    -------
    dHij : (2, 2, ndof) float64
    """
    ndof = len(R)
    dHij = np.zeros((2, 2, ndof))
    for d in range(ndof):
        dHij[0, 0, d] =  c[d]
        dHij[1, 1, d] = -c[d]
    return dHij


@njit(cache=True, fastmath=True, boundscheck=False, error_model='numpy')
def dhel0(R, omega):
    """
    State-independent gradient  dV_0 / dR = omega^2 * R  (harmonic bath).

    Returns
    -------
    dH0 : (ndof,) float64
    """
    ndof = len(R)
    dH0 = np.empty(ndof)
    for d in range(ndof):
        dH0[d] = omega[d] * omega[d] * R[d]
    return dH0
