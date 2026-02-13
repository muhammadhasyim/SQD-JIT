"""
Model/vsc1m_effbath.py

Single-molecule vibrational strong coupling (VSC) model with an
*effective bath* description.

The double-well matter system is treated quantum mechanically (truncated
to Nm eigenstates).  The cavity photon is NOT treated explicitly;
instead, the combined effect of the cavity mode and the cavity loss
bath is captured by an effective spectral density

    J_eff(w) = 2 * eta^2 * wc^3 * w / [tau_c * ((wc^2 - w^2)^2 + w^2/tau_c^2)]

discretised into Ncbath harmonic oscillators.  The molecular bath
(Debye spectral density) is discretised into Nmbath oscillators.  All
bath modes couple through the matter position operator Rij.

References
----------
* Lindoy, Mandal, Reichman, Nature Commun. 14, 2733 (2023).
"""

import numpy as np
from numpy import random as rn

# ================================================================ #
#  Unit conversions (atomic units)
# ================================================================ #
au   = 27.2113961
ps   = 41341.374575751
fs   = 41.34136
cm   = 1 / 4.556335e-6    # 219474.63
kB   = 3.166829e-6
eV   = 1.0 / 27.2114
cmn1 = 1.0 / 219474.63


# ================================================================ #
#  Helper functions (used only at build time)
# ================================================================ #

def _double_well_potential(R, wb, Eb):
    """Symmetric double-well potential V(R), shifted so V_min = 0."""
    a1 = wb * wb / 2.0
    a2 = a1 * a1 / (4.0 * Eb)
    V = -a1 * R**2 + a2 * R**4
    return V - np.min(V)


def _kinetic_energy_dvr(rgrid):
    """Kinetic energy operator (mass = 1) via sinc-DVR."""
    N = len(rgrid)
    Tij = np.zeros((N, N))
    step = (rgrid[-1] - rgrid[0]) / N
    K = np.pi / step
    for ri in range(N):
        for rj in range(N):
            if ri == rj:
                Tij[ri, ri] = 0.5 * K**2 / 3.0 * (1.0 + 2.0 / N**2)
            else:
                Tij[ri, rj] = (0.5 * 2.0 * K**2 / N**2) * \
                    ((-1)**(rj - ri) / np.sin(np.pi * (rj - ri) / N)**2)
    return Tij


def _discretize_debye(N, wmax, lamb, gamm):
    """Discretise a Debye spectral density into N oscillators."""
    w = np.linspace(1e-8, wmax, N * 1000)
    Fw = (2.0 * lamb / np.pi) * np.arctan(w / gamm)
    lambs = Fw[-1]
    wj = np.zeros(N)
    cj = np.zeros(N)
    for i in range(N):
        j = i + 1
        wj[i] = w[np.argmin(np.abs(Fw - ((j - 0.5) / N) * lambs))]
        cj[i] = wj[i] * (2.0 * lambs / N)**0.5
    return wj, cj


def _discretize_general(N, J_func, w):
    """
    Discretise an arbitrary spectral density J(w) into N oscillators
    via the cumulative-density method.

    Parameters
    ----------
    N      : int            -- number of oscillators
    J_func : callable       -- J(w), spectral density function
    w      : (M,) array     -- frequency grid for numerical integration

    Returns
    -------
    wj : (N,) array  -- oscillator frequencies
    cj : (N,) array  -- coupling coefficients
    """
    dw = w[1] - w[0]
    Jw = J_func(w)
    Fw = np.zeros(len(w))
    for iw in range(len(w)):
        Fw[iw] = (1.0 / np.pi) * np.sum(Jw[:iw] / w[:iw]) * dw
    lambs = Fw[-1]
    wj = np.zeros(N)
    cj = np.zeros(N)
    for i in range(N):
        j = i + 1
        wj[i] = w[np.argmin(np.abs(Fw - ((j - 0.5) / N) * lambs))]
        cj[i] = wj[i] * (2.0 * lambs / N)**0.5
    return wj, cj


# ================================================================ #
#  Model construction (runs once at import time)
# ================================================================ #

# ---- Matter parameters ---- #
_Nm   = 4
_wb   = 1000.0 * cmn1
_Eb   = 2250.0 * cmn1
_T    = 300.0
_beta = 1.0 / (kB * _T)
_nR   = 2001
_Rmax_grid = 100.0
_Rgrid = np.linspace(-_Rmax_grid, _Rmax_grid, _nR)

# ---- Cavity frequency (enters effective bath) ---- #
_wc  = 1190.0 * cmn1       # default; overridable via $wc

# ---- Light-matter coupling ---- #
_eta = 1.25e-3

# ---- Bath parameters ---- #
_Nmbath = 300
_Ncbath = 300
_M_mass = 1.0
_etav   = 0.1
_gammv  = 200.0 * cmn1
_gammc  = 1000.0 * cmn1
_tauc   = 2000.0 * fs
_wigner = True

# Derived
_lambv    = 0.5 * _etav * _wb * _gammv
_lambc    = (1.0 - np.exp(-_beta * _wc)) * (_wc**2 + _gammc**2) / (2.0 * _tauc * _gammc)
_upper_wv = 10.0 * _gammv
_upper_wc = 3.0 * _gammc


# ---------- Matter Hamiltonian (DVR) ---------- #
_V_dvr = _double_well_potential(_Rgrid, _wb, _Eb)
_Hmij_dvr = _kinetic_energy_dvr(_Rgrid) + np.diag(_V_dvr)
_Ei, _Um = np.linalg.eigh(_Hmij_dvr)

# Position and Heaviside operators in energy eigenbasis
_Rij = _Um.T @ np.diag(_Rgrid) @ _Um
_hv_grid = np.array([1.0 if _Rgrid[i] < 0 else 0.0 for i in range(_nR)])
_hvij = _Um.T @ np.diag(_hv_grid) @ _Um

# Rotate lowest two states to localised (L / R) basis
_Urot = np.identity(_nR)
_Urot[0, 0] =  1.0 / np.sqrt(2)
_Urot[0, 1] =  1.0 / np.sqrt(2)
_Urot[1, 0] = -1.0 / np.sqrt(2)
_Urot[1, 1] =  1.0 / np.sqrt(2)

_Hmij = _Urot.T @ np.diag(_Ei) @ _Urot
_Rij  = _Urot.T @ _Rij  @ _Urot
_hvij = _Urot.T @ _hvij @ _Urot

# Truncate to _Nm states
_Rij  = _Rij[:_Nm, :_Nm]
_hvij = _hvij[:_Nm, :_Nm]
_Hmij = _Hmij[:_Nm, :_Nm]


# ---------- Initial state ---------- #
_initMatter = 0 if _Rij[0, 0] < 0 else 1
_Psi_m0 = np.zeros(_Nm)
_Psi_m0[_initMatter] = 1.0
_initState = _initMatter
_NStates = _Nm


# ---------- Effective spectral density and bath discretisation ---------- #

def _Jeff(w):
    """Effective spectral density from tracing out the photon + cavity loss."""
    return (
        2.0 * (1.0 / _tauc) * _eta**2 * _wc**3 * w
        / ((_wc**2 - w**2)**2 + w**2 / _tauc**2)
    )


# Molecular bath (Debye)
_wk_mol, _ck_mol = _discretize_debye(_Nmbath, _upper_wv, _lambv, _gammv)

# Effective cavity bath (general spectral density)
_w_grid = np.linspace(1e-16, _upper_wc, 200000)
_wk_eff, _ck_eff = _discretize_general(_Ncbath, _Jeff, _w_grid)

# Concatenate into a single bath
_NBath = _Nmbath + _Ncbath
_wk = np.concatenate((_wk_mol, _wk_eff))
_ck = np.concatenate((_ck_mol, _ck_eff))

# Timestep from fastest bath mode
_dtN = 2.0 * np.pi / (10.0 * np.max(_wk))

# Simulation timing
_finalT  = 50.0 * ps
_Nframes = 5000
_NSteps  = int(_finalT / _dtN)
_NSkip   = max(1, _NSteps // _Nframes)


# ================================================================ #
#  parameters class  (standard model interface)
# ================================================================ #

class parameters():
    """
    VSC single-molecule model with effective bath.

    System dimension: Nm = NStates (no explicit photon).
    Bath DOFs: Nmbath (molecular) + Ncbath (effective cavity) = ndof.
    """
    # Unit conversions
    au   = au
    ps   = ps
    fs   = fs
    cm   = cm
    kB   = kB
    cmn1 = cmn1

    # Simulation
    NSteps    = _NSteps
    NTraj     = 10000
    dtN       = _dtN
    dtE       = _dtN / 10.0
    NStates   = _NStates
    M         = _M_mass
    initState = _initState
    nskip     = _NSkip
    ndof      = _NBath

    # Model-specific
    Nm     = _Nm
    wc     = _wc
    eta    = _eta
    wigner = _wigner
    beta   = _beta
    T      = _T

    # Bath
    Nmbath = _Nmbath
    Ncbath = _Ncbath
    NBath  = _NBath
    wb     = _wb
    Eb     = _Eb
    etav   = _etav
    gammv  = _gammv
    gammc  = _gammc
    tauc   = _tauc
    lambv  = _lambv
    lambc  = _lambc

    # Operators
    hvij    = _hvij       # (NStates, NStates) Heaviside operator
    Rij_op  = _Rij        # (NStates, NStates) matter position operator
    Hmij    = _Hmij       # (NStates, NStates) bare matter Hamiltonian


# ================================================================ #
#  Standard model functions
# ================================================================ #

def Hel(R):
    """
    Electronic Hamiltonian H_el(R) including all bath couplings
    and the counter-term.

    All bath modes (molecular + effective cavity) couple through
    the matter position operator Rij.

    Parameters
    ----------
    R : ndarray, shape (NBath,)
        Classical bath coordinates.

    Returns
    -------
    Vij : ndarray, shape (NStates, NStates)
    """
    Vij = _Hmij + _Rij * np.sum(_ck * R)
    # Counter-term
    Vij += 0.5 * np.sum(_ck**2 / _wk**2) * (_Rij @ _Rij)
    return Vij


def dHel(R):
    """
    State-dependent gradient of H_el w.r.t. bath coordinates.

    Returns
    -------
    dHij : ndarray, shape (NStates, NStates, NBath)
    """
    return _Rij[:, :, np.newaxis] * _ck[np.newaxis, np.newaxis, :]


def dHel0(R):
    """
    State-independent gradient (harmonic-bath restoring force).

    Returns
    -------
    dH0 : ndarray, shape (NBath,)
    """
    return _wk**2 * R


def initR():
    """
    Sample initial bath positions and momenta.

    Returns
    -------
    R : ndarray, shape (NBath,)
    P : ndarray, shape (NBath,)
    """
    # Equilibrium displacement for initial electronic state
    R0 = -_ck * (_Psi_m0.T @ _Rij @ _Psi_m0) / _wk**2
    P0 = np.zeros(_NBath)

    if _wigner:
        sigP = np.sqrt(_wk / (2.0 * np.tanh(0.5 * _beta * _wk)))
        sigR = sigP / _wk
    else:
        sigP = np.sqrt(1.0 / _beta) * np.ones(_NBath)
        sigR = 1.0 / np.sqrt(_beta * _wk**2)

    R = np.zeros(_NBath)
    P = np.zeros(_NBath)
    for d in range(_NBath):
        R[d] = rn.normal() * sigR[d] + R0[d]
        P[d] = rn.normal() * sigP[d] + P0[d]
    return R, P
