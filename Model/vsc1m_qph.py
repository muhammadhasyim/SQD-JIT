"""
Model/vsc1m_qph.py

Single-molecule vibrational strong coupling (VSC) model.

The double-well matter system and a single cavity photon mode are treated
quantum mechanically in a tensor product basis (Nm matter states x Nf
photon Fock states), while molecular and cavity baths are treated as
classical harmonic oscillators coupled linearly.

An optional polaron (Mulliken-Hush) transformation diagonalises the
matter position operator and dresses the photon operators accordingly.

References
----------
* Lindoy, Mandal, Reichman, Nature Commun. 14, 2733 (2023).
"""

import numpy as np
from numpy import kron
from numpy import random as rn
import math
from scipy.special import hermite

# ================================================================ #
#  Unit conversions (atomic units)
# ================================================================ #
au   = 27.2113961          # 1 a.u. -> eV
ps   = 41341.374575751     # 1 ps -> a.u.
fs   = 41.34136            # 1 fs -> a.u.
cm   = 1 / 4.556335e-6    # 219474.63  (Hz -> cm-1)
kB   = 3.166829e-6         # Boltzmann constant (a.u.)
eV   = 1.0 / 27.2114       # 1 eV -> a.u.
cmn1 = 1.0 / 219474.63     # 1 cm-1 -> a.u.


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


def _photon_wavefunction(x, w, n):
    """Normalised harmonic-oscillator wavefunction psi_n(x; w)."""
    cons1 = 1.0 / ((2.0**n) * math.factorial(n))**0.5
    cons2 = (w / math.pi)**0.25
    return cons1 * cons2 * np.exp(-w * x**2 / 2.0) * hermite(n)(np.sqrt(w) * x)


def _discretize_debye(N, wmax, lamb, gamm):
    """
    Discretise a Debye spectral density J(w) = 2*lamb*gamm*w/(w^2+gamm^2)
    into N harmonic oscillators using the cumulative-density method.

    Returns
    -------
    wj : (N,) array  -- oscillator frequencies
    cj : (N,) array  -- system-bath coupling coefficients
    """
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


# ================================================================ #
#  Model construction (runs once at import time)
# ================================================================ #

# ---- Matter parameters ---- #
_Nm   = 4                   # matter eigenstates retained
_wb   = 1000.0 * cmn1       # barrier frequency (a.u.)
_Eb   = 2250.0 * cmn1       # barrier height (a.u.)
_T    = 300.0               # temperature (K)
_beta = 1.0 / (kB * _T)    # inverse temperature (a.u.)
_nR   = 2001                # DVR grid points
_Rmax_grid = 100.0          # DVR grid extent (a.u.)
_Rgrid = np.linspace(-_Rmax_grid, _Rmax_grid, _nR)

# ---- Photon parameters ---- #
_Nf  = 2                    # photon Fock states
_wc  = 1190.0 * cmn1        # cavity frequency (a.u.) -- default

# ---- Light-matter coupling ---- #
_eta = 2.5e-3               # coupling constant (a.u.)
_polaron = True              # use polaron (Mulliken-Hush) transform

# ---- Bath parameters ---- #
_Nmbath = 300                # molecular bath modes
_Ncbath = 300                # cavity bath modes
_M_mass = 1.0                # nuclear mass (a.u.)
_etav   = 0.1               # molecular bath friction (dimensionless)
_gammv  = 200.0 * cmn1      # molecular bath characteristic freq (a.u.)
_gammc  = 1000.0 * cmn1     # cavity bath characteristic freq (a.u.)
_tauc   = 2000.0 * fs       # cavity lifetime (a.u.)
_wigner = True               # Wigner sampling for bath modes

# Derived bath reorganisation energies
_lambv = 0.5 * _etav * _wb * _gammv
_lambc = (1.0 - np.exp(-_beta * _wc)) * (_wc**2 + _gammc**2) / (2.0 * _tauc * _gammc)

# Upper frequency cutoffs
_upper_wv = 10.0 * _gammv
_upper_wc = 3.0 * _gammc


# ---------- Matter Hamiltonian (DVR) ---------- #
_V_dvr = _double_well_potential(_Rgrid, _wb, _Eb)
_Hmij_dvr = _kinetic_energy_dvr(_Rgrid) + np.diag(_V_dvr)
_Ei, _Um = np.linalg.eigh(_Hmij_dvr)

# Position and Heaviside operators in the energy eigenbasis
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
_Rij  = _Urot.T @ _Rij @ _Urot
_hvij = _Urot.T @ _hvij @ _Urot

# Truncate to _Nm states
_Rij  = _Rij[:_Nm, :_Nm]
_hvij = _hvij[:_Nm, :_Nm]
_Hmij = _Hmij[:_Nm, :_Nm]


# ---------- Photon Hamiltonian ---------- #
_Hp = np.diag((np.arange(_Nf) + 0.5) * _wc)

# Creation operator (Nf x Nf)
_adag = np.zeros((_Nf, _Nf))
for _m in range(1, _Nf):
    _adag[_m, _m - 1] = np.sqrt(_m)
_adag = _adag.T

# Photon position operator
_Qcij_small = np.sqrt(0.5 / _wc) * (_adag + _adag.conj().T)


# ---------- Initial state ---------- #
# Matter: localised on the left (R < 0)
_initMatter = 0 if _Rij[0, 0] < 0 else 1
_Psi_m0 = np.zeros(_Nm)
_Psi_m0[_initMatter] = 1.0

# Photon: Boltzmann sampling at temperature T
_Eph = np.diag(_Hp)
_p_boltz = np.exp(-_beta * _Eph)
_p_boltz /= np.sum(_p_boltz)
_initPhoton = int(np.random.choice(np.arange(_Nf), p=_p_boltz))
_Psi_p0 = np.zeros(_Nf)
_Psi_p0[_initPhoton] = 1.0

# Full initial state = |matter> x |photon>
_Psi0 = kron(_Psi_m0, _Psi_p0)
_initState = int(np.argmin(np.abs(_Psi0 - 1.0)))


# ---------- Polaron (Mulliken-Hush) transform ---------- #
_Rii, _Umu = np.linalg.eigh(_Rij)   # diagonalise position operator
_Rmij = np.copy(_Rij)               # save un-transformed copy

if _polaron:
    _Hmij = _Umu.T @ _Hmij @ _Umu
    _Rij  = _Umu.T @ _Rij  @ _Umu
    _hvij = _Umu.T @ _hvij @ _Umu
    _Rmij = np.copy(_Rij)


# ---------- Expand to tensor-product space ---------- #
_Ip = np.identity(_Nf)
_Im = np.identity(_Nm)

_Rij_full  = kron(_Rij, _Ip)           # matter position (NStates x NStates)
_hvij_full = kron(_hvij, _Ip)          # Heaviside operator
_Qcij_full = kron(_Im, _Qcij_small)    # photon position

# Polaron basis-change matrix
if _polaron:
    _Upolaron = kron(_Umu, _Ip)
    _Psi0     = np.conj(_Upolaron).T @ _Psi0
    _Qcij_full += _eta * _Rij_full * (2.0 / _wc)**0.5
else:
    _Upolaron = kron(_Im, _Ip)         # identity

_NStates = _Nm * _Nf


# ---------- Full quantum Hamiltonian H_0 ---------- #
_Hij = np.zeros((_NStates, _NStates))

if _polaron:
    _dr_pol = 0.001
    _rc_pol = np.arange(-_Rmax_grid, _Rmax_grid, _dr_pol)

for _i in range(_NStates):
    _a = _i // _Nf
    _m_idx = _i % _Nf
    for _j in range(_i, _NStates):
        _b = _j // _Nf
        _n_idx = _j % _Nf

        if _polaron:
            # Franck-Condon overlap integral
            _qc0_a = -_eta * _Rii[_a] * (2.0 / _wc)**0.5
            _qc0_b = -_eta * _Rii[_b] * (2.0 / _wc)**0.5
            _smn = np.sum(
                _photon_wavefunction(_rc_pol - _qc0_a, _wc, _m_idx) *
                _photon_wavefunction(_rc_pol - _qc0_b, _wc, _n_idx)
            ) * _dr_pol
            _Hij[_i, _j]  = _Hmij[_a, _b] * _smn
            _Hij[_i, _j] += _Hp[_m_idx, _n_idx] * (_a == _b)
        else:
            _Hij[_i, _j]  = _Hmij[_a, _b] * (_m_idx == _n_idx)
            _Hij[_i, _j] += _Hp[_m_idx, _n_idx] * (_a == _b)
            # Light-matter interaction
            _Hij[_i, _j] += (
                _Rmij[_a, _b]
                * (_adag.conj().T[_m_idx, _n_idx] + _adag[_m_idx, _n_idx])
                * _wc * _eta
            )
            # Dipole self-energy
            _Hij[_i, _j] += (
                (_Rmij @ _Rmij)[_a, _b]
                * (_m_idx == _n_idx)
                * _wc * _eta**2
            )

        _Hij[_j, _i] = np.conj(_Hij[_i, _j])


# ---------- Discretise baths ---------- #
_wk,  _ck  = _discretize_debye(_Nmbath, _upper_wv, _lambv, _gammv)
_wph, _cph = _discretize_debye(_Ncbath, _upper_wc, _lambc, _gammc)
_NBath = _Nmbath + _Ncbath

# Timestep from fastest bath frequency
_dtN = 2.0 * np.pi / (10.0 * np.max(_wph))

# Simulation timing
_finalT   = 2.0 * ps
_Nframes  = 5000
_NSteps   = int(_finalT / _dtN)
_NSkip    = max(1, _NSteps // _Nframes)


# ================================================================ #
#  parameters class  (standard model interface)
# ================================================================ #

class parameters():
    """
    VSC single-molecule model: quantum photon + classical baths.

    System dimension: Nm x Nf = NStates.
    Bath DOFs: Nmbath (molecular) + Ncbath (cavity) = ndof.
    """
    # Unit conversions (convenience)
    au   = au
    ps   = ps
    fs   = fs
    cm   = cm
    kB   = kB
    cmn1 = cmn1

    # Simulation
    NSteps    = _NSteps
    NTraj     = 100000
    dtN       = _dtN
    dtE       = _dtN / 10.0
    NStates   = _NStates
    M         = _M_mass
    initState = _initState
    nskip     = _NSkip
    ndof      = _NBath

    # Model-specific
    Nm      = _Nm
    Nf      = _Nf
    wc      = _wc
    eta     = _eta
    polaron = _polaron
    wigner  = _wigner
    beta    = _beta
    T       = _T

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

    # Operators in the working basis
    Upolaron = _Upolaron    # (NStates, NStates) basis rotation
    hvij     = _hvij_full   # (NStates, NStates) Heaviside operator
    Rij_op   = _Rij_full    # (NStates, NStates) matter position
    Qcij_op  = _Qcij_full   # (NStates, NStates) photon position
    Hij_bare = _Hij          # (NStates, NStates) bare Hamiltonian


# ================================================================ #
#  Standard model functions
# ================================================================ #

def Hel(R):
    """
    Electronic Hamiltonian H_el(R) including system-bath coupling
    and counter-term.

    Parameters
    ----------
    R : ndarray, shape (NBath,)
        Classical bath coordinates.  R[:Nmbath] are molecular-bath
        modes, R[Nmbath:] are cavity-bath modes.

    Returns
    -------
    Vij : ndarray, shape (NStates, NStates)
    """
    Vij = (
        _Hij
        + _Rij_full  * np.sum(_ck  * R[:_Nmbath])
        + _Qcij_full * np.sum(_cph * R[_Nmbath:])
    )
    # Counter-terms
    Vij += 0.5 * np.sum(_ck**2  / _wk**2)  * (_Rij_full  @ _Rij_full)
    Vij += 0.5 * np.sum(_cph**2 / _wph**2) * (_Qcij_full @ _Qcij_full)
    return Vij


def dHel(R):
    """
    State-dependent gradient of H_el w.r.t. bath coordinates.

    Returns
    -------
    dHij : ndarray, shape (NStates, NStates, NBath)
    """
    dHij_mol = _Rij_full[:, :, np.newaxis]  * _ck[np.newaxis, np.newaxis, :]
    dHij_cav = _Qcij_full[:, :, np.newaxis] * _cph[np.newaxis, np.newaxis, :]
    return np.concatenate((dHij_mol, dHij_cav), axis=2)


def dHel0(R):
    """
    State-independent gradient (harmonic-bath restoring force).

    Returns
    -------
    dH0 : ndarray, shape (NBath,)
    """
    dH0_mol = _wk**2  * R[:_Nmbath]
    dH0_cav = _wph**2 * R[_Nmbath:]
    return np.concatenate((dH0_mol, dH0_cav))


def initR():
    """
    Sample initial bath positions and momenta.

    Uses Wigner or classical thermal sampling, displaced to the
    equilibrium position for the initial electronic state.

    Returns
    -------
    R : ndarray, shape (NBath,)
    P : ndarray, shape (NBath,)
    """
    omegas    = np.concatenate((_wk, _wph))
    couplings = np.concatenate((_ck, _cph))

    # Equilibrium displacement for initial electronic state
    R0 = -couplings * (_Psi0.T @ _Rij_full @ _Psi0) / omegas**2
    P0 = np.zeros(_NBath)

    if _wigner:
        sigP = np.sqrt(omegas / (2.0 * np.tanh(0.5 * _beta * omegas)))
        sigR = sigP / omegas
    else:
        sigP = np.sqrt(1.0 / _beta) * np.ones(_NBath)
        sigR = 1.0 / np.sqrt(_beta * omegas**2)

    R = np.zeros(_NBath)
    P = np.zeros(_NBath)
    for d in range(_NBath):
        R[d] = rn.normal() * sigR[d] + R0[d]
        P[d] = rn.normal() * sigP[d] + P0[d]
    return R, P
