#!/usr/bin/env python
"""
Comprehensive visual validation & benchmark for ALL JIT-compiled methods.

For each method (pldm, mfe, sqc, zpesqc, spinlsc, spinpldm, mash):
  1. Runs original Python and JIT-compiled versions with the SAME random seed.
  2. Plots population overlay (original solid, JIT dashed).
  3. Plots residual (original - JIT) vs time.
  4. Benchmarks JIT speedup (NTraj=200).
  5. Produces summary bar chart of speedups.

Random numbers are pre-generated in the EXACT same consumption order
as the original code to enable bitwise-identical comparison.

Output: tests/output/<method>_population.png
        tests/output/<method>_residual.png
        tests/output/benchmark_all_methods.png
"""

import sys
import os
import time
import random as pyrandom

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUTDIR = os.path.join(ROOT, 'tests', 'output')
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Imports -- original methods
# ---------------------------------------------------------------------------
from Model import spinBoson as orig_model
from Method import pldm as orig_pldm
from Method import mfe as orig_mfe
from Method import sqc as orig_sqc
from Method import zpesqc as orig_zpesqc
from Method import spinlsc as orig_spinlsc
from Method import spinpldm as orig_spinpldm
from Method import mash as orig_mash

# ---------------------------------------------------------------------------
# Imports -- JIT methods & model
# ---------------------------------------------------------------------------
from ModelJIT.spinBoson import hel, dhel, dhel0, get_model_params
from MethodJIT.pldm import make_pldm_kernel
from MethodJIT.mfe import make_mfe_kernel
from MethodJIT.sqc import make_sqc_kernel
from MethodJIT.zpesqc import make_zpesqc_kernel
from MethodJIT.spinlsc import make_spinlsc_kernel
from MethodJIT.spinpldm import make_spinpldm_kernel
from MethodJIT.mash import make_mash_kernel
from MethodJIT.unsmash import make_unsmash_kernel

# ---------------------------------------------------------------------------
# Model parameters (spin-boson model 3)
# ---------------------------------------------------------------------------
mp = get_model_params(3)
NStates = mp['NStates']
ndof = mp['ndof']
SEED = 42

# Time axis for plots
n_skip = mp['NSteps'] // mp['nskip']
t_axis = np.arange(n_skip) * mp['nskip'] * mp['dtN']

# ---------------------------------------------------------------------------
# Helper: create original-style parameters object
# ---------------------------------------------------------------------------
def make_orig_params(NTraj):
    """Return a parameters object wired to the original spinBoson model."""
    par = orig_model.parameters
    par.NTraj = NTraj
    par.dHel = orig_model.dHel
    par.dHel0 = orig_model.dHel0
    par.initR = orig_model.initR
    par.Hel = orig_model.Hel
    par.SEED = SEED
    return par


# ---------------------------------------------------------------------------
# Pre-generation helpers  (match original RNG consumption order exactly)
# ---------------------------------------------------------------------------
def pre_gen_nuclear(NTraj):
    """Generate R, P in same interleaved order as original initR().

    Original: for d in ndof: R[d]=normal(), P[d]=normal()
    """
    R = np.empty((NTraj, ndof))
    P = np.empty((NTraj, ndof))
    for itraj in range(NTraj):
        for d in range(ndof):
            R[itraj, d] = np.random.normal()
            P[itraj, d] = np.random.normal()
    return R, P


# ===================================================================
#  PLDM  (focused -- no mapping randoms)
# ===================================================================
def run_orig_pldm(NTraj):
    par = make_orig_params(NTraj)
    par.stype = "focused"
    return orig_pldm.runTraj(par)


def run_jit_pldm(NTraj):
    kernel = make_pldm_kernel(hel, dhel, dhel0)
    np.random.seed(SEED)
    R_rand, P_rand = pre_gen_nuclear(NTraj)
    map_rand = np.empty((NTraj, 4, NStates))  # unused placeholder
    return kernel(NTraj, mp['NSteps'], NStates, mp['nskip'], mp['EStep'],
                  mp['dtN'], mp['M_mass'], mp['initState'], True,
                  R_rand, P_rand, mp['sigR'], mp['sigP'], map_rand,
                  mp['c'], mp['epsilon'], mp['Delta'], mp['omega'])


# ===================================================================
#  MFE  (no mapping randoms -- ci[initState]=1 deterministic)
# ===================================================================
def run_orig_mfe(NTraj):
    par = make_orig_params(NTraj)
    return orig_mfe.runTraj(par)


def run_jit_mfe(NTraj):
    kernel = make_mfe_kernel(hel, dhel, dhel0)
    np.random.seed(SEED)
    R_rand, P_rand = pre_gen_nuclear(NTraj)
    return kernel(NTraj, mp['NSteps'], NStates, mp['nskip'], mp['EStep'],
                  mp['dtN'], mp['M_mass'], mp['initState'],
                  R_rand, P_rand, mp['sigR'], mp['sigP'],
                  mp['c'], mp['epsilon'], mp['Delta'], mp['omega'])


# ===================================================================
#  SQC  (square window -- mapping uses np.random.random)
# ===================================================================
def run_orig_sqc(NTraj):
    par = make_orig_params(NTraj)
    par.stype = "square"
    return orig_sqc.runTraj(par)


def run_jit_sqc(NTraj):
    kernel = make_sqc_kernel(hel, dhel, dhel0)
    np.random.seed(SEED)
    R_rand = np.empty((NTraj, ndof))
    P_rand = np.empty((NTraj, ndof))
    map_rand = np.empty((NTraj, 2, NStates))
    for itraj in range(NTraj):
        for d in range(ndof):
            R_rand[itraj, d] = np.random.normal()
            P_rand[itraj, d] = np.random.normal()
        # initMapping: eta = 2*gamma*random(N), theta = 2*pi*random(N)
        map_rand[itraj, 0, :] = np.random.random(NStates)
        map_rand[itraj, 1, :] = np.random.random(NStates)
    return kernel(NTraj, mp['NSteps'], NStates, mp['nskip'], mp['EStep'],
                  mp['dtN'], mp['M_mass'], mp['initState'], True,
                  R_rand, P_rand, mp['sigR'], mp['sigP'], map_rand,
                  mp['c'], mp['epsilon'], mp['Delta'], mp['omega'])


# ===================================================================
#  ZPESQC  (square window + per-state gamma0)
# ===================================================================
def run_orig_zpesqc(NTraj):
    par = make_orig_params(NTraj)
    par.stype = "square"
    return orig_zpesqc.runTraj(par)


def run_jit_zpesqc(NTraj):
    kernel = make_zpesqc_kernel(hel, dhel, dhel0)
    np.random.seed(SEED)
    R_rand = np.empty((NTraj, ndof))
    P_rand = np.empty((NTraj, ndof))
    map_rand = np.empty((NTraj, 2, NStates))
    for itraj in range(NTraj):
        for d in range(ndof):
            R_rand[itraj, d] = np.random.normal()
            P_rand[itraj, d] = np.random.normal()
        map_rand[itraj, 0, :] = np.random.random(NStates)
        map_rand[itraj, 1, :] = np.random.random(NStates)
    return kernel(NTraj, mp['NSteps'], NStates, mp['nskip'], mp['EStep'],
                  mp['dtN'], mp['M_mass'], mp['initState'], True,
                  R_rand, P_rand, mp['sigR'], mp['sigP'], map_rand,
                  mp['c'], mp['epsilon'], mp['Delta'], mp['omega'])


# ===================================================================
#  spinLSC  (complex zF, phi from random.random())
# ===================================================================
def run_orig_spinlsc(NTraj):
    pyrandom.seed(SEED)          # seed Python random for phi angles
    par = make_orig_params(NTraj)
    return orig_spinlsc.runTraj(par)


def run_jit_spinlsc(NTraj):
    kernel = make_spinlsc_kernel(hel, dhel, dhel0)
    np.random.seed(SEED)
    pyrandom.seed(SEED)
    R_rand = np.empty((NTraj, ndof))
    P_rand = np.empty((NTraj, ndof))
    phi_rand = np.empty((NTraj, NStates))
    for itraj in range(NTraj):
        for d in range(ndof):
            R_rand[itraj, d] = np.random.normal()
            P_rand[itraj, d] = np.random.normal()
        # initMapping: phiF = random.random() per state (Python RNG)
        for s in range(NStates):
            phi_rand[itraj, s] = pyrandom.random()
    return kernel(NTraj, mp['NSteps'], NStates, mp['nskip'], mp['EStep'],
                  mp['dtN'], mp['M_mass'], mp['initState'],
                  R_rand, P_rand, mp['sigR'], mp['sigP'], phi_rand,
                  mp['c'], mp['epsilon'], mp['Delta'], mp['omega'])


# ===================================================================
#  spinPLDM  (focused: FB=[[init,init]], W=[1])
# ===================================================================
def run_orig_spinpldm(NTraj):
    pyrandom.seed(SEED)
    par = make_orig_params(NTraj)
    par.stype = "focused"
    return orig_spinpldm.runTraj(par)


def run_jit_spinpldm(NTraj):
    kernel = make_spinpldm_kernel(hel, dhel, dhel0)
    initState = mp['initState']
    FB_arr = np.array([[initState, initState]], dtype=np.int64)
    W_arr = np.array([1.0])
    nFB = 1

    np.random.seed(SEED)
    pyrandom.seed(SEED)
    R_rand = np.empty((NTraj, ndof))
    P_rand = np.empty((NTraj, ndof))
    phi_rand = np.empty((NTraj, nFB, 2, NStates))
    for itraj in range(NTraj):
        for d in range(ndof):
            R_rand[itraj, d] = np.random.normal()
            P_rand[itraj, d] = np.random.normal()
        # initMapping: phiF, phiB per state (Python RNG, interleaved)
        for s in range(NStates):
            phi_rand[itraj, 0, 0, s] = pyrandom.random()   # phiF
            phi_rand[itraj, 0, 1, s] = pyrandom.random()   # phiB
    return kernel(NTraj, mp['NSteps'], NStates, mp['nskip'], mp['EStep'],
                  mp['dtN'], mp['M_mass'], initState,
                  R_rand, P_rand, mp['sigR'], mp['sigP'], phi_rand,
                  FB_arr, W_arr,
                  mp['c'], mp['epsilon'], mp['Delta'], mp['omega'])


# ===================================================================
#  MASH  (surface hopping, random phase from np.random.random)
# ===================================================================
def run_orig_mash(NTraj):
    par = make_orig_params(NTraj)
    return orig_mash.runTraj(par)


def run_jit_mash(NTraj):
    kernel = make_mash_kernel(hel, dhel, dhel0)
    np.random.seed(SEED)
    R_rand = np.empty((NTraj, ndof))
    P_rand = np.empty((NTraj, ndof))
    uni_rand = np.empty((NTraj, NStates))
    for itraj in range(NTraj):
        for d in range(ndof):
            R_rand[itraj, d] = np.random.normal()
            P_rand[itraj, d] = np.random.normal()
        for s in range(NStates):
            uni_rand[itraj, s] = np.random.random()
    return kernel(NTraj, mp['NSteps'], NStates, mp['nskip'],
                  mp['dtN'], mp['M_mass'], mp['initState'], 10,
                  R_rand, P_rand, mp['sigR'], mp['sigP'], uni_rand,
                  mp['c'], mp['epsilon'], mp['Delta'], mp['omega'])


# ===================================================================
#  unSMASH  (uncoupled spheres MASH -- JIT only, no original Python)
#  For NStates=2 it rigorously recovers MASH, so we compare vs MASH.
# ===================================================================
def run_jit_unsmash(NTraj):
    kernel = make_unsmash_kernel(hel, dhel, dhel0)
    np.random.seed(SEED)
    R_rand = np.empty((NTraj, ndof))
    P_rand = np.empty((NTraj, ndof))
    for itraj in range(NTraj):
        for d in range(ndof):
            R_rand[itraj, d] = np.random.normal()
            P_rand[itraj, d] = np.random.normal()
    # Bloch sphere random numbers: (NTraj, NStates, 2)
    #   [:, :, 0] -> u for S_z = u (uniform on upper hemisphere)
    #   [:, :, 1] -> v for phi = 2*pi*v
    sphere_rand = np.random.random((NTraj, NStates, 2))
    # Initial active state: uniform over {0, ..., NStates-1}
    acst_rand = np.random.random(NTraj)
    return kernel(NTraj, mp['NSteps'], NStates, mp['nskip'],
                  mp['dtN'], mp['M_mass'], mp['initState'], 10,
                  R_rand, P_rand, mp['sigR'], mp['sigP'],
                  sphere_rand, acst_rand,
                  mp['c'], mp['epsilon'], mp['Delta'], mp['omega'])


# ===================================================================
#  Method registry
# ===================================================================
METHODS = [
    ('pldm',     run_orig_pldm,     run_jit_pldm),
    ('mfe',      run_orig_mfe,      run_jit_mfe),
    ('sqc',      run_orig_sqc,      run_jit_sqc),
    ('zpesqc',   run_orig_zpesqc,   run_jit_zpesqc),
    ('spinlsc',  run_orig_spinlsc,  run_jit_spinlsc),
    ('spinpldm', run_orig_spinpldm, run_jit_spinpldm),
    ('mash',     run_orig_mash,     run_jit_mash),
]

# unSMASH has no original Python equivalent; compare vs MASH instead
UNSMASH_VS_MASH = ('unsmash', run_jit_mash, run_jit_unsmash)


# ===================================================================
#  Plotting helpers
# ===================================================================
def plot_population(name, rho_orig, rho_jit, NTraj, max_err):
    """Population overlay: original (solid) vs JIT (dashed)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in range(NStates):
        label_o = f'P{s+1}{s+1} original'
        label_j = f'P{s+1}{s+1} JIT'
        ax.plot(t_axis, rho_orig[s, s, :].real / NTraj,
                '-', lw=2, label=label_o)
        ax.plot(t_axis, rho_jit[s, s, :].real / NTraj,
                '--', lw=2, label=label_j)
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Population')
    ax.set_title(f'{name.upper()} -- Population dynamics  '
                 f'(NTraj={NTraj}, max|diff|={max_err:.2e})')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTDIR, f'{name}_population.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def plot_residual(name, rho_orig, rho_jit, NTraj, max_err):
    """Residual plot: (original - JIT) / NTraj vs time."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for s in range(NStates):
        diff = (rho_orig[s, s, :].real - rho_jit[s, s, :].real) / NTraj
        ax.plot(t_axis, diff, '-', lw=1.5, label=f'P{s+1}{s+1} residual')
    ax.axhline(0, color='k', ls=':', lw=0.8)
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Original - JIT')
    ax.set_title(f'{name.upper()} -- Residual  (max|diff|={max_err:.2e})')
    ax.legend(loc='best', fontsize=9)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTDIR, f'{name}_residual.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def plot_benchmark_summary(results):
    """Bar chart of speedups for all methods."""
    names = [r[0] for r in results]
    speedups = [r[1] for r in results]
    t_orig = [r[2] for r in results]
    t_jit = [r[3] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Speedup bars
    bars = ax1.bar(names, speedups, color='steelblue', edgecolor='k', alpha=0.85)
    for bar, sp in zip(bars, speedups):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{sp:.1f}x', ha='center', va='bottom', fontsize=9,
                 fontweight='bold')
    ax1.set_ylabel('Speedup (original / JIT)')
    ax1.set_title('JIT Speedup per Method (NTraj=200)')
    ax1.grid(axis='y', alpha=0.3)

    # Runtime comparison
    x = np.arange(len(names))
    w = 0.35
    ax2.bar(x - w / 2, t_orig, w, label='Original', color='salmon',
            edgecolor='k', alpha=0.85)
    ax2.bar(x + w / 2, t_jit, w, label='JIT', color='steelblue',
            edgecolor='k', alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel('Wall-clock time (s)')
    ax2.set_title('Runtime Comparison (NTraj=200)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTDIR, 'benchmark_all_methods.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'\nSaved {path}')


# ===================================================================
#  Main
# ===================================================================
def main():
    NTraj_test = 200       # for correctness comparison
    NTraj_bench = 200      # for benchmark timing

    print('=' * 70)
    print('  JIT vs Original -- Visual Validation & Benchmark')
    print(f'  Model: spinBoson (M=3), NStates={NStates}, ndof={ndof}')
    print(f'  NSteps={mp["NSteps"]}, nskip={mp["nskip"]}, dtN={mp["dtN"]}')
    print(f'  Seed={SEED}')
    print('=' * 70)

    benchmark_results = []

    for name, run_orig, run_jit in METHODS:
        print(f'\n--- {name.upper()} ---')

        # ---- Correctness: run both ----
        print(f'  Running original ({NTraj_test} traj)...', end=' ', flush=True)
        t0 = time.perf_counter()
        rho_orig = run_orig(NTraj_test)
        t_orig = time.perf_counter() - t0
        print(f'{t_orig:.2f}s')

        print(f'  Running JIT      ({NTraj_test} traj)...', end=' ', flush=True)
        # Warmup (first call compiles)
        _ = run_jit(1)
        t0 = time.perf_counter()
        rho_jit = run_jit(NTraj_test)
        t_jit = time.perf_counter() - t0
        print(f'{t_jit:.2f}s')

        # Compute max residual
        diff = np.abs(rho_orig - rho_jit)
        max_err = np.max(diff) / NTraj_test
        print(f'  max|diff|/NTraj = {max_err:.4e}')

        # ---- Population overlay PNG ----
        plot_population(name, rho_orig, rho_jit, NTraj_test, max_err)

        # ---- Residual PNG ----
        plot_residual(name, rho_orig, rho_jit, NTraj_test, max_err)

        # ---- Benchmark ----
        print(f'  Benchmark: original={t_orig:.3f}s  JIT={t_jit:.3f}s'
              f'  speedup={t_orig / max(t_jit, 1e-6):.1f}x')
        benchmark_results.append(
            (name, t_orig / max(t_jit, 1e-6), t_orig, t_jit))

    # ---- unSMASH vs MASH comparison (different sampling, statistical) ----
    name_u, run_mash_ref, run_unsmash = UNSMASH_VS_MASH
    NTraj_u = 2000  # need more trajectories for statistical comparison
    print(f'\n--- {name_u.upper()} vs MASH (statistical, NTraj={NTraj_u}) ---')

    print(f'  Running MASH reference ({NTraj_u} traj)...', end=' ', flush=True)
    t0 = time.perf_counter()
    rho_mash_ref = run_mash_ref(NTraj_u)
    t_mash = time.perf_counter() - t0
    print(f'{t_mash:.2f}s')

    print(f'  Running unSMASH        ({NTraj_u} traj)...', end=' ', flush=True)
    _ = run_unsmash(1)  # warmup
    t0 = time.perf_counter()
    rho_unsmash = run_unsmash(NTraj_u)
    t_unsmash = time.perf_counter() - t0
    print(f'{t_unsmash:.2f}s')

    # Plot unSMASH vs MASH population overlay
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in range(NStates):
        ax.plot(t_axis, rho_mash_ref[s, s, :].real / NTraj_u,
                '-', lw=2, label=f'P{s+1}{s+1} MASH')
        ax.plot(t_axis, rho_unsmash[s, s, :].real / NTraj_u,
                '--', lw=2, label=f'P{s+1}{s+1} unSMASH')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Population')
    ax.set_title(f'unSMASH vs MASH (NStates={NStates}, NTraj={NTraj_u})')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path_u = os.path.join(OUTDIR, 'unsmash_vs_mash_population.png')
    fig.savefig(path_u, dpi=150)
    plt.close(fig)
    print(f'  Saved {path_u}')

    # Check populations are physical
    for s in range(NStates):
        pop_s = rho_unsmash[s, s, :].real / NTraj_u
        print(f'  P{s+1}{s+1} range: [{pop_s.min():.4f}, {pop_s.max():.4f}]')

    # Compute max difference vs MASH (should be small for NStates=2)
    max_diff = 0.0
    for s in range(NStates):
        diff_s = np.abs(rho_mash_ref[s, s, :].real - rho_unsmash[s, s, :].real)
        max_diff = max(max_diff, np.max(diff_s) / NTraj_u)
    print(f'  max|MASH - unSMASH|/NTraj = {max_diff:.4e}')

    benchmark_results.append(
        (name_u, t_mash / max(t_unsmash, 1e-6), t_mash, t_unsmash))

    # ---- Summary benchmark bar chart ----
    plot_benchmark_summary(benchmark_results)

    print('\n' + '=' * 70)
    print('  All done. PNGs saved to tests/output/')
    print('=' * 70)


if __name__ == '__main__':
    main()
