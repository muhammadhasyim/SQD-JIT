#!/usr/bin/env python
"""
Long-time error accumulation test for ALL JIT-compiled methods.

Runs NSteps=2000 (10x the standard test) with NTraj=200 to see
whether residuals grow over time or stay bounded at machine precision.

Output: tests/output/long_time_residual_all.png
"""
import sys, os, time
import random as pyrandom
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUTDIR = os.path.join(ROOT, 'tests', 'output')
os.makedirs(OUTDIR, exist_ok=True)

# --- Imports ---
from Model import spinBoson as orig_model
from Method import pldm as orig_pldm
from Method import mfe as orig_mfe
from Method import sqc as orig_sqc
from Method import zpesqc as orig_zpesqc
from Method import spinlsc as orig_spinlsc
from Method import spinpldm as orig_spinpldm
from Method import mash as orig_mash
from Method import fsmash as orig_fsmash
from Method import unsmash as orig_unsmash

from ModelJIT.spinBoson import hel, dhel, dhel0, get_model_params
from MethodJIT.pldm import make_pldm_kernel
from MethodJIT.mfe import make_mfe_kernel
from MethodJIT.sqc import make_sqc_kernel
from MethodJIT.zpesqc import make_zpesqc_kernel
from MethodJIT.spinlsc import make_spinlsc_kernel
from MethodJIT.spinpldm import make_spinpldm_kernel
from MethodJIT.mash import make_mash_kernel
from MethodJIT.fsmash import make_fsmash_kernel
from MethodJIT.unsmash import make_unsmash_kernel

# --- Parameters (LONG run) ---
SEED = 42
NTraj = 200
NSTEPS_LONG = 2000   # 10x standard
NSKIP = 10
DTN = 0.01

mp = get_model_params(3)
NStates = mp['NStates']
ndof = mp['ndof']

n_skip = NSTEPS_LONG // NSKIP
t_axis = np.arange(n_skip) * NSKIP * DTN


# --- Original parameter builder ---
def make_orig_params_long():
    par = orig_model.parameters
    par.NTraj = NTraj
    par.NSteps = NSTEPS_LONG
    par.nskip = NSKIP
    par.dtN = DTN
    par.dtE = DTN / 20
    par.dHel = orig_model.dHel
    par.dHel0 = orig_model.dHel0
    par.initR = orig_model.initR
    par.Hel = orig_model.Hel
    par.SEED = SEED
    return par


# --- RNG helpers (same as test_all_methods.py) ---
def pre_gen_nuclear(n):
    R = np.empty((n, ndof))
    P = np.empty((n, ndof))
    for itraj in range(n):
        for d in range(ndof):
            R[itraj, d] = np.random.normal()
            P[itraj, d] = np.random.normal()
    return R, P


# === Method runners (matching test_all_methods.py exactly) ===

def run_orig_pldm():
    par = make_orig_params_long()
    par.stype = "focused"
    return orig_pldm.runTraj(par)

def run_jit_pldm():
    kernel = make_pldm_kernel(hel, dhel, dhel0)
    np.random.seed(SEED)
    R_rand, P_rand = pre_gen_nuclear(NTraj)
    map_rand = np.empty((NTraj, 4, NStates))  # unused placeholder
    return kernel(NTraj, NSTEPS_LONG, NStates, NSKIP, mp['EStep'],
                  DTN, mp['M_mass'], mp['initState'], True,
                  R_rand, P_rand, mp['sigR'], mp['sigP'], map_rand,
                  mp['c'], mp['epsilon'], mp['Delta'], mp['omega'])


def run_orig_mfe():
    par = make_orig_params_long()
    return orig_mfe.runTraj(par)

def run_jit_mfe():
    kernel = make_mfe_kernel(hel, dhel, dhel0)
    np.random.seed(SEED)
    R_rand, P_rand = pre_gen_nuclear(NTraj)
    return kernel(NTraj, NSTEPS_LONG, NStates, NSKIP, mp['EStep'],
                  DTN, mp['M_mass'], mp['initState'],
                  R_rand, P_rand, mp['sigR'], mp['sigP'],
                  mp['c'], mp['epsilon'], mp['Delta'], mp['omega'])


def run_orig_sqc():
    par = make_orig_params_long()
    par.stype = "square"
    return orig_sqc.runTraj(par)

def run_jit_sqc():
    kernel = make_sqc_kernel(hel, dhel, dhel0)
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
    return kernel(NTraj, NSTEPS_LONG, NStates, NSKIP, mp['EStep'],
                  DTN, mp['M_mass'], mp['initState'], True,
                  R_rand, P_rand, mp['sigR'], mp['sigP'], map_rand,
                  mp['c'], mp['epsilon'], mp['Delta'], mp['omega'])


def run_orig_zpesqc():
    par = make_orig_params_long()
    par.stype = "square"
    return orig_zpesqc.runTraj(par)

def run_jit_zpesqc():
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
    return kernel(NTraj, NSTEPS_LONG, NStates, NSKIP, mp['EStep'],
                  DTN, mp['M_mass'], mp['initState'], True,
                  R_rand, P_rand, mp['sigR'], mp['sigP'], map_rand,
                  mp['c'], mp['epsilon'], mp['Delta'], mp['omega'])


def run_orig_spinlsc():
    pyrandom.seed(SEED)
    par = make_orig_params_long()
    return orig_spinlsc.runTraj(par)

def run_jit_spinlsc():
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
        for s in range(NStates):
            phi_rand[itraj, s] = pyrandom.random()
    return kernel(NTraj, NSTEPS_LONG, NStates, NSKIP, mp['EStep'],
                  DTN, mp['M_mass'], mp['initState'],
                  R_rand, P_rand, mp['sigR'], mp['sigP'], phi_rand,
                  mp['c'], mp['epsilon'], mp['Delta'], mp['omega'])


def run_orig_spinpldm():
    pyrandom.seed(SEED)
    par = make_orig_params_long()
    par.stype = "focused"
    return orig_spinpldm.runTraj(par)

def run_jit_spinpldm():
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
        for s in range(NStates):
            phi_rand[itraj, 0, 0, s] = pyrandom.random()  # phiF
            phi_rand[itraj, 0, 1, s] = pyrandom.random()  # phiB
    return kernel(NTraj, NSTEPS_LONG, NStates, NSKIP, mp['EStep'],
                  DTN, mp['M_mass'], initState,
                  R_rand, P_rand, mp['sigR'], mp['sigP'], phi_rand,
                  FB_arr, W_arr,
                  mp['c'], mp['epsilon'], mp['Delta'], mp['omega'])


def run_orig_mash():
    par = make_orig_params_long()
    return orig_mash.runTraj(par)

def run_jit_mash():
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
    return kernel(NTraj, NSTEPS_LONG, NStates, NSKIP,
                  DTN, mp['M_mass'], mp['initState'], 10,
                  R_rand, P_rand, mp['sigR'], mp['sigP'], uni_rand,
                  mp['c'], mp['epsilon'], mp['Delta'], mp['omega'])


# ===================================================================
#  FS-MASH  (stochastic hop -- uses random.random() for phases,
#            np.random.uniform() for Tully decisions)
# ===================================================================

def run_orig_fsmash():
    pyrandom.seed(SEED)
    par = make_orig_params_long()
    return orig_fsmash.runTraj(par)

def run_jit_fsmash():
    kernel = make_fsmash_kernel(hel, dhel, dhel0)
    # Phase randoms come from Python random (matching original's
    # random.random() calls in initElectronic).
    # Nuclear ICs and hop randoms come from numpy.
    pyrandom.seed(SEED)
    np.random.seed(SEED)
    R_rand = np.empty((NTraj, ndof))
    P_rand = np.empty((NTraj, ndof))
    phase_rand = np.empty((NTraj, NStates))
    # Pre-generate matching original consumption order:
    # For each traj: R[d]/P[d] interleaved (numpy), then NStates phases (python random)
    for itraj in range(NTraj):
        for d in range(ndof):
            R_rand[itraj, d] = np.random.normal()
            P_rand[itraj, d] = np.random.normal()
        for s in range(NStates):
            phase_rand[itraj, s] = pyrandom.random()
    # Hop randoms: pre-generate a pool per trajectory.
    # The original draws np.random.uniform() during simulation,
    # but those draws shift the numpy RNG for subsequent trajectories.
    # For bitwise match of the first few trajectories (before any hop),
    # we consume from the same numpy RNG after all ICs.
    hop_rand = np.random.uniform(0, 1, size=(NTraj, NSTEPS_LONG))
    return kernel(NTraj, NSTEPS_LONG, NStates, NSKIP,
                  DTN, mp['M_mass'], mp['initState'],
                  R_rand, P_rand, mp['sigR'], mp['sigP'],
                  phase_rand, hop_rand,
                  mp['c'], mp['epsilon'], mp['Delta'], mp['omega'])


# ===================================================================
#  unSMASH  (Bloch spheres, deterministic hops)
# ===================================================================

def run_orig_unsmash():
    par = make_orig_params_long()
    par.maxhop = 10
    return orig_unsmash.runTraj(par)

def run_jit_unsmash():
    kernel = make_unsmash_kernel(hel, dhel, dhel0)
    np.random.seed(SEED)
    R_rand = np.empty((NTraj, ndof))
    P_rand = np.empty((NTraj, ndof))
    sphere_rand = np.empty((NTraj, NStates, 2))
    acst_rand = np.empty(NTraj)
    for itraj in range(NTraj):
        for d in range(ndof):
            R_rand[itraj, d] = np.random.normal()
            P_rand[itraj, d] = np.random.normal()
        for b in range(NStates):
            sphere_rand[itraj, b, 0] = np.random.random()
            sphere_rand[itraj, b, 1] = np.random.random()
        acst_rand[itraj] = np.random.random()
    return kernel(NTraj, NSTEPS_LONG, NStates, NSKIP,
                  DTN, mp['M_mass'], mp['initState'], 10,
                  R_rand, P_rand, mp['sigR'], mp['sigP'],
                  sphere_rand, acst_rand,
                  mp['c'], mp['epsilon'], mp['Delta'], mp['omega'])


# === Main ===

METHODS = [
    ('pldm',     run_orig_pldm,     run_jit_pldm),
    ('mfe',      run_orig_mfe,      run_jit_mfe),
    ('sqc',      run_orig_sqc,      run_jit_sqc),
    ('zpesqc',   run_orig_zpesqc,   run_jit_zpesqc),
    ('spinlsc',  run_orig_spinlsc,  run_jit_spinlsc),
    ('spinpldm', run_orig_spinpldm, run_jit_spinpldm),
    ('mash',     run_orig_mash,     run_jit_mash),
    ('fsmash',   run_orig_fsmash,   run_jit_fsmash),
    ('unsmash',  run_orig_unsmash,  run_jit_unsmash),
]


def main():
    print('=' * 70)
    print('  Long-time error accumulation test')
    print(f'  NSteps={NSTEPS_LONG}, nskip={NSKIP}, dtN={DTN}')
    print(f'  t_max = {NSTEPS_LONG * DTN:.1f} a.u., NTraj={NTraj}')
    print('=' * 70)

    # Collect results for plotting
    all_residuals = {}
    all_rho_orig = {}
    all_rho_jit = {}

    for name, run_orig, run_jit in METHODS:
        print(f'\n--- {name.upper()} ---')

        print(f'  Running original...', end=' ', flush=True)
        t0 = time.perf_counter()
        rho_orig = run_orig()
        t_orig = time.perf_counter() - t0
        print(f'{t_orig:.1f}s')

        print(f'  Running JIT (warmup)...', end=' ', flush=True)
        _ = run_jit()
        print('done')

        print(f'  Running JIT (benchmark)...', end=' ', flush=True)
        t0 = time.perf_counter()
        rho_jit = run_jit()
        t_jit = time.perf_counter() - t0
        print(f'{t_jit:.2f}s  (speedup={t_orig/max(t_jit,1e-6):.0f}x)')

        # Per-timestep max residual
        residual_t = np.zeros(n_skip)
        for t in range(n_skip):
            residual_t[t] = np.max(np.abs(rho_orig[:, :, t] - rho_jit[:, :, t])) / NTraj

        all_residuals[name] = residual_t
        all_rho_orig[name] = rho_orig
        all_rho_jit[name] = rho_jit

        max_err = np.max(residual_t)
        print(f'  max|diff|/NTraj = {max_err:.4e}')

    # ==========================================
    # FIGURE 1: Residual over time (all methods)
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    continuous = ['pldm', 'mfe', 'sqc', 'zpesqc', 'spinlsc', 'spinpldm']
    colors_cont = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple',
                   'tab:brown', 'tab:cyan']
    for name, col in zip(continuous, colors_cont):
        if name in all_residuals:
            ax1.plot(t_axis, all_residuals[name], '-', lw=1.3,
                     label=name.upper(), color=col)
    ax1.set_ylabel('max|orig - JIT| / NTraj')
    ax1.set_title(f'Long-time residual: Continuous methods  '
                  f'(NSteps={NSTEPS_LONG}, NTraj={NTraj})')
    ax1.legend(loc='best', fontsize=8, ncol=2)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    hopping = ['mash', 'fsmash', 'unsmash']
    colors_hop = ['tab:red', 'tab:pink', 'tab:olive']
    for name, col in zip(hopping, colors_hop):
        if name in all_residuals:
            ax2.plot(t_axis, all_residuals[name], '-', lw=1.5,
                     label=name.upper(), color=col)
    ax2.set_xlabel('Time (a.u.)')
    ax2.set_ylabel('max|orig - JIT| / NTraj')
    ax2.set_title('Long-time residual: Surface-hopping methods')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path1 = os.path.join(OUTDIR, 'long_time_residual_all.png')
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f'\nSaved {path1}')

    # ==========================================
    # FIGURE 2: Population dynamics + residual per method
    # ==========================================
    all_names = continuous + ['mash', 'fsmash', 'unsmash']
    n_methods = len(all_names)

    fig2, axes = plt.subplots(n_methods, 2, figsize=(16, 3.5 * n_methods),
                              sharex=True)

    for idx, name in enumerate(all_names):
        ax_pop = axes[idx, 0]
        ax_res = axes[idx, 1]
        rho_o = all_rho_orig[name]
        rho_j = all_rho_jit[name]

        # Population: rho[0,0,:] / NTraj
        pop_orig = rho_o[0, 0, :].real / NTraj
        pop_jit = rho_j[0, 0, :].real / NTraj

        ax_pop.plot(t_axis, pop_orig, 'b-', lw=1.3, label='Original', alpha=0.8)
        ax_pop.plot(t_axis, pop_jit, 'r--', lw=1.3, label='JIT', alpha=0.8)
        ax_pop.set_ylabel(r'$\rho_{00}$')
        ax_pop.set_title(f'{name.upper()} — Population', fontsize=10)
        ax_pop.legend(fontsize=7, loc='best')
        ax_pop.grid(True, alpha=0.3)

        ax_res.plot(t_axis, all_residuals[name], '-', lw=1.2, color='tab:red')
        ax_res.set_ylabel('Residual')
        ax_res.set_title(f'{name.upper()} — Residual', fontsize=10)
        ax_res.grid(True, alpha=0.3)
        ax_res.set_yscale('log')

    axes[-1, 0].set_xlabel('Time (a.u.)')
    axes[-1, 1].set_xlabel('Time (a.u.)')

    fig2.suptitle(f'Long-time dynamics: Original vs JIT  '
                  f'(NSteps={NSTEPS_LONG}, NTraj={NTraj}, t_max={NSTEPS_LONG*DTN:.0f} a.u.)',
                  fontsize=13, y=1.005)
    fig2.tight_layout()
    path2 = os.path.join(OUTDIR, 'long_time_population_all.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved {path2}')

    # ==========================================
    # Summary table
    # ==========================================
    print('\n' + '=' * 70)
    print(f'{"Method":>12s}  {"max residual":>14s}  {"final residual":>14s}  {"trend":>10s}')
    print('-' * 55)
    for name in all_names:
        res = all_residuals[name]
        mx = np.max(res)
        final = res[-1]
        # Simple trend: ratio of last-quarter to first-quarter mean
        q1 = np.mean(res[:n_skip // 4])
        q4 = np.mean(res[3 * n_skip // 4:])
        if q1 > 0:
            ratio = q4 / q1
            if ratio > 2:
                trend = 'GROWING'
            elif ratio < 0.5:
                trend = 'shrinking'
            else:
                trend = 'stable'
        else:
            trend = 'zero'
        print(f'{name:>12s}  {mx:14.4e}  {final:14.4e}  {trend:>10s}')
    print('=' * 70)
    print('Done.')


if __name__ == '__main__':
    main()
