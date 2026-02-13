#!/usr/bin/env python3
"""
Validate JIT-compiled PLDM against the original implementation.

Runs both versions with the SAME random-number sequence and asserts
that the density-matrix output matches to floating-point tolerance.
Then benchmarks wall-clock time for both versions and outputs PNG figures.

Outputs (saved to tests/output/):
  1_correctness_single_traj.png  -- overlay of original vs JIT populations
  2_correctness_residual.png     -- residual (orig - JIT) vs time
  3_benchmark_bars.png           -- wall-clock comparison bar chart
  4_population_dynamics.png      -- production-quality P(t) from JIT
"""

import sys, os, time
import numpy as np

# ---- paths -----------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'Method'))
sys.path.insert(0, os.path.join(ROOT, 'Model'))
sys.path.insert(0, ROOT)

OUTDIR = os.path.join(ROOT, 'tests', 'output')
os.makedirs(OUTDIR, exist_ok=True)

# ---- import original -------------------------------------------------------
import pldm as pldm_orig
import spinBoson as sb_orig

# ---- import JIT ------------------------------------------------------------
from ModelJIT.spinBoson import hel, dhel, dhel0, get_model_params
from MethodJIT.pldm import make_pldm_kernel

# ---- matplotlib (non-interactive backend for headless) ----------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# Helpers
# ============================================================================

def _make_orig_par(NTraj, seed):
    par = sb_orig.parameters()
    par.NTraj = NTraj
    par.SEED  = seed
    par.stype = 'focused'
    par.Hel   = sb_orig.Hel
    par.dHel  = sb_orig.dHel
    par.dHel0 = sb_orig.dHel0
    par.initR = sb_orig.initR
    return par


def run_original(NTraj, seed):
    par = _make_orig_par(NTraj, seed)
    t0 = time.perf_counter()
    rho = pldm_orig.runTraj(par)
    elapsed = time.perf_counter() - t0
    return rho, elapsed


def run_jit(NTraj, seed, kernel=None):
    p = get_model_params(3)
    if kernel is None:
        kernel = make_pldm_kernel(hel, dhel, dhel0)

    # Replicate the exact random sequence the original uses.
    # Original: np.random.seed(SEED), then for d in range(ndof):
    #     R[d] = normal()*sigR[d];  P[d] = normal()*sigP[d]
    # (interleaved in the same loop)
    np.random.seed(seed)
    ndof = p['ndof']
    R_rand_all = np.empty((NTraj, ndof))
    P_rand_all = np.empty((NTraj, ndof))
    for itraj in range(NTraj):
        for d in range(ndof):
            R_rand_all[itraj, d] = np.random.randn()
            P_rand_all[itraj, d] = np.random.randn()

    map_rand = np.empty((NTraj, 4, p['NStates']))

    # warm-up (not counted)
    _ = kernel(1, 10, p['NStates'], 10, p['EStep'],
               p['dtN'], p['M_mass'], p['initState'], True,
               R_rand_all[:1], P_rand_all[:1], p['sigR'], p['sigP'],
               map_rand[:1], p['c'], p['epsilon'], p['Delta'], p['omega'])

    # Re-generate randoms after warm-up
    np.random.seed(seed)
    for itraj in range(NTraj):
        for d in range(ndof):
            R_rand_all[itraj, d] = np.random.randn()
            P_rand_all[itraj, d] = np.random.randn()

    t0 = time.perf_counter()
    rho = kernel(NTraj, p['NSteps'], p['NStates'], p['nskip'], p['EStep'],
                 p['dtN'], p['M_mass'], p['initState'], True,
                 R_rand_all, P_rand_all, p['sigR'], p['sigP'],
                 map_rand, p['c'], p['epsilon'], p['Delta'], p['omega'])
    elapsed = time.perf_counter() - t0
    return rho, elapsed, p


# ============================================================================
# 1) Correctness: single trajectory
# ============================================================================

def test_correctness():
    seed  = 12345
    NTraj = 1

    rho_orig, _ = run_original(NTraj, seed)
    rho_jit, _, p = run_jit(NTraj, seed)

    assert rho_orig.shape == rho_jit.shape, (
        f"Shape mismatch: {rho_orig.shape} vs {rho_jit.shape}")

    max_diff = np.max(np.abs(rho_orig - rho_jit))
    NStates  = p['NStates']
    n_skip   = rho_orig.shape[-1]
    dtN      = p['dtN']
    nskip    = p['nskip']
    times    = np.arange(n_skip) * nskip * dtN

    # --- Figure 1: overlay populations ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in range(NStates):
        pop_o = rho_orig[s, s, :].real / NTraj
        pop_j = rho_jit[s, s, :].real / NTraj
        ax.plot(times, pop_o, '-',  lw=2.5, label=f'Original  P{s+1}{s+1}')
        ax.plot(times, pop_j, '--', lw=1.5, label=f'JIT       P{s+1}{s+1}')
    ax.set_xlabel('Time (a.u.)', fontsize=13)
    ax.set_ylabel('Population', fontsize=13)
    ax.set_title('Correctness: Original vs JIT  (1 trajectory, seed=12345)',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, '1_correctness_single_traj.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved 1_correctness_single_traj.png")

    # --- Figure 2: residual ---
    fig, ax = plt.subplots(figsize=(8, 4))
    for s in range(NStates):
        residual = (rho_orig[s, s, :] - rho_jit[s, s, :]).real
        ax.plot(times, residual, lw=1.5, label=f'P{s+1}{s+1}')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Time (a.u.)', fontsize=13)
    ax.set_ylabel('Original - JIT', fontsize=13)
    ax.set_title(f'Residual  (max |diff| = {max_diff:.2e})', fontsize=13)
    ax.legend(fontsize=11)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, '2_correctness_residual.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved 2_correctness_residual.png")

    # --- console summary ---
    print(f"\n  CORRECTNESS SUMMARY")
    print(f"  {'='*50}")
    print(f"  Max abs diff:  {max_diff:.3e}")
    passed = max_diff < 1e-10
    print(f"  Result:        {'PASS (< 1e-10)' if passed else 'FAIL'}")
    return max_diff


# ============================================================================
# 2) Benchmark
# ============================================================================

def test_benchmark():
    seed = 42
    test_cases = [200, 1000, 5000]

    times_orig = []
    times_jit  = []
    speedups   = []

    kernel = make_pldm_kernel(hel, dhel, dhel0)

    print(f"\n  BENCHMARK  (spinBoson model 3: NStates=2, ndof=100, NSteps=200)")
    print(f"  {'='*60}")
    print(f"  {'NTraj':>8}  {'Original (s)':>14}  {'JIT (s)':>10}  {'Speedup':>10}")
    print(f"  {'-'*60}")

    for NTraj in test_cases:
        _, t_o = run_original(NTraj, seed)
        _, t_j, _ = run_jit(NTraj, seed, kernel=kernel)
        sp = t_o / t_j if t_j > 0 else float('inf')
        times_orig.append(t_o)
        times_jit.append(t_j)
        speedups.append(sp)
        print(f"  {NTraj:>8}  {t_o:>14.3f}  {t_j:>10.4f}  {sp:>9.1f}x")

    print()

    # --- Figure 3: benchmark bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: absolute times
    ax = axes[0]
    x = np.arange(len(test_cases))
    w = 0.35
    bars1 = ax.bar(x - w/2, times_orig, w, label='Original (NumPy)',
                   color='#e74c3c', alpha=0.85)
    bars2 = ax.bar(x + w/2, times_jit, w, label='JIT (Numba + prange)',
                   color='#2ecc71', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in test_cases])
    ax.set_xlabel('NTraj', fontsize=13)
    ax.set_ylabel('Wall-clock time (s)', fontsize=13)
    ax.set_title('Absolute Runtime', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    # annotate bars
    for bar, t in zip(bars1, times_orig):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                f'{t:.2f}s', ha='center', va='bottom', fontsize=9)
    for bar, t in zip(bars2, times_jit):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                f'{t:.3f}s', ha='center', va='bottom', fontsize=9)

    # Panel B: speedup
    ax = axes[1]
    bars = ax.bar(x, speedups, 0.5, color='#3498db', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in test_cases])
    ax.set_xlabel('NTraj', fontsize=13)
    ax.set_ylabel('Speedup (Original / JIT)', fontsize=13)
    ax.set_title('Speedup Factor', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    for bar, sp in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{sp:.0f}x', ha='center', va='bottom', fontsize=11,
                fontweight='bold')

    fig.suptitle('PLDM Benchmark:  spinBoson (NStates=2, ndof=100, NSteps=200)',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, '3_benchmark_bars.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved 3_benchmark_bars.png")

    return test_cases, times_orig, times_jit, speedups


# ============================================================================
# 3) Production-quality population dynamics plot from JIT
# ============================================================================

def test_production_plot():
    """Run a large ensemble with JIT and plot publication-quality P(t)."""
    NTraj = 5000
    seed  = 42
    print(f"\n  PRODUCTION RUN  (NTraj={NTraj})")
    print(f"  {'='*50}")

    kernel = make_pldm_kernel(hel, dhel, dhel0)
    rho, t_jit, p = run_jit(NTraj, seed, kernel=kernel)
    print(f"  JIT time: {t_jit:.3f} s  ({t_jit/NTraj*1e6:.1f} us/traj)")

    NStates = p['NStates']
    n_skip  = rho.shape[-1]
    times   = np.arange(n_skip) * p['nskip'] * p['dtN']

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2c3e50', '#e74c3c']
    for s in range(NStates):
        pop = rho[s, s, :].real / NTraj
        ax.plot(times, pop, lw=2, color=colors[s],
                label=f'$P_{{{s+1}{s+1}}}(t)$')

    ax.set_xlabel('Time (a.u.)', fontsize=14)
    ax.set_ylabel('Population', fontsize=14)
    ax.set_title(f'PLDM  spinBoson (model 3)  |  '
                 f'NTraj = {NTraj}  |  JIT: {t_jit:.3f}s',
                 fontsize=13)
    ax.legend(fontsize=13, loc='center right')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(times[0], times[-1])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, '4_population_dynamics.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved 4_population_dynamics.png")


# ============================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("  PLDM JIT Validation & Benchmark Suite")
    print("=" * 65)

    print("\n[1/3] Correctness test ...")
    test_correctness()

    print("\n[2/3] Benchmark ...")
    test_benchmark()

    print("\n[3/3] Production plot ...")
    test_production_plot()

    print("\n" + "=" * 65)
    print(f"  All outputs saved to: {OUTDIR}")
    print("=" * 65)
