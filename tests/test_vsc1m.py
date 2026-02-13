#!/usr/bin/env python
"""
tests/test_vsc1m.py

Validation tests for the VSC (vibrational strong coupling) model
implementations:

  1. Model construction sanity checks (Hamiltonian symmetry, eigenvalues).
  2. Single-trajectory MFE propagation (basic smoke test).
  3. Single-trajectory FS-MASH propagation (basic smoke test).
  4. Effective bath model sanity checks.
  5. hvij observable post-processing correctness.
  6. Backward-compatible Upolaron support in standard MASH and MFE.

Usage:
    python tests/test_vsc1m.py
"""

import sys
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUTDIR = os.path.join(ROOT, 'tests', 'output')
os.makedirs(OUTDIR, exist_ok=True)

SEED = 42


# ---------------------------------------------------------------------------
# Helper: wire model functions to parameters object
# ---------------------------------------------------------------------------
def make_params(model_module, NTraj=5, NSteps=None):
    """Create a parameters object from a model module, ready for methods."""
    par = model_module.parameters()
    par.dHel  = model_module.dHel
    par.dHel0 = model_module.dHel0
    par.initR = model_module.initR
    par.Hel   = model_module.Hel
    par.SEED  = SEED
    par.NTraj = NTraj
    if NSteps is not None:
        par.NSteps = NSteps
        par.nskip  = max(1, NSteps // 100)
    return par


# ===================================================================
#  Test 1: QPH model construction
# ===================================================================
def test_qph_model():
    """Validate the quantum-photon VSC model construction."""
    print('\n--- Test 1: QPH Model Construction ---')
    from Model import vsc1m_qph as qph_model

    par = qph_model.parameters

    # Check dimensions
    NStates = par.NStates
    assert NStates == par.Nm * par.Nf, \
        f"NStates={NStates} != Nm*Nf={par.Nm}*{par.Nf}"
    print(f"  NStates = {NStates} (Nm={par.Nm}, Nf={par.Nf})")
    print(f"  ndof = {par.ndof} (Nmbath={par.Nmbath}, Ncbath={par.Ncbath})")

    # Check Hamiltonian symmetry
    R_test = np.zeros(par.ndof)
    H = qph_model.Hel(R_test)
    assert H.shape == (NStates, NStates), f"H shape: {H.shape}"
    assert np.allclose(H, H.T), "Hamiltonian is not symmetric at R=0"
    print(f"  H(R=0) is {NStates}x{NStates}, symmetric: OK")

    # Check eigenvalues are real
    E = np.linalg.eigvalsh(H)
    assert np.all(np.isfinite(E)), "Eigenvalues contain non-finite values"
    print(f"  Eigenvalues at R=0: {E[:4]} ...")

    # Check gradient shapes
    dH = qph_model.dHel(R_test)
    assert dH.shape == (NStates, NStates, par.ndof), f"dHel shape: {dH.shape}"
    dH0 = qph_model.dHel0(R_test)
    assert dH0.shape == (par.ndof,), f"dHel0 shape: {dH0.shape}"
    print(f"  dHel shape: {dH.shape}, dHel0 shape: {dH0.shape}: OK")

    # Check Upolaron shape
    assert par.Upolaron.shape == (NStates, NStates), \
        f"Upolaron shape: {par.Upolaron.shape}"
    print(f"  Upolaron shape: {par.Upolaron.shape}: OK")

    # Check hvij shape and Hermiticity
    assert par.hvij.shape == (NStates, NStates)
    assert np.allclose(par.hvij, par.hvij.T), "hvij is not symmetric"
    print(f"  hvij shape: {par.hvij.shape}, symmetric: OK")

    # Check initR returns correct shapes
    np.random.seed(SEED)
    R, P = qph_model.initR()
    assert R.shape == (par.ndof,), f"R shape: {R.shape}"
    assert P.shape == (par.ndof,), f"P shape: {P.shape}"
    print(f"  initR() shapes: R={R.shape}, P={P.shape}: OK")

    print('  PASSED')


# ===================================================================
#  Test 2: Effective bath model construction
# ===================================================================
def test_effbath_model():
    """Validate the effective-bath VSC model construction."""
    print('\n--- Test 2: Effective Bath Model Construction ---')
    from Model import vsc1m_effbath as eff_model

    par = eff_model.parameters

    NStates = par.NStates
    assert NStates == par.Nm, f"NStates={NStates} != Nm={par.Nm}"
    print(f"  NStates = {NStates} (Nm={par.Nm})")
    print(f"  ndof = {par.ndof} (Nmbath={par.Nmbath}, Ncbath={par.Ncbath})")

    # Hamiltonian check
    R_test = np.zeros(par.ndof)
    H = eff_model.Hel(R_test)
    assert H.shape == (NStates, NStates)
    assert np.allclose(H, H.T), "Hamiltonian not symmetric"
    print(f"  H(R=0) is {NStates}x{NStates}, symmetric: OK")

    # Gradient shapes
    dH = eff_model.dHel(R_test)
    assert dH.shape == (NStates, NStates, par.ndof)
    dH0 = eff_model.dHel0(R_test)
    assert dH0.shape == (par.ndof,)
    print(f"  Gradient shapes: OK")

    # No Upolaron for effective bath
    assert not hasattr(par, 'Upolaron') or par.__dict__.get('Upolaron') is None \
        or not hasattr(par, 'Upolaron'), \
        "Effective bath model should not have Upolaron"
    print(f"  No Upolaron: OK")

    # hvij check
    assert par.hvij.shape == (NStates, NStates)
    assert np.allclose(par.hvij, par.hvij.T)
    print(f"  hvij shape: {par.hvij.shape}, symmetric: OK")

    # initR
    np.random.seed(SEED)
    R, P = eff_model.initR()
    assert R.shape == (par.ndof,)
    assert P.shape == (par.ndof,)
    print(f"  initR() shapes: OK")

    print('  PASSED')


# ===================================================================
#  Test 3: MFE smoke test with QPH model
# ===================================================================
def test_mfe_qph():
    """Run a short MFE trajectory with the QPH model."""
    print('\n--- Test 3: MFE with QPH Model (smoke test) ---')
    from Model import vsc1m_qph as qph_model
    from Method import mfe

    NTraj = 2
    NSteps = 50
    par = make_params(qph_model, NTraj=NTraj, NSteps=NSteps)

    t0 = time.time()
    rho = mfe.runTraj(par)
    elapsed = time.time() - t0

    n_frames = rho.shape[-1]
    print(f"  rho shape: {rho.shape}, n_frames={n_frames}")
    print(f"  Elapsed: {elapsed:.2f}s")

    # Check that populations are roughly normalised
    for t_idx in [0, n_frames // 2, n_frames - 1]:
        diag_sum = sum(rho[i, i, t_idx].real for i in range(par.NStates))
        pop_norm = diag_sum / NTraj
        assert 0.5 < pop_norm < 1.5, \
            f"Population sum at frame {t_idx}: {pop_norm} (expected ~1)"

    print(f"  Population normalisation: OK")

    # hvij observable
    hvij = par.hvij
    left_pop_0 = np.trace(hvij @ rho[:, :, 0]).real / NTraj
    print(f"  Left-well pop at t=0: {left_pop_0:.4f}")
    assert 0.0 < left_pop_0 < 1.5, \
        f"Left-well population out of range: {left_pop_0}"

    print('  PASSED')


# ===================================================================
#  Test 4: FS-MASH smoke test with QPH model
# ===================================================================
def test_fsmash_qph():
    """Run a short FS-MASH trajectory with the QPH model."""
    print('\n--- Test 4: FS-MASH with QPH Model (smoke test) ---')
    from Model import vsc1m_qph as qph_model
    from Method import fsmash

    NTraj = 2
    NSteps = 50
    par = make_params(qph_model, NTraj=NTraj, NSteps=NSteps)

    t0 = time.time()
    rho = fsmash.runTraj(par)
    elapsed = time.time() - t0

    n_frames = rho.shape[-1]
    print(f"  rho shape: {rho.shape}, n_frames={n_frames}")
    print(f"  Elapsed: {elapsed:.2f}s")

    # Check populations
    for t_idx in [0, n_frames // 2, n_frames - 1]:
        diag_sum = sum(rho[i, i, t_idx].real for i in range(par.NStates))
        pop_norm = diag_sum / NTraj
        assert 0.3 < pop_norm < 2.0, \
            f"Population sum at frame {t_idx}: {pop_norm} (expected ~1)"

    print(f"  Population normalisation: OK")
    print('  PASSED')


# ===================================================================
#  Test 5: MFE with effective bath model
# ===================================================================
def test_mfe_effbath():
    """Run a short MFE trajectory with the effective bath model."""
    print('\n--- Test 5: MFE with Effective Bath Model (smoke test) ---')
    from Model import vsc1m_effbath as eff_model
    from Method import mfe

    NTraj = 2
    NSteps = 50
    par = make_params(eff_model, NTraj=NTraj, NSteps=NSteps)

    t0 = time.time()
    rho = mfe.runTraj(par)
    elapsed = time.time() - t0

    n_frames = rho.shape[-1]
    print(f"  rho shape: {rho.shape}, n_frames={n_frames}")
    print(f"  Elapsed: {elapsed:.2f}s")

    # Population check
    for t_idx in [0, n_frames - 1]:
        diag_sum = sum(rho[i, i, t_idx].real for i in range(par.NStates))
        pop_norm = diag_sum / NTraj
        assert 0.5 < pop_norm < 1.5, \
            f"Population sum at frame {t_idx}: {pop_norm}"

    print(f"  Population normalisation: OK")
    print('  PASSED')


# ===================================================================
#  Test 6: Standard MASH with QPH model (Upolaron backward compat)
# ===================================================================
def test_mash_qph():
    """Run standard MASH with the QPH model to test Upolaron support."""
    print('\n--- Test 6: Standard MASH with QPH Model (Upolaron) ---')
    from Model import vsc1m_qph as qph_model
    from Method import mash

    NTraj = 2
    NSteps = 50
    par = make_params(qph_model, NTraj=NTraj, NSteps=NSteps)

    t0 = time.time()
    rho = mash.runTraj(par)
    elapsed = time.time() - t0

    n_frames = rho.shape[-1]
    print(f"  rho shape: {rho.shape}, n_frames={n_frames}")
    print(f"  Elapsed: {elapsed:.2f}s")

    # Population check
    for t_idx in [0, n_frames - 1]:
        diag_sum = sum(rho[i, i, t_idx].real for i in range(par.NStates))
        pop_norm = diag_sum / NTraj
        assert 0.3 < pop_norm < 2.0, \
            f"Population sum at frame {t_idx}: {pop_norm}"

    print(f"  Population normalisation: OK")
    print('  PASSED')


# ===================================================================
#  Test 7: Standard MASH with spin-boson (no Upolaron, regression)
# ===================================================================
def test_mash_spinboson_regression():
    """Ensure standard MASH still works without Upolaron (backward compat)."""
    print('\n--- Test 7: Standard MASH with spinBoson (regression) ---')
    from Model import spinBoson as sb_model
    from Method import mash

    par = sb_model.parameters()
    par.NTraj = 5
    par.NSteps = 200
    par.nskip = 10
    par.dHel  = sb_model.dHel
    par.dHel0 = sb_model.dHel0
    par.initR = sb_model.initR
    par.Hel   = sb_model.Hel
    par.SEED  = SEED

    t0 = time.time()
    rho = mash.runTraj(par)
    elapsed = time.time() - t0

    n_frames = rho.shape[-1]
    NTraj = par.NTraj
    print(f"  rho shape: {rho.shape}, elapsed: {elapsed:.2f}s")

    # Populations should sum to NTraj at each frame
    pop_0 = sum(rho[i, i, 0].real for i in range(par.NStates))
    assert abs(pop_0 / NTraj - 1.0) < 0.3, \
        f"Population sum at t=0: {pop_0/NTraj}"

    print(f"  Backward compatibility: OK")
    print('  PASSED')


# ===================================================================
#  Main
# ===================================================================
def main():
    print('=' * 60)
    print('  VSC Model Integration Tests')
    print('=' * 60)

    tests = [
        test_qph_model,
        test_effbath_model,
        test_mfe_qph,
        test_fsmash_qph,
        test_mfe_effbath,
        test_mash_qph,
        test_mash_spinboson_regression,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f'  FAILED: {e}')
            import traceback
            traceback.print_exc()
            failed += 1

    print('\n' + '=' * 60)
    print(f'  Results: {passed} passed, {failed} failed')
    print('=' * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
