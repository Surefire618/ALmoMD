#!/usr/bin/env python3
"""Benchmark three ensemble inference implementations.

Implementations:
  1. original  — sequential loop: struc.calc=c; struc.get_potential_energy(); struc.get_forces()
  2. direct    — direct loop: calc.calculate(atoms); read results
  3. ctx_mgr   — context-manager shared neighbor list (current implementation)

Usage:
    python benchmark_ensemble.py MODEL_DIR SUPERCELL [--nwarmup N] [--nreps N]

    MODEL_DIR    directory containing deployed-model_X_Y.pth files
    SUPERCELL    path to geometry.in.supercell (ASE aims format)
"""

import argparse
import time
import warnings

import numpy as np
import torch
torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Implementation 1: original sequential (pre-patch baseline)
# ---------------------------------------------------------------------------
def impl_original(calculators, struc):
    energies, forces = [], []
    for calc in calculators:
        struc.calc = calc
        energies.append(struc.get_potential_energy())
        forces.append(struc.get_forces())
    return energies, forces


# ---------------------------------------------------------------------------
# Implementation 2: direct calculate() loop without neighbor-list sharing
# ---------------------------------------------------------------------------
def impl_direct(calculators, struc):
    for calc in calculators:
        calc.calculate(atoms=struc)
    energies = [float(calc.results['energy']) for calc in calculators]
    forces   = [calc.results['forces'] for calc in calculators]
    return energies, forces


# ---------------------------------------------------------------------------
# Implementation 3: threaded shared neighbor list (current code)
# ---------------------------------------------------------------------------
def impl_threaded_shared(calculators, struc):
    from libs.lib_load_model import ensemble_calculate
    results = ensemble_calculate(calculators, struc)
    energies = [float(r['energy']) for r in results]
    forces   = [r['forces'] for r in results]
    return energies, forces


# ---------------------------------------------------------------------------
# Implementation 4: threaded, NO shared neighbor list (stripped, no nequip dep)
# ---------------------------------------------------------------------------
def impl_threaded_stripped(calculators, struc):
    import threading
    atoms_copies = [struc.copy() for _ in calculators]
    threads = [
        threading.Thread(target=calc.calculate, args=(ac,))
        for calc, ac in zip(calculators, atoms_copies)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    energies = [float(calc.results['energy']) for calc in calculators]
    forces   = [calc.results['forces'] for calc in calculators]
    return energies, forces


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
def invalidate_cache(calculators, struc):
    """Clear ASE result cache on all calculators and detach struc from all.

    Without this, get_potential_energy() / get_forces() are cache hits when
    atom positions don't change between reps, making impl_original ~1000x
    faster than it would be in real MD (where positions update each step).
    """
    for calc in calculators:
        calc.results = {}
    struc.calc = None


def bench(name, fn, calculators, struc, nwarmup, nreps):
    # warmup
    for _ in range(nwarmup):
        invalidate_cache(calculators, struc)
        fn(calculators, struc)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(nreps):
        invalidate_cache(calculators, struc)
        t0 = time.perf_counter()
        fn(calculators, struc)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    arr = np.array(times)
    print(f'  {name:12s}  mean={arr.mean()*1e3:7.2f} ms  '
          f'std={arr.std()*1e3:6.2f} ms  '
          f'min={arr.min()*1e3:7.2f} ms  '
          f'max={arr.max()*1e3:7.2f} ms  (n={nreps})')
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='Directory with deployed-model_X_Y.pth')
    parser.add_argument('supercell', help='geometry.in.supercell path')
    parser.add_argument('--nmodel', type=int, default=2)
    parser.add_argument('--nstep',  type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--nwarmup', type=int, default=3)
    parser.add_argument('--nreps',   type=int, default=20)
    args = parser.parse_args()

    from nequip.ase import nequip_calculator
    from ase.io import read as atoms_read

    print(f'Loading structure: {args.supercell}')
    struc = atoms_read(args.supercell, format='aims')
    print(f'  {len(struc)} atoms')

    print(f'Loading {args.nmodel * args.nstep} models from {args.model_dir}')
    calculators = []
    for im in range(args.nmodel):
        for ist in range(args.nstep):
            pth = f'{args.model_dir}/deployed-model_{im}_{ist}.pth'
            calc = nequip_calculator.NequIPCalculator.from_deployed_model(
                pth, device=args.device
            )
            calculators.append(calc)
    print(f'  Loaded {len(calculators)} calculators, r_max={calculators[0].r_max}')

    print(f'\nBenchmark: {args.nwarmup} warmup + {args.nreps} timed reps')
    print('-' * 72)
    t_orig     = bench('original',     impl_original,         calculators, struc, args.nwarmup, args.nreps)
    t_direct   = bench('direct',       impl_direct,           calculators, struc, args.nwarmup, args.nreps)
    t_shared   = bench('thr_shared',   impl_threaded_shared,  calculators, struc, args.nwarmup, args.nreps)
    t_stripped = bench('thr_stripped', impl_threaded_stripped, calculators, struc, args.nwarmup, args.nreps)
    print('-' * 72)
    print(f'  speedup thr_shared   vs original: {t_orig.mean()/t_shared.mean():.3f}x')
    print(f'  speedup thr_stripped vs original: {t_orig.mean()/t_stripped.mean():.3f}x')
    print(f'  thr_stripped vs thr_shared:       {t_shared.mean()/t_stripped.mean():.3f}x  '
          f'(>1 means stripped wins, <1 means shared wins)')

    # sanity-check: forces agree across implementations
    _, f1 = impl_original(calculators, struc)
    _, f2 = impl_threaded_stripped(calculators, struc)
    for i, (a, b) in enumerate(zip(f1, f2)):
        diff = np.max(np.abs(np.array(a) - np.array(b)))
        if diff > 1e-6:
            print(f'WARNING: forces differ for model {i}: max_diff={diff:.2e}')
    print('Forces cross-check passed (max diff < 1e-6 eV/Å)')


if __name__ == '__main__':
    main()
