import os
import sys
import threading
import warnings
from libs.lib_util import single_print



def ensemble_calculate(calculators, atoms):
    """Run every NequIPCalculator on `atoms` sharing one neighbor list.

    The neighbor list is built once from the current atoms and reused
    across all ensemble models for this single call. It is rebuilt on
    every call, so successive MD steps see fresh lists built from the
    updated positions -- identical freshness to the pre-patch code,
    just without the N-way redundancy.

    Each calculator runs in its own thread so that models placed on
    different GPU devices execute their forward passes concurrently.
    Each thread receives its own copy of `atoms` to avoid shared-state
    races. The GIL is released during CUDA kernel launches, so threads
    on separate devices run truly in parallel. On a single device the
    kernels are still serialized by CUDA, but there is no regression
    versus sequential execution.

    Delegates all result extraction to each calculator's own
    calculate(), so unit conversions, output key handling, stress
    Voigt conversion, and ASE cache writeback stay inside nequip and
    track upstream changes automatically.

    All calculators must share r_max and transform; asserted in
    load_model after the calculators are built.

    Returns (energies, forces) as parallel lists.
    """
    import threading

    from nequip.data import AtomicData

    prebuilt = AtomicData.from_ase(atoms=atoms, r_max=calculators[0].r_max)
    original = AtomicData.__dict__['from_ase']
    call_count = 0
    call_lock = threading.Lock()

    def _cached(atoms=None, r_max=None, **kwargs):
        nonlocal call_count
        with call_lock:
            call_count += 1
        return prebuilt.clone()

    AtomicData.from_ase = _cached
    try:
        atoms_copies = [atoms.copy() for _ in calculators]
        threads = [
            threading.Thread(target=calc.calculate, args=(ac,))
            for calc, ac in zip(calculators, atoms_copies)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        AtomicData.from_ase = original

    if call_count < len(calculators):
        warnings.warn(
            f'ensemble_calculate: AtomicData.from_ase was called '
            f'{call_count} times, expected {len(calculators)}. Upstream '
            f'nequip may have stopped calling from_ase in calculate; '
            f'neighbor-list dedup is no longer effective.',
            RuntimeWarning,
        )

    energies = [float(calc.results['energy']) for calc in calculators]
    forces = [calc.results['forces'] for calc in calculators]
    return energies, forces


def load_model(inputs):

    # Set the path to folders storing the training data for NequIP
    workpath = f'./MODEL/{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}'

    # Initialization of a termination signal
    signal = 0

    single_print(f'\t\tDevice: {inputs.device}')

    # Load the trained models as calculators
    inputs.calc_MLIP = []
    if inputs.MLIP == 'nequip':
        from nequip.ase import nequip_calculator
                        
        import torch
        torch.set_default_dtype(torch.float64)

        # check cuda availability
        if torch.cuda.is_available():
            single_print("CUDA is available!")
            num_devices = torch.cuda.device_count()
            single_print(f"Number of CUDA devices: {num_devices}")

            for i in range(num_devices):
                device_name = torch.cuda.get_device_name(i)
                single_print(f"Device {i}: {device_name}")

            current_device = torch.cuda.current_device()
            single_print(f"Current device index: {current_device}")
        else:
            num_devices = 0
            single_print("CUDA is not available.")

        for index_nmodel in range(inputs.nmodel):
            for index_nstep in range(inputs.nstep):
                dply_model = f'deployed-model_{index_nmodel}_{index_nstep}.pth'
                if os.path.exists(f'{workpath}/{dply_model}'):
                    single_print(f'\t\tFound the deployed model: {dply_model}')
                    if num_devices > 0:
                        index_totalmodel = index_nmodel*int(inputs.nstep) + index_nstep
                        device_index = index_totalmodel%num_devices
                        device_name = f"{inputs.device}:{device_index}"
                        single_print(f"\t\tLoad to device: {device_name}")
                        inputs.calc_MLIP.append(
                                nequip_calculator.NequIPCalculator.from_deployed_model(f'{workpath}/{dply_model}', device=device_name)
                        )
                    else:
                        inputs.calc_MLIP.append(
                            nequip_calculator.NequIPCalculator.from_deployed_model(f'{workpath}/{dply_model}', device=inputs.device)
                        )
                else:
                    # If there is no model, turn on the termination signal
                    single_print(f'\t\tCannot find the model: {dply_model}')
                    signal = 1

        # ensemble_calculate shares one neighbor list across the ensemble,
        # which is only valid if every model has the same r_max and the
        # same TypeMapper transform.
        if signal == 0 and len(inputs.calc_MLIP) > 1:
            ref = inputs.calc_MLIP[0]
            for c in inputs.calc_MLIP[1:]:
                assert c.r_max == ref.r_max, \
                    'Ensemble models have different r_max; ensemble_calculate cannot share neighbor list'
                assert type(c.transform) is type(ref.transform), \
                    'Ensemble models have different TypeMapper types'

    elif inputs.MLIP == 'so3krates':
        from glp import instantiate
        from glp.ase import Calculator
        for index_nmodel in range(inputs.nmodel):
            for index_nstep in range(inputs.nstep):
                potential_dict = {"mlff": {"folder": f"{workpath}/deployed-model_{index_nmodel}_{index_nstep}"}}
                get_calculator = instantiate.get_calculator(potential_dict, {"atom_pair": {"skin": inputs.skin}})
                inputs.calc_MLIP.append(Calculator(get_calculator))

    # from ase.io.aims import read_aims
    # ref = read_aims('/u/kkang/scratch/r_ZrO_ALmoMD_so3krates/4_ALmoMD/geometry.in.supercell')

    # for calculator in inputs.calc_MLIP:
    #     ref.calc = calculator
    #     print(ref.get_potential_energy())

    # Check the termination signal
    if signal == 1:
        single_print('[Termi]\tSome training processes are not finished.')
        sys.exit()

    return inputs
