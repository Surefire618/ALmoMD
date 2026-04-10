import os
import sys
from libs.lib_util     import single_print


def ensemble_calculate(calculators, atoms):
    """Run an ensemble of NequIPCalculators on `atoms`, sharing the
    AtomicData (neighbor list + type mapping) across all of them.

    Bypasses ASE's per-call Calculator.calculate so the GIL-held
    AtomicData.from_ase / TypeMapper pipeline runs once instead of N
    times. Each calculator's `results` dict and `atoms` snapshot are
    populated as if Calculator.calculate had been called, so any later
    get_potential_energy / get_forces / get_potential_energies /
    get_stress on the same atoms is a cache hit.

    All calculators must share the same r_max and transform; this is
    asserted in load_model after the calculators are built.
    """
    import torch
    from nequip.data import AtomicData, AtomicDataDict
    from ase.stress import full_3x3_to_voigt_6_stress

    ref = calculators[0]

    data = AtomicData.from_ase(atoms=atoms, r_max=ref.r_max)
    for k in AtomicDataDict.ALL_ENERGY_KEYS:
        if k in data:
            del data[k]
    data = ref.transform(data)
    data_cpu = AtomicData.to_AtomicDataDict(data)

    energies = []
    forces = []
    for calc in calculators:
        data_dev = {
            k: (v.to(calc.device) if torch.is_tensor(v) else v)
            for k, v in data_cpu.items()
        }
        out = calc.model(data_dev)

        results = {}
        if AtomicDataDict.TOTAL_ENERGY_KEY in out:
            results['energy'] = calc.energy_units_to_eV * (
                out[AtomicDataDict.TOTAL_ENERGY_KEY]
                .detach().cpu().numpy().reshape(tuple())
            )
            results['free_energy'] = results['energy']
        if AtomicDataDict.PER_ATOM_ENERGY_KEY in out:
            results['energies'] = calc.energy_units_to_eV * (
                out[AtomicDataDict.PER_ATOM_ENERGY_KEY]
                .detach().squeeze(-1).cpu().numpy()
            )
        if AtomicDataDict.FORCE_KEY in out:
            results['forces'] = (
                calc.energy_units_to_eV / calc.length_units_to_A
            ) * out[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
        if AtomicDataDict.STRESS_KEY in out:
            stress = out[AtomicDataDict.STRESS_KEY].detach().cpu().numpy()
            stress = stress.reshape(3, 3) * (
                calc.energy_units_to_eV / calc.length_units_to_A ** 3
            )
            results['stress'] = full_3x3_to_voigt_6_stress(stress)

        calc.results = results
        calc.atoms = atoms.copy()

        energies.append(float(results['energy']))
        forces.append(results['forces'])

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
