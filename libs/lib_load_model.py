import os
import sys
from libs.lib_util     import single_print

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
