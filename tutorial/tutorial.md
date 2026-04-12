#Tutorials


## Initialization

We are going to have a tutorial of the ALmoMD with an example of CuI.

### Ground truth
But the actual implementation of ALmoMD with DFT is too demanding for the tutorial purpose due to the large amount of supercell calculations via DFT. Therefore, in this tutorial, we are going to have a pretrained MLIP model as a ground truth instead of DFT. To do that, you need __REFER__ instead of __DFT_INPUTS__.

__REFER__: A directory containing the pretrained MLIP depolyed model (deployed-model\_0\_0.pth). 

Accodingly, your ALmoMD inputs should be modified as below.

```
output_format : nequip             # Make the pretrained NequIP model as the ground truth
E_gs          : -0.142613646819514 # New corresponding reference potential energy (Energy of the geometry.in.supercell)
```

### Train initial MLIP models
1) Split the aiMD trajectories into training and testing data. This can be implemented by a command of `almomd utils split (# of testing data) (E_gs)`. In this practice, we will use 100 testing data.

```
almomd utils split 100 -0.142613646819514
```

It will create two files __trajectory_train.son__, __trajectory_test.son__, and a directory __MODEL__ containing __data_test.npz__.

2) Create the training inputs for initial MLIP modes.

```
almomd init
```

It will create a directory of __300K-0bar\_0__ inside of __MODEL__. 

```
cd MODEL/300-bar_0
```

You will find 3 training data (__data-train\_\*.npz__), and 6 NequIP inputs (__input\_\*\_\*.yaml__) and corresponding job scripts (__job-nequip-gpu
\_\*\_\*.slurm__). This is because you assign 3 subsampling and 2 random initialization in __input.in__, leading to a total of 6 (=2*3) different MLIP models.

3) Submit your job scripts to train MLIP models.

```
sbatch job-nequip-gpu\_0.slurm; sbatch job-nequip-gpu\_1.slurm; sbatch job-nequip-gpu\_2.slurm; sbatch job-nequip-gpu\_3.slurm; sbatch job-nequip-gpu\_4.slurm; sbatch job-nequip-gpu\_5.slurm
```

4) When your training is done, you will get deployed MLIP models (__depolyed-model\_\*\_\*.pth__).


## Active Learning Procedure
The active learning iterative loop in the ALmoMD consists of three major steps (MLIP exploration, DFT calculation, and MLIP training).

### MLIP exploration
When you have MLIP models, the ALmoMD will explore the configurational space via MLIP-MD. This can be conducted by submit your __job-cont.slurm__.

```
sbatch job-cont.slurm
```

It will generate many files and directories. But, __almomd.out__, __result.txt__, and __UNCERT/uncertainty-300K-0bar\_*.txt__ are important files that users know.

1) __almomd.out__: It shows the overall process of the ALmoMD.
   
2) __result.txt__: It contains the testing results and their MLIP uncertainty at each active learning step.

3) __UNCERT/uncertainty-300K-0bar\_*.txt__: It records the result of the MLIP-MD steps. You can recognize which MD snapshots are sampled.

When it samples all data, it will create a directory of __CALC/300K-0bar\_\*__, where all DFT inputs for the sampled snapshots are prepared.

### DFT calculation
In each iteration, you need to go into the most recent __CALC/300K-0bar\_\*__.

```
cd CALC/300-0bar_1
```

You need to submit all job scripts.

```
sbatch job-vibes_0.slurm; sbatch job-vibes_1.slurm; sbatch job-vibes_2.slurm; sbatch job-vibes_3.slurm; sbatch job-vibes_4.slurm; sbatch job-vibes_5.slurm; sbatch job-vibes_6.slurm; sbatch job-vibes_7.slurm; sbatch job-vibes_8.slurm; sbatch job-vibes_9.slurm; sbatch job-vibes_10.slurm; sbatch job-vibes_11.slurm; sbatch job-vibes_12.slurm; sbatch job-vibes_13.slurm; sbatch job-vibes_14.slurm; sbatch job-vibes_15.slurm
```

### MLIP training
Once all DFT calculations are finished, go back to main directory where __result.txt__ exists.

```
almomd gen
```

This will add all new DFT outcomes into the training data. The new training data, inputs, and corresponding job scripts are generated in the most recent __MODEL/300K-0bar\_\*__. Then, submit all job scripts.

```
sbatch job-nequip-gpu\_0.slurm; sbatch job-nequip-gpu\_1.slurm; sbatch job-nequip-gpu\_2.slurm; sbatch job-nequip-gpu\_3.slurm; sbatch job-nequip-gpu\_4.slurm; sbatch job-nequip-gpu\_5.slurm
```

When your training is done, you will get deployed MLIP models (__depolyed-model\_\*\_\*.pth__). Then, go back to __MLIP exploration__ section to complete the loop.


## Optional: automatic job submission

By default ALmoMD only **writes** the DFT and training job scripts to disk — it does not submit them. You run every `sbatch` yourself. That is the safe default for a first-time user or a shared cluster.

If instead you want ALmoMD to chain the whole __cont → DFT → train → cont → ...__ loop on its own, uncomment **four** lines across three files. The chain is built out of two parts: each step auto-submits the jobs it just wrote, and each step uses SLURM job dependencies to kick off the next step once those jobs finish.

### Enable full auto mode (four lines to uncomment)

1) __libs/lib_dft.py__ — submit the DFT jobs written by `almomd cont`:
```
# subprocess.run([inputs.job_command, job_script])
```

2) __scripts/lib_run_dft_cont.py__ — after submitting the DFT jobs, submit `job-gen.slurm` with a SLURM dependency on them:
```
# if inputs.rank == 0:
#     job_dependency('gen', inputs.num_calc)
```

3) __libs/lib_train.py__ — submit the training jobs written by `almomd gen`:
```
# subprocess.run([inputs.job_command, job_script]);
```

4) __scripts/lib_run_dft_gen.py__ — after submitting the training jobs, submit the next `job-cont.slurm` with a SLURM dependency on them:
```
# job_dependency('cont', inputs.num_mdl_calc)
```

All four need to be uncommented together. Enabling only some of them leaves the chain broken: e.g. if `#4` fires but `#3` didn't submit anything, `job_dependency` has no job IDs to depend on and will pass an empty `--dependency=afterany:` to sbatch.

### What the auto loop looks like once it's running

With all four enabled, you start the loop yourself exactly once:
```
sbatch job-cont.slurm
```

From there each iteration flows like this:
```
  job-cont.slurm
      ↓  runs `almomd cont` (MLMD exploration)
      ↓  #1 sbatch job-vibes_*.slurm         (DFT jobs for sampled snapshots)
      ↓  #2 sbatch job-gen.slurm --dep=DFT   (chained)
  job-gen.slurm
      ↓  runs `almomd gen` (ingest DFT, write NequIP inputs)
      ↓  #3 sbatch job-nequip-gpu_*.slurm    (training jobs)
      ↓  #4 sbatch job-cont.slurm --dep=train (chained — next iteration)
  next iteration's job-cont.slurm starts when training finishes
      ↓  ...
```

### `almomd cont` may still need manual resubmissions

On some clusters a single SLURM walltime is shorter than the MLMD sampling window (e.g. ~2 ns for our runs), so `job-cont.slurm` will be killed before the iteration's sampling quota is reached. `almomd cont` is restart-safe (state lives in `UNCERT/`, `TEMPORARY/`, and `result.txt`), so you just resubmit it and MLMD resumes. The auto-chain (`#1`/`#2`) only fires once the quota is actually reached.

# Contents
- [Back to Home](https://keysongkang.github.io/ALmoMD/)
- [Installation Guide](../docs/installation.md)
- [User Manuals](../docs/documentation.md) ([Input Files](../docs/doc_input_file.md), [Input Parameters](../docs/doc_input_para.md))
- [Tutorials](tutorial.md)
