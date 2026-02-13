#!/usr/bin/env python
# coding: utf-8

import numpy as np
wc_res = 1190
wc_list = np.linspace(wc_res-500,wc_res+500,50)
import os
for j, wc in enumerate(wc_list):
    if not os.path.exists(str(j+1)):
        os.mkdir(str(j+1))
    with open(f"{j+1}/frequency.txt","w") as f:
        f.write(f'{wc} \n')
    with open(f"{j+1}/submitjob.sh","w") as f:
        f.write("""#!/bin/bash

#SBATCH --nodes=1                        
#SBATCH --ntasks-per-node=1              
#SBATCH --cpus-per-task=2               
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1GB
#SBATCH --job-name=mash-qph
#SBATCH --array=1-40

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python mash_qph.py $SLURM_ARRAY_TASK_ID
""")

