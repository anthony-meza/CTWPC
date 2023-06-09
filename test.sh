#!/bin/bash
#SBATCH --partition=compute
#SBATCH --mail-type=END
#SBATCH --mail-user=anthony.meza@whoi.edu
#SBATCH --ntasks=1
#SBATCH --mem=1gb
#SCATCH --time=07:00:00
#SBATCH --job-name jupyter_ameza
#SBATCH --output=test.log
# get tunneling info
XDG_RUNTIME_DIR=""
# Alternatively, activate your desired environment in the command line before you run this # script.
module load anaconda
source activate notebook_env
# DON'T USE ADDRESS BELOW.
# DO USE TOKEN BELOW

python test.py
