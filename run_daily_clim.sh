#!/bin/bash
#SBATCH --partition=compute
#SBATCH --mail-type=END
#SBATCH --mail-user=anthony.meza@whoi.edu
#SBATCH --ntasks=1
#SBATCH --mem=16gb
#SCATCH --time=07:00:00
#SBATCH --job-name jupyter_ameza
#SBATCH --output=log-script-%j.log
# get tunneling info
XDG_RUNTIME_DIR=""
module load anaconda
source activate notebook_env
# DON'T USE ADDRESS BELOW.
# DO USE TOKEN BELOW
python CoastalWaves_Climatology.py
