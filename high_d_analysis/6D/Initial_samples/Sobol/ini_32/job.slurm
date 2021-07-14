#!/bin/bash
#SBATCH --job-name=py-job1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=joonjaer@princeton.edu

module purge
module load anaconda3/2020.11
conda activate surrogate

python new_onefile.py