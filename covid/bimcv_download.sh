#!/bin/bash
#SBATCH --job-name=bimcv_download
#SBATCH --nodes=1
#SBATCH --mem=16000M
#SBATCH --time=2-00:00
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1
#SBATCH --output=%N-%j.out 
#SBATCH --account=rrg-hamarneh

module load python/3.7
source ~/ENV/bin/activate
python download.py