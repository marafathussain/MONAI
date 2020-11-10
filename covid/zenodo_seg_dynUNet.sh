#!/bin/bash
#SBATCH --job-name=zen_seg_dUnet
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64000M
#SBATCH --time=2-00:00
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4 
#SBATCH --output=%N-%j.out 
#SBATCH --account=rrg-hamarneh

module load python/3.7
source ~/ENV/bin/activate
python zenodo_dynUNet_seg.py