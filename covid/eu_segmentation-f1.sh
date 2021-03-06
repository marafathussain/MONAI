#!/bin/bash
#SBATCH --job-name=f1_eu_seg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64000M
#SBATCH --time=2-00:00
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4 
#SBATCH --output=%N-%j.out 
#SBATCH --account=rrg-hamarneh

module load python/3.7
source ~/ENV/bin/activate
python eu_96_data_dict_fold1.py