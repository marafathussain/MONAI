#!/bin/bash
#SBATCH --job-name=eu_iran_class
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32000M
#SBATCH --time=08:00:00
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4 
#SBATCH --output=%N-%j.out 
#SBATCH --account=rrg-hamarneh

module load python/3.7
source ~/ENV/bin/activate
python eu_trained_iran_inferred.py