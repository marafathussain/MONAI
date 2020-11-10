#!/bin/bash
#SBATCH --job-name=ir_class_d2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32000M
#SBATCH --time=2-00:00
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1 
#SBATCH --output=%N-%j.out 
#SBATCH --account=rrg-hamarneh

module load python/3.7
source ~/ENV/bin/activate
python ir_data_densenet_no_avg_pool_infection_only.py