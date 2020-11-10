#!/bin/bash
#SBATCH --job-name=ir_class_f1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64000M
#SBATCH --time=2-00:00
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1 
#SBATCH --output=%N-%j.out 
#SBATCH --account=rrg-hamarneh

module load python/3.7
source ~/ENV/bin/activate
python ir_data_densenet_imhistnet_fold1.py
