#!/bin/bash
#SBATCH --job-name=rad_ir_seg
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
python ssl_radiopedia_to_ir_data.py