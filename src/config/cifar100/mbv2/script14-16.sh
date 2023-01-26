#!/bin/bash

#SBATCH --mem=32G
#SBATCH --time=6-00:00:00

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

#SBATCH --output=100mbv2(14-16).out
#SBATCH --error=100mbv2(14-16).err
#SBATCH --job-name=100m2(14-16)
#SBATCH --exclude=singularity


echo "Loading modules"

module load miniconda3/4.3.30-qdauveb
source activate /work/LAS/jannesar-lab/sixing/.conda/envs/rafl

cd /work/LAS/jannesar-lab/yusx/rafl/src

echo "Starting"
python eval.py --cf config/cifar100/mbv2/100reset14-16.yaml
echo "Terminated"