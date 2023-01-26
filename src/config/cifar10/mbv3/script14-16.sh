#!/bin/bash

#SBATCH --mem=32G
#SBATCH --time=6-00:00:00

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

#SBATCH --output=10mbv3(14-16).out
#SBATCH --error=10mbv3(14-16).err
#SBATCH --job-name=10m3(14-16)
#SBATCH --exclude=singularity


echo "Loading modules"

module load miniconda3/4.3.30-qdauveb
source activate /work/LAS/jannesar-lab/sixing/.conda/envs/rafl

cd /work/LAS/jannesar-lab/yusx/rafl/src

echo "Starting"
python eval.py --cf config/cifar10/mbv3/100reset14-16.yaml
echo "Terminated"