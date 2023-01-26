#!/bin/bash

#SBATCH --mem=16G
#SBATCH --time=3-00:00:00

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

#SBATCH --output=100c40r.out
#SBATCH --error=100c40r.err
#SBATCH --job-name=100c40r
# SBATCH --exclude=singularity,amp-4,amp-6

echo "Loading modules"

module load miniconda3/4.3.30-qdauveb
source activate /work/LAS/jannesar-lab/sixing/.conda/envs/rafl

cd /work/LAS/jannesar-lab/yusx/rafl/src

echo "Starting"
python eval.py --cf config/cifar10/resnet/reset/100c40r.yaml
echo "Terminated"