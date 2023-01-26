#!/bin/bash

#SBATCH --mem=16G
#SBATCH --time=4-10:00:00

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

#SBATCH --output=res70-80r.out
#SBATCH --error=res70-80r.out
#SBATCH --job-name=res70-80r.out
#SBATCH --exclude=singularity,amp-4,amp-6

echo "Loading modules"

module load miniconda3/4.3.30-qdauveb
source activate /work/LAS/jannesar-lab/sixing/.conda/envs/rafl

cd /work/LAS/jannesar-lab/yusx/rafl/src

echo "Starting"
python eval.py --cf config/cifar100/resnet/100clients70-80.yaml
echo "Terminated"