#!/bin/bash

#SBATCH --mem=16G
#SBATCH --time=0-10:00:00

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

#SBATCH --output=tinyimagenet.out
#SBATCH --error=tinyimagenet.out
#SBATCH --job-name=tinyimagenet.out
#SBATCH --exclude=singularity,amp-4,amp-6

echo "Loading modules"

module load miniconda3/4.3.30-qdauveb
source activate /work/LAS/jannesar-lab/sixing/.conda/envs/rafl

cd /work/LAS/jannesar-lab/yusx/rafl/src

echo "Starting"
python eval.py --cf config/debug.yaml
echo "Terminated"