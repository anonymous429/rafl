#!/bin/bash

#SBATCH --mem=16G
#SBATCH --time=3-00:00:00

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

#SBATCH --output=res40-50r.out
#SBATCH --error=res40-50r.err
#SBATCH --job-name=res40-50r
#SBATCH --exclude=singularity, amp-4, amp-6

echo "Loading modules"

module load miniconda3/4.3.30-qdauveb
source activate /work/LAS/jannesar-lab/sixing/.conda/envs/rafl

cd /work/LAS/jannesar-lab/yusx/rafl/src

echo "Starting"
python eval.py --cf config/cifar100/resnet/100clients40-50.yaml
echo "Terminated"