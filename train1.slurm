#!/bin/bash
#SBATCH --job-name=wav2vec_train
#SBATCH --output=wav2vec_train-%j.out
#SBATCH --error=wav2vec_train-%j.err
#SBATCH --account=ohalad
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# ✅ Path to Singularity image
SINGULARITY_IMAGE="/contrib/hammond/nemo2.sif"

echo "Starting Wav2Vec2 fine-tuning..."

# ✅ Run training
singularity exec --nv $SINGULARITY_IMAGE python3 /home/u5/shawnabirnbaum/train1.py
