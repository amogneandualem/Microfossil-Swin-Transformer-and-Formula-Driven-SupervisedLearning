#!/bin/bash

# --- 1. EXPLICITLY REQUEST COMPATIBLE GPU (A800) ---
# This is the single, non-conflicting line that fixes the CUDA crash.
# It requests 1 GPU of the A800 type.
#SBATCH --gres=gpu:a800:1

# --- 2. JOB CONFIGURATION ---
#SBATCH --job-name=Microfossil_ViT
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 

# --- 3. CPU/MEMORY/TIME CONFIGURATION ---
# We are using 12 DataLoader workers in the Python script, 
# so requesting 12 CPUs here is more appropriate for maximum speed.
#SBATCH --cpus-per-task=12 
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# --- 4. PARTITION/QOS (If needed) ---
# Only include these if necessary for access control:
#SBATCH --partition=gpu
#SBATCH --qos=gpunormal 
# Note: Changing back to 'gpunormal' as 'long' is often a high-demand QOS.

# --- Load the necessary environment/modules ---
source /aifs/user/home/amogneandualem/miniconda3/bin/activate microfossil_fixed

# --- Run the Python script ---
echo "Starting training script at $(date)"
python train_microfossil_vit.py
echo "Finished training script at $(date)"