#!/bin/bash
#SBATCH --job-name=ctk_ea
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-3%3

# --- 1. EA Parameters (edit these) ---
RUN_NAME="test"
POPULATION_SIZE=10
NUM_GENERATIONS=10

# Flags (true/false)
QUIET=false
NO_TOURNAMENT=false      # -T: disable tournament selection
DO_PENALTY
NO_FITNESS_NOVELTY=false # -N: disable novelty in fitness
NO_FITNESS_SPEED=false   # -S: disable speed in fitness
STORE_NOVELTY=false      # -n: store novelty (even if not in fitness)
STORE_SPEED=false        # -s: store speed (even if not in fitness)

# --- 2. Create output dir, redirect all output there ---
TASK_ID=$(printf '%04d' "$SLURM_ARRAY_TASK_ID")
OUTPUT_DIR="__data__/${RUN_NAME}_${TASK_ID}"
mkdir -p "$OUTPUT_DIR"

# Redirect all stdout/stderr to output dir from now on
exec > "$OUTPUT_DIR/logs/slurm.out" 2> "$OUTPUT_DIR/logs/slurm.err"

# --- 2. Setup Environment ---
# Load Modules (Standard Snellius 2023 stack)
module purge
module load 2023
# module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

# System Optimizations
# export CUDA_VISIBLE_DEVICES=0
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# --- 2. GPU Monitoring (Background with Safety Trap) ---
# GPU_LOG="$LOG_DIR/gpu_mon_task_${SLURM_ARRAY_TASK_ID}.csv"

# Start nvidia-smi in the background
# nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw \
#     --format=csv --loop=10 > "$GPU_LOG" &
# MONITOR_PID=$!

# TRAP: Automatically kill the monitor when script exits (success or fail)
# trap "kill $MONITOR_PID" EXIT

# --- 3. Set Variables for Python ---
# We export these so Python can read them via os.environ

# Use the Array Task ID as the unique Seed
# export AGENT_SEED=$SLURM_ARRAY_TASK_ID

# Define where to save files
# export AGENT_OUTPUT_DIR="$LOG_DIR/run_${SLURM_ARRAY_TASK_ID}"
# mkdir -p "$AGENT_OUTPUT_DIR"

# echo "=========================================="
# echo "Starting Run with Seed: $AGENT_SEED"
# echo "Output Directory: $AGENT_OUTPUT_DIR"
# echo "=========================================="

# --- 4. Run Python Script ---
FLAGS=""
$QUIET && FLAGS="$FLAGS -q"
$NO_TOURNAMENT && FLAGS="$FLAGS -T"
$NO_FITNESS_NOVELTY && FLAGS="$FLAGS -N"
$NO_FITNESS_SPEED && FLAGS="$FLAGS -S"
$STORE_NOVELTY && FLAGS="$FLAGS -n"
$STORE_SPEED && FLAGS="$FLAGS -s"

uv run run.py -r "$RUN_NAME" -p "$POPULATION_SIZE" -g "$NUM_GENERATIONS" -o "$OUTPUT_DIR" $FLAGS

echo "Run $SLURM_ARRAY_TASK_ID finished. Output: $OUTPUT_DIR"
