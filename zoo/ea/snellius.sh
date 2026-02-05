#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=192
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --time=03:30:00
#SBATCH --array=1-10%1



# --- 1. EA Parameters (edit these) ---
RUN_NAME="penalty"
POPULATION_SIZE=500
NUM_GENERATIONS=100

# Flags (true/false)
QUIET=false
NO_TOURNAMENT=false      # -T: disable tournament selection
PENALTY=true            # -P: enable penalty
NO_FITNESS_NOVELTY=false # -N: disable novelty in fitness
NO_FITNESS_SPEED=false   # -S: disable speed in fitness
STORE_NOVELTY=false      # -n: store novelty (even if not in fitness)
STORE_SPEED=false        # -s: store speed (even if not in fitness)

# --- 2. Setup directories ---
TASK_ID=$(printf '%04d' "$SLURM_ARRAY_TASK_ID")
FINAL_OUTPUT_DIR="__data__/${RUN_NAME}_${TASK_ID}"
LOCAL_OUTPUT_DIR="$TMPDIR/${FINAL_OUTPUT_DIR}"

# Create final output dir for logs (on network storage - readable during run)
mkdir -p "$FINAL_OUTPUT_DIR/__slurm__"

# Redirect all stdout/stderr to final output dir (so you can read __slurm__ during run)
exec > "$FINAL_OUTPUT_DIR/__slurm__/slurm.out" 2> "$FINAL_OUTPUT_DIR/__slurm__/slurm.err"

echo "=== SSD-accelerated run ==="
echo "Local SSD dir: $LOCAL_OUTPUT_DIR"
echo "Final output:  $FINAL_OUTPUT_DIR"
echo "==========================="

# --- 3. Setup Environment ---
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# --- ADD THESE LINES HERE ---
unset DISPLAY
unset WAYLAND_DISPLAY
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
# ----------------------------


# Test which GL backend is available
echo "Testing MuJoCo GL backends..."
uv run python -c "import os; os.environ['MUJOCO_GL']='egl'; import mujoco; print('EGL works')" 2>/dev/null && echo "✓ EGL available" || echo "✗ EGL failed"
uv run python -c "import os; os.environ['MUJOCO_GL']='osmesa'; import mujoco; print('OSMesa works')" 2>/dev/null && echo "✓ OSMesa available" || echo "✗ OSMesa failed"

# --- 4. Build flags ---
FLAGS=""
$QUIET && FLAGS="$FLAGS -q"
$NO_TOURNAMENT && FLAGS="$FLAGS -T"
$PENALTY && FLAGS="$FLAGS -P"
$NO_FITNESS_NOVELTY && FLAGS="$FLAGS -N"
$NO_FITNESS_SPEED && FLAGS="$FLAGS -S"
$STORE_NOVELTY && FLAGS="$FLAGS -n"
$STORE_SPEED && FLAGS="$FLAGS -s"

# --- 5. Run on local SSD (fast I/O for database) ---
echo "Starting EA on local SSD..."
uv run run.py -r "$RUN_NAME" -p "$POPULATION_SIZE" -g "$NUM_GENERATIONS" -o "$LOCAL_OUTPUT_DIR" --SEED "$SLURM_ARRAY_TASK_ID" $FLAGS
EXIT_CODE=$?

echo "$(date),--RUN_NAME=$RUN_NAME,...,$FINAL_OUTPUT_DIR" >> run_history.csv

# --- 6. Copy results back to network storage ---
echo "Copying results from SSD to network storage..."
cp -r "$LOCAL_OUTPUT_DIR"/* "$FINAL_OUTPUT_DIR"/
echo "Copy complete."

echo "Run $SLURM_ARRAY_TASK_ID finished with exit code $EXIT_CODE"
echo "Output: $FINAL_OUTPUT_DIR"

exit $EXIT_CODE
