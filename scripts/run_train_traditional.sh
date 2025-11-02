#!/bin/bash
# Job name
#SBATCH --job-name=image_segmentation
#SBATCH --partition=gpu

## GPU type constraint: A100-40 on xgph node or H100-96 on xgpi node
## #SBATCH --constraint=xgph # Use A100-40 GPU
#SBATCH --constraint=xgpi # Use H100-96 GPU

## Request the appropriate GPU:
## #SBATCH --gres=gpu:a100-40:1  # Use A100-40 GPU
#SBATCH --gres=gpu:h100-47:1  # Use H100-96 GPU

## Set the runtime duration (adjust based on how long you expect the job to take)
#SBATCH --time=02:59:00  # HH:MM:SS (change as necessary)

# Resources: single task, single CPU core, 16 GB of memory
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G

## Log file names for output and error
#SBATCH --output=./logs/output_%j.slurmlog
#SBATCH --error=./logs/error_%j.slurmlog

# Check if both arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch $0 <path_to_python_script> <input_path>"
    exit 1
fi

SCRIPT_PATH=$1
INPUT_PATH=$2

echo "Running script: $SCRIPT_PATH"
echo "Using input: $INPUT_PATH"

# Display GPU status
nvidia-smi

# Activate virtual environment
source ".venv/bin/activate"

# Optional: Remove __pycache__
# rm -r __pycache__

# Run the Python script with the input path as an argument
python "$SCRIPT_PATH" "$INPUT_PATH" --epochs 1000 --hidden_sizes 4096 2048 1024 512 256 --use_spatial --use_freq --use_texture

