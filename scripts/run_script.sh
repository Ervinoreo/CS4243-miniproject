#!/bin/bash
# Job name
#SBATCH --job-name=image_segmentation
#SBATCH --partition=gpu

## GPU type constraint: A100-40 on xgph node or H100-96 on xgpi node
#SBATCH --constraint=xgph # Use A100-40 GPU
## #SBATCH --constraint=xgpi # Use H100-96 GPU

## Request the appropriate GPU:
#SBATCH --gres=gpu:a100-40:1  # Use A100-40 GPU
## #SBATCH --gres=gpu:h100-47:1  # Use H100-96 GPU

## Set the runtime duration (adjust based on how long you expect the job to take)
#SBATCH --time=01:00:00  # HH:MM:SS (change as necessary)

# Resources: single task, single CPU core, 20 GB of memory
#SBATCH --ntasks=1                          # Number of tasks (1 task)
#SBATCH --cpus-per-task=1                   # Number of CPU cores per task
#SBATCH --mem=16G                           # 20GB of memory per task

## Log file names for output and error
#SBATCH --output=./logs/output_%j.slurmlog
#SBATCH --error=./logs/error_%j.slurmlog

# Check if script argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch $0 <path_to_python_script>"
    exit 1
fi

SCRIPT_PATH=$1

# Display GPU status
nvidia-smi

# Activate virtual environment
source ".venv/bin/activate"

# Remove __pycache__ if needed
# rm -r __pycache__

# Run the Python script provided as an argument
python "$SCRIPT_PATH"
