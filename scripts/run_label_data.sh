#!/bin/bash
# Job name
#SBATCH --job-name=image_segmentation
#SBATCH --partition=normal  # Use normal CPU partition

# Resources: single task, multiple CPU cores if needed
#SBATCH --ntasks=1                          # Number of tasks
#SBATCH --cpus-per-task=8                   # Adjust CPUs as needed
#SBATCH --mem=16G                           # Memory per task

# Set the runtime duration
#SBATCH --time=01:00:00  # HH:MM:SS

# Log files
#SBATCH --output=./logs/output_image_seg_%j.slurmlog
#SBATCH --error=./logs/error_image_seg_%j.slurmlog

# Check if script arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: sbatch $0 <input_folder> <output_folder> <num_workers>"
    exit 1
fi

INPUT_FOLDER=$1
OUTPUT_FOLDER=$2
NUM_WORKERS=$3

# Activate virtual environment
source ".venv/bin/activate"

# Optional: remove __pycache__
# rm -r __pycache__

# Run the Python preprocessing script with user-specified number of workers
python label_data_par.py "$INPUT_FOLDER" "$OUTPUT_FOLDER" -t "$NUM_WORKERS"
