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
python preprocess/process_unclear_images.py "$INPUT_FOLDER" -o "$OUTPUT_FOLDER" -w 250 -b 5 -k 3 -s 3 -m 40 -t 1.1 -p 3 -c 30 -mul 2.0 --size-ratio-threshold 0.4 --large-box-ratio 2.5 --wide-box-color-threshold 30 --workers "$NUM_WORKERS"

