#!/bin/bash

#SBATCH --job-name=BERTTest     # Job name
#SBATCH --output=job.%j.out     # Name of the output file
#SBATCH --cpus-per-task=2       # No. CPUs
#SBATCH --gres=gpu              # Request GPU
#SBATCH --time=02:30:00         # Max run time
#SBATCH --partition=brown       # Desktop machines
#SBATCH --mail-type=END,FAIL    # Send Email if it Fails or when it Ends

# Display start time
date
# Activate conda environment
source activate bachelor
# Run the model
python test.py
# Display end time
date
