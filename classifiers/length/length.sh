#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --account=kempner_emalach_lab
#SBATCH --output=outputs/length-predictor-%j/logs.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --mem=256GB
#SBATCH --partition=kempner
#SBATCH --constraint=a100
#SBATCH --array=1-1

echo "Starting job $SLURM_JOB_ID"

# Load modules
module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01

# Load API keys
source ~/.bashrc

# Activate virtual environment
source venv/bin/activate
echo "Python interpreter: $(which python)"

# Add current directory to PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"
echo $PYTHONPATH

python classifiers/length/length.py
