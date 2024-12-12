#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --account=kempner_emalach_lab
#SBATCH --output=outputs/experiment-%j/logs.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --mem=256GB
#SBATCH --partition=kempner_h100
#SBATCH --constraint=h100
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

# Llama 3.2 Instruct (1B) - vLLM
CUDA_VISIBLE_DEVICES=2 vllm serve "meta-llama/Llama-3.2-1B-Instruct" \
    --gpu-memory-utilization 0.8 \
    --max-model-len 1024 \
    --tensor-parallel-size 1 \
    --port 8000 > outputs/experiment-$SLURM_JOB_ID/llama_1b.out 2>&1 &
VLLM_PID_LLAMA_1B=$!

# Llama 3.1 Instruct (70B) - vLLM
CUDA_VISIBLE_DEVICES=0,1 vllm serve "meta-llama/Llama-3.1-70B-Instruct" \
    --max-model-len 1024 \
    --tensor-parallel-size 2 \
    --port 8001 > outputs/experiment-$SLURM_JOB_ID/llama_70b.out 2>&1 &
VLLM_PID_LLAMA_70B=$!

# Run experiment
CUDA_VISIBLE_DEVICES=2 python llm_routing/run.py > outputs/experiment-$SLURM_JOB_ID/client.out 2>&1

kill $VLLM_PID_LLAMA_1B $VLLM_PID_LLAMA_70B
