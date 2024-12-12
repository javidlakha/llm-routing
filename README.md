# LLM Routing under Cost and Latency Constraints

## Installation

``` bash
# Install virtual environment
python3.12 -m venv venv

# Activate virtual environment
source venv/bin/activate
echo "Python interpreter: $(which python)"

# Install required packages
pip install -r requirements.txt
```

### Environment variables

Set the following environment variables. A convenient place to do this is in `venv/bin/activate`.

```bash
  # HuggingFace - to download Llama 3 base models
  export HUGGING_FACE_HUB_TOKEN="[...]"

  # OpenAI - to evaluate generated responses
  export OPENAI_API_KEY="[...]"

  # Weights and Biases - for monitoring
  export WANDB_API_KEY="[...]"  
```

## Usage

Train response quality and length predictors
```bash
sbatch classifiers/quality/quality.sh
sbatch classifiers/length/length.sh
```

Run experiments
```bash
sbatch llm_routing/run.sh
```
