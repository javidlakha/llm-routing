import asyncio
import json
import logging
import os
import random
import time
from asyncio import Queue
from datetime import datetime
from pathlib import Path
from pprint import pformat
from time import sleep
from typing import Any

import numpy as np
import torch
import wandb
from openai import APIConnectionError, AsyncOpenAI
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from utils import convert_to_json, load_jsonl
from llm_routing.statistics import compute_statistics


JOB_ID = os.getenv("SLURM_JOB_ID", random.randint(0, 2**31 - 1))
NUM_CPU = os.cpu_count()
NUM_GPU = torch.cuda.device_count()
DEVICE = "cuda"

SEED = 4161
random.seed(SEED)
np.random.seed(SEED)

LLAMA_1B = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_1B_URL = "http://localhost:8000/v1"
LLAMA_1B_QUALITY_PREDICTOR = "models/quality-predictor-1b"
LLAMA_1B_LENGTH_PREDICTOR = "models/length-predictor-1b"

LLAMA_70B = "meta-llama/Llama-3.1-70B-Instruct"
LLAMA_70B_URL = "http://localhost:8001/v1"
LLAMA_70B_QUALITY_PREDICTOR = "models/quality-predictor-70b"
LLAMA_70B_LENGTH_PREDICTOR = "models/length-predictor-70b"

DATASET = Path("datasets/subset_labelled.jsonl")
OUTPUTS = Path(f"outputs/experiment-{JOB_ID}")

MAX_LENGTH = 512
NUM_TOKEN_BUCKETS = 8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("llm-routing")


class PerfectQualityPredictor:
    """Predicts the quality of each model's response under perfect information"""

    def __init__(self, model_a: str, model_b: str) -> None:
        self.model_a = model_a
        self.model_b = model_b

    def __call__(self, request: dict[str, Any]) -> float:
        strong = int(request[self.model_a]["score"])
        weak = int(request[self.model_b]["score"])
        score = (strong - weak) / 8 + 0.5
        return score


class DistilbertQualityPredictor:
    """Predicts the quality of each model's response using DistilBERT"""

    def __init__(
            self, 
            model_a: str, 
            predictor_a: str,
            model_b: str, 
            predictor_b: str,
        ) -> None:
        self.device = DEVICE

        self.model_a = model_a
        self.tokenizer_a = DistilBertTokenizer.from_pretrained(predictor_a)
        self.predictor_a = DistilBertForSequenceClassification.from_pretrained(predictor_a).to(self.device)
        self.predictor_a.eval()

        self.model_b = model_b
        self.tokenizer_b = DistilBertTokenizer.from_pretrained(predictor_b)
        self.predictor_b = DistilBertForSequenceClassification.from_pretrained(predictor_b).to(self.device)
        self.predictor_b.eval()

    def predict(self, tokenizer, model, prompt: str) -> int:
        """
        Tokenizes the input prompt and get the predicted quality score
        """
        inputs = tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        predictions = torch.argmax(outputs.logits, dim=1)
        return predictions.item() + 1  # Scores are 1-based

    def __call__(self, request: dict[str, dict[str, dict[str, str]]]) -> float:
        """Predicts the quality scores for the responses from model_a and model_b"""
        # Get prompts
        prompt_a = request[self.model_a]["modelInput"]["prompt"]
        prompt_b = request[self.model_b]["modelInput"]["prompt"]

        # Predict scores
        score_a = self.predict(self.tokenizer_a, self.predictor_a, prompt_a)
        score_b = self.predict(self.tokenizer_b, self.predictor_b, prompt_b)

        # Compute final score
        final_score = (score_a - score_b) / 8 + 0.5
        return final_score


class RandomQualityPredictor:
    """'Predicts' the quality of each model's responses randomly"""

    def __init__(self, model_a: str, model_b: str) -> None:
        self.model_a = model_a
        self.model_b = model_b

    def __call__(self, request: dict[str, Any]) -> float:
        return random.uniform(0, 1)


class PerfectTokenPredictor:
    """Predicts the length of each model's response under perfect information"""

    def __init__(self, model_a: str, model_b: str) -> None:
        self.model_a = model_a
        self.model_b = model_b

    def __call__(self, request: dict[str, Any], model: str) -> tuple[int, int]:
        prompt_token_count = int(request[model]["modelOutput"]["prompt_token_count"])
        generation_token_count = int(request[model]["modelOutput"]["generation_token_count"])
        return prompt_token_count, generation_token_count


class DistilbertTokenPredictor:
    def __init__(self, model_a: str, predictor_a_path: str, model_b: str, predictor_b_path: str) -> None:
        """Predicts the length of each model's response using DistilBERT"""
        self.device = DEVICE

        self.model_a = model_a
        self.tokenizer_a = DistilBertTokenizer.from_pretrained(predictor_a_path)
        self.predictor_a = DistilBertForSequenceClassification.from_pretrained(predictor_a_path).to(self.device)
        self.predictor_a.eval()

        self.model_b = model_b
        self.tokenizer_b = DistilBertTokenizer.from_pretrained(predictor_b_path)
        self.predictor_b = DistilBertForSequenceClassification.from_pretrained(predictor_b_path).to(self.device)
        self.predictor_b.eval()

        # Bucket definitions
        self.num_buckets = NUM_TOKEN_BUCKETS
        self.max_length = MAX_LENGTH
        self.bucket_ranges = [int(self.max_length * (i + 1) / self.num_buckets) for i in range(self.num_buckets)]

    def predict_bucket(self, tokenizer, model, prompt: str) -> int:
        inputs = tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        predictions = torch.argmax(outputs.logits, dim=1)
        return predictions.item()

    def bucket_to_token_count(self, bucket: int) -> int:
        """Converts the bucket index to the maximum token count in that bucket."""
        if bucket < 0 or bucket >= self.num_buckets:
            raise ValueError("Invalid bucket index")
        return self.bucket_ranges[bucket]

    def __call__(self, request: dict[str, Any], model: str) -> tuple[int, int]:
        """
        Returns the prompt length and predicts the response lengths for model_a and model_b
        
        Returns a tuple (prompt_token_count, generation_token_count).
        """
        if model == self.model_a:
            tokenizer, predictor = self.tokenizer_a, self.predictor_a
        elif model == self.model_b:
            tokenizer, predictor = self.tokenizer_b, self.predictor_b
        else:
            raise ValueError(f"Model {model} is not recognized.")

        # Extract prompt token count
        prompt_token_count = int(request[model]["modelOutput"]["prompt_token_count"])

        # Predict generation token count
        prompt = request[model]["modelInput"]["prompt"]
        bucket = self.predict_bucket(tokenizer, predictor, prompt)
        generation_token_count = self.bucket_to_token_count(bucket)

        return prompt_token_count, generation_token_count


class ModelClient:
    """A client for interacting with a vLLM model server."""

    def __init__(self, model: str, url: str) -> None:
        self.model = model
        self.url = url
        self.client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=url,
        )

    async def __call__(self, model_input: dict[str, Any]) -> dict[str, Any]:
        """Sends a request to the model and returns the response"""
        # Send request
        prompt = model_input["prompt"]
        max_tokens = model_input.get("max_gen_len", 512)
        temperature = model_input.get("temperature", 0.0)
        top_p = model_input.get("top_p", 1.0)
        response = await self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Extract response
        completion = response.choices[0].text
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        model_output = {
            "generation": completion,
            "prompt_token_count": prompt_tokens,
            "generation_token_count": completion_tokens,
            "stop_reason": response.choices[0].finish_reason,
        }
        return model_output


async def wait_for_models_to_start(model_clients: list[AsyncOpenAI]) -> None:
    """
    Blocks execution until the model represented by each client in
    model_clients is ready to respond to requests.
    """
    while True:
        try:
            for client in model_clients:
                await client.client.models.list()
        except APIConnectionError:
            pass
        else:
            break
        sleep(1)


async def run_experiment(
    dataset_path: Path,
    subset: int,
    arrival_rate: float,
    target_cost: float,
    cost_weight: float,
    target_load: float,
    load_weight: float,
    learning_rate: float,
    quality_predictor: str,
    token_predictor: str,
    run_identifier: str,
):
    config = {
        "job_id": JOB_ID,
        "run_identifier": run_identifier,
        "num_cpu": NUM_CPU,
        "num_gpu": NUM_GPU,
        "seed": SEED,
        "dataset_path": str(dataset_path),
        "subset": subset,
        "arrival_rate": arrival_rate,
        "target_cost": target_cost,
        "cost_weight": cost_weight,
        "target_load": target_load,
        "load_weight": load_weight,
        "learning_rate": learning_rate,
        "quality_predictor": quality_predictor,
        "token_predictor": token_predictor,
    }
    # Log experiment to Weights and Biases
    wandb.init(
        project="llm-routing",
        name=f"experiment-{run_identifier}",
        config=config,
    )
    logger.info(f"Started run {run_identifier} at {datetime.now().isoformat()}")
    logger.info("Parameters:\n%s", pformat(config))

    # Prepare dataset
    logger.info("Loading dataset")
    dataset = list(load_jsonl(str(DATASET)))
    
    # Dataset is already shuffled, we want consistency across experiments
    #random.shuffle(dataset)
    
    dataset = dataset[:subset]

    # Load models
    llama_1b_model = ModelClient(model=LLAMA_1B, url=LLAMA_1B_URL)
    llama_70b_model = ModelClient(model=LLAMA_70B, url=LLAMA_70B_URL)

    # Load the response quality predictor
    if quality_predictor == "distilbert":
        quality_predictor = DistilbertQualityPredictor(
            model_a=LLAMA_70B, 
            predictor_a=LLAMA_70B_QUALITY_PREDICTOR, 
            model_b=LLAMA_1B, 
            predictor_b=LLAMA_1B_QUALITY_PREDICTOR,
        )
    if quality_predictor == "perfect":
        quality_predictor = PerfectQualityPredictor(model_a=LLAMA_70B, model_b=LLAMA_1B)
    elif quality_predictor == "random":
        quality_predictor = RandomQualityPredictor(model_a=LLAMA_70B, model_b=LLAMA_1B)

    # Load the response length predictor
    if token_predictor == "distilbert":
        token_predictor = DistilbertTokenPredictor(
            model_a=LLAMA_70B,
            predictor_a_path=LLAMA_70B_LENGTH_PREDICTOR,
            model_b=LLAMA_1B,
            predictor_b_path=LLAMA_1B_LENGTH_PREDICTOR,
        )
    if token_predictor == "perfect":
        token_predictor = PerfectTokenPredictor(model_a=LLAMA_70B, model_b=LLAMA_1B)

    # Wait for vLLM to be ready (it can take up to 5 minutes to load the weights for Llama 3.1 Instruct (70B))
    logger.info("Waiting for models to start")
    await wait_for_models_to_start([llama_1b_model, llama_70b_model])

    # Track cost and load
    counters = {
        "cost": {LLAMA_70B: 0, LLAMA_1B: 0},
        "load": {LLAMA_70B: 0, LLAMA_1B: 0},
    }
    counter_lock = asyncio.Lock()

    # Track quality and latency
    metrics = []
    metrics_lock = asyncio.Lock()

    async def process_request(
        request,
        llama_70b_queue,
        llama_1b_queue,
        threshold,
        learning_rate,
        cost_target,
        cost_weight,
        load_target,
        load_weight,
    ):
        """Routes a request to the best model, implementing Algorithm 1 in the paper"""
        async with counter_lock:
            # Cost
            llama_70b_cost = counters["cost"][LLAMA_70B]
            llama_1b_cost = counters["cost"][LLAMA_1B]

            # Load
            llama_70b_load = counters["load"][LLAMA_70B]
            llama_1b_load = counters["load"][LLAMA_1B]

        # Compute error
        cost_error = llama_70b_cost / (llama_70b_cost + llama_1b_cost + 1e-6) - cost_target
        if llama_70b_load <= load_target:
            load_error = 0
        else:
            load_error = (llama_70b_load - load_target) / (load_target + 1e-6)
        error = cost_weight * cost_error + load_weight * load_error

        # Adjust threshold
        threshold += learning_rate * error
        threshold = max(0, min(threshold, 1))

        status = {
            "cost": {
                LLAMA_70B: llama_70b_cost,
                LLAMA_1B: llama_1b_cost,
                "relative_cost": round(llama_70b_cost / (llama_70b_cost + llama_1b_cost + 1e-6), 3),
                "cost_target": cost_target,
                "cost_error": round(cost_error, 3),
                "cost_weight": cost_weight,
            },
            "load": {
                LLAMA_70B: llama_70b_load,
                LLAMA_1B: llama_1b_load,
                "relative_load": round(llama_70b_load / (llama_70b_load + llama_1b_load + 1e-6), 3),
                "load_target": load_target,
                "load_error": round(load_error, 3),
                "load_weight": load_weight,
            },
            "time": time.time(),
            "threshold": round(threshold, 3),
            "total_error": round(error, 3),
        }
        async with metrics_lock:
            metrics.append(status)
        logger.info(pformat(status))

        request["queue_time"] = time.time()
        
        # Predict response quality of each model
        predicted_quality = quality_predictor(request)
        
        # Route query
        query = request["prompt"]
        if predicted_quality > threshold:

            # Predict response length; this will update the error for the next request
            query_tokens, response_tokens = token_predictor(request, LLAMA_70B)
            request["predicted_response_tokens"] = response_tokens

            logger.info(f"Routed query to {LLAMA_70B}: {query} (p={predicted_quality}, adjusted_threshold={threshold})")
            async with counter_lock:
                counters["cost"][LLAMA_70B] += query_tokens + response_tokens
                counters["load"][LLAMA_70B] += query_tokens + response_tokens
            await llama_70b_queue.put(request)

        else:
            # Predict response length; this will update the error for the next request
            query_tokens, response_tokens = token_predictor(request, LLAMA_1B)
            request["predicted_response_tokens"] = response_tokens

            logger.info(f"Routed query to {LLAMA_1B}: {query} (p={predicted_quality}, adjusted_threshold={threshold})")
            async with counter_lock:
                counters["cost"][LLAMA_1B] += query_tokens + response_tokens
                counters["load"][LLAMA_1B] += query_tokens + response_tokens
            await llama_1b_queue.put(request)

        return threshold

    async def process_queue(model, queue, result_queue, model_name):
        """Dequeues a request, sends it to the model and extracts its response."""
        while True:
            request = await queue.get()
            if request is None:
                break

            start_time = time.time()
            model_input = request[model_name]["modelInput"]
            prompt = model_input["prompt"]
            logger.info(f"{model_name} started processing query '{prompt}' at {datetime.now()}")
            
            # Send request to model
            await model(request)

            # Extract model output
            model_output = request[model_name]["modelOutput"]
            processing_time = time.time() - start_time
            request["processing_time"] = processing_time
            request["end_time"] = time.time()            
            await result_queue.put((request, model_output))
            logger.info(
                f"{model_name} finished processing query '{prompt}' at "
                f"{datetime.now()}, took {processing_time:.2f}s"
            )

            # Update counters
            query_tokens = model_output["prompt_token_count"]
            response_tokens = model_output["generation_token_count"]
            async with counter_lock:
                counters["load"][model_name] -= query_tokens + response_tokens

            queue.task_done()

    # Set up Asyncio to work in the background
    strong_model_inputs, strong_model_outputs = Queue(), Queue()
    weak_model_inputs, weak_model_outputs = Queue(), Queue()
    strong_model_task = asyncio.create_task(
        process_queue(
            llama_70b_model,
            strong_model_inputs,
            strong_model_outputs,
            LLAMA_70B,
        )
    )
    weak_model_task = asyncio.create_task(
        process_queue(
            llama_1b_model,
            weak_model_inputs,
            weak_model_outputs,
            LLAMA_1B,
        )
    )

    # Simulate request arrival as a Poisson process and route and process incoming requests
    threshold = 0.5
    for request in dataset:
        threshold = await process_request(
            request,
            strong_model_inputs,
            weak_model_inputs,
            threshold,
            learning_rate,
            target_cost,
            cost_weight,
            target_load,
            load_weight,
        )
        delay = np.random.exponential(1 / arrival_rate)
        await asyncio.sleep(delay)

    # Signal completion
    await strong_model_inputs.put(None)
    await weak_model_inputs.put(None)

    # Wait for both models to finish
    await asyncio.gather(strong_model_task, weak_model_task)

    # Signal completion for result retrieval
    await strong_model_outputs.put((None, None))
    await weak_model_outputs.put((None, None))

    # Compute statistics
    async def get_results(queue, queue_name):
        """Obtains and processes the results from queue"""
        items = []
        while not queue.empty():
            queue_item = await queue.get()
            if queue_item is None:
                continue
            request, result = queue_item
            items.append(convert_to_json({"queue_name": queue_name, "request": request, "result": result}))

        return items
    strong_model_results = await get_results(strong_model_outputs, LLAMA_70B)
    weak_model_results = await get_results(weak_model_outputs, LLAMA_1B)
    statistics = compute_statistics(strong_model_results, weak_model_results, config)
    logger.info(pformat(statistics))

    # Save results
    runs = OUTPUTS / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    with open(runs / f"statistics-{run_identifier}.json", "w") as f:
        json.dump(statistics, f, indent=4)

    logger.info(f"Ended run {run_identifier} at {datetime.now().isoformat()}")
    wandb.finish()


if __name__ == "__main__":
    run_idx = 0

    for target_cost in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        for target_load in [0, 2_000, 4_000, 6_000, 8_000, 10_000, 1_000, 3_000, 5_000, 7_000, 9_000]:
            for quality_predictor in ["perfect", "random"]:
                asyncio.run(
                    run_experiment(
                        dataset_path=DATASET,
                        subset=100,
                        arrival_rate=1,
                        target_cost=target_cost,
                        cost_weight=1,
                        target_load=target_load,
                        load_weight=1,
                        quality_predictor=quality_predictor,
                        token_predictor="perfect",
                        learning_rate=0.1,
                        run_identifier=f"{JOB_ID}-{run_idx}",
                    )
                )
                run_idx += 1
