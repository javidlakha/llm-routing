from asyncio import Queue

import numpy as np

from utils import JSONType


LLAMA_1B = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_70B = "meta-llama/Llama-3.1-70B-Instruct"


def compute_statistics(strong_model_results: Queue, weak_model_results: Queue, config: JSONType) -> JSONType:
    data = {
        LLAMA_70B: {"latency": [], "num_tokens": [], "quality": [], "processing_time": []},
        LLAMA_1B: {"latency": [], "num_tokens": [], "quality": [], "processing_time": []},
    }
    for record in strong_model_results:
        if not record["request"]:
            continue
        data[LLAMA_70B]["processing_time"].append(record["request"]["processing_time"])
        data[LLAMA_70B]["latency"].append(record["request"]["end_time"] - record["request"]["queue_time"])
        data[LLAMA_70B]["quality"].append(int(record["request"][LLAMA_70B]["score"]))
        total_tokens = int(record["request"][LLAMA_70B]["modelOutput"]["prompt_token_count"]) + int(record["request"][LLAMA_70B]["modelOutput"]["generation_token_count"])
        data[LLAMA_70B]["num_tokens"].append(total_tokens)

    for record in weak_model_results:
        if not record["request"]:
            continue
        data[LLAMA_1B]["processing_time"].append(record["request"]["processing_time"])
        data[LLAMA_1B]["latency"].append(record["request"]["end_time"] - record["request"]["queue_time"])
        data[LLAMA_1B]["quality"].append(int(record["request"][LLAMA_1B]["score"]))
        total_tokens = int(record["request"][LLAMA_1B]["modelOutput"]["prompt_token_count"]) + int(record["request"][LLAMA_1B]["modelOutput"]["generation_token_count"])
        data[LLAMA_1B]["num_tokens"].append(total_tokens)

    data_total_num_tokens = data[LLAMA_70B]["num_tokens"] + data[LLAMA_1B]["num_tokens"]
    data_total_latency = data[LLAMA_70B]["latency"] + data[LLAMA_1B]["latency"]
    data_total_quality = data[LLAMA_70B]["quality"] + data[LLAMA_1B]["quality"]

    statistics = {
        LLAMA_70B: {
            "latency": {
                "mean": np.mean(data[LLAMA_70B]["latency"]),
                "std": np.std(data[LLAMA_70B]["latency"]),
            },
            "num_tokens": {
                "mean": np.mean(data[LLAMA_70B]["num_tokens"]),
                "share": round(
                    sum(data[LLAMA_70B]["num_tokens"])
                    / sum(data_total_num_tokens),
                    3,
                ),
                "std": np.std(data[LLAMA_70B]["num_tokens"]),
                "total": sum(data[LLAMA_70B]["num_tokens"]),
            },
            "quality": {
                "mean": np.mean(data[LLAMA_70B]["quality"]),
                "std": np.std(data[LLAMA_70B]["quality"]),
            },
            "requests": len(data[LLAMA_70B]["num_tokens"]),
        },
        LLAMA_1B: {
            "latency": {
                "mean": np.mean(data[LLAMA_1B]["latency"]),
                "std": np.std(data[LLAMA_1B]["latency"]),
            },
            "num_tokens": {
                "mean": np.mean(data[LLAMA_1B]["num_tokens"]),
                "share": round(
                    sum(data[LLAMA_1B]["num_tokens"])
                    / sum(data_total_num_tokens),
                    3,
                ),
                "std": np.std(data[LLAMA_1B]["num_tokens"]),
                "total": sum(data[LLAMA_1B]["num_tokens"]),
            },
            "quality": {
                "mean": np.mean(data[LLAMA_1B]["quality"]),
                "std": np.std(data[LLAMA_1B]["quality"]),
            },
            "requests": len(data[LLAMA_1B]["num_tokens"]),
        },
        "total": {
            "latency": {
                "mean": np.mean(data_total_latency),
                "std": np.std(data_total_latency),
            },
            "num_tokens": {
                "mean": np.mean(data_total_num_tokens),
                "std": np.std(data_total_num_tokens),
                "total": sum(data_total_num_tokens),
            },
            "quality": {
                "mean": np.mean(data_total_quality),
                "std": np.std(data_total_quality),
            },
            "requests": len(data_total_num_tokens),
        },
        "config": config,
    }

    return statistics
