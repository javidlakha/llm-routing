"""
Uses GPT-4o Mini to evaluate the quality of Llama 3.2 Instruct (1B) and Llama 3.1 Instruct (70B) responses.
"""

import json
import os
import random
from pathlib import Path
from typing import Any

import openai
import tenacity
import asyncio
from tqdm.asyncio import tqdm as tqdm_asyncio

JSONType = dict[str, Any] | list[Any] | str | int | float | bool | None

JOB_ID = os.getenv("SLURM_JOB_ID", random.randint(0, 2**31 - 1))
NUM_CPU = os.cpu_count()
INPUT_SEED = 4013284336
MAX_COMPLETION_TOKENS = 1
MAX_INPUT_TOKENS = 512
MODEL = "gpt-4o-mini-2024-07-18"
TRAINING_SET = "datasets/training_set.jsonl"
TEST_SET = "datasets/test_set.jsonl"
OUTPUT_DIRECTORY = Path(f"outputs/quality-labels-{JOB_ID}")
TEMPERATURE = 0.7
TOP_P = 1
RATE_LIMIT = 3_000
DELAY = 60 / RATE_LIMIT
RETRY_ATTEMPTS = 5

PROMPT = """Please evaluate the quality of the response provided by a large language model to an input query. Rate the response on a scale from 1 to 5 based on the criteria below.

# Input Query:
{query}

# Model Response:
{response}

# Rating Scale and Criteria

- **5 (Excellent)**
  - The response is exceptionally clear, concise, and complete, fully addressing the query in a way that is both accurate and easy to understand.
  - It effectively explains complex ideas in a simple, relatable manner.
  - There are no significant factual errors, and the explanation is engaging, encouraging further curiosity about the topic.

- **4 (Good)**
  - The response is clear and mostly complete, with only minor omissions or areas that could benefit from further simplification.
  - It is mostly accurate and appropriately simplified for a general audience, though the explanation could be slightly more relatable or accessible.
  - The response maintains coherence and engagement but may miss a small opportunity to fully capture the user's interest.

- **3 (Adequate)**
  - The response is moderately clear and covers the main points, though there may be areas where it could be simplified or rephrased for better understanding.
  - It is mostly accurate but may contain minor inaccuracies or lack sufficient detail, limiting the effectiveness of the explanation.
  - The response generally makes sense, though it might feel slightly off-tone, lacking full relatability or coherence.

- **2 (Poor)**
  - The response is difficult to understand, incomplete, or somewhat inaccurate, causing potential confusion.
  - It may overuse technical language or fail to break down concepts sufficiently for the intended audience.
  - The response lacks coherence, and its structure or tone may feel disengaging or irrelevant.

- **1 (Very Poor)**
  - The response is severely lacking in clarity, completeness, and accuracy, failing to address the query in a meaningful way.
  - It may contain significant factual errors or be entirely off-topic, confusing, or unhelpful.
  - The response is disengaging, incoherent, or overly technical, making it wholly unsuitable.

Please apply these criteria carefully and respond with a single integer from the set {{1, 2, 3, 4, 5}} based on your assessment.

It is important that you respond only with a single integer, otherwise your response will be discarded. Here are three example responses (one per line):
2
5
1
"""

openai_client = openai.OpenAI()
semaphore = asyncio.Semaphore(RATE_LIMIT)


@tenacity.retry(
    wait=tenacity.wait_exponential(),
    stop=tenacity.stop_after_attempt(RETRY_ATTEMPTS),
    retry=tenacity.retry_if_exception_type(openai.OpenAIError),
)
async def call_openai_api(
    client: openai.OpenAI,
    prompt: str,
) -> tuple[str, dict[str, int]]:
    """
    Calls the OpenAI API. Retries if the request fails or does not meet our
    standard, with exponential backoff.
    """
    async with semaphore:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[{"role": "system", "content": prompt}],
            model=MODEL,
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            top_p=TOP_P,
        )
        content = response.choices[0].message.content.strip()
        return content


@tenacity.retry(
    stop=tenacity.stop_after_attempt(RETRY_ATTEMPTS),
    retry=tenacity.retry_if_exception_type(ValueError),
)
async def _evaluate_example(example: JSONType) -> JSONType:
    """
    Evaluates an LLM response. Requires the evaluation to be an integer in the
    set {1, 2, 3, 4, 5}.
    """
    models = {k for k in example.keys() if k != "prompt"}
    for model in models:
        query = example["prompt"]
        response = example[model]["modelOutput"]["generation"]
        prompt = PROMPT.format(query=query, response=response)
        completion = await call_openai_api(
            openai_client,
            prompt,
        )
        if completion.isdigit() and int(completion) in {1, 2, 3, 4, 5}:
            example[model]["score"] = int(completion)
        else:
            raise ValueError(f"'{completion}' is not a valid score")

    return example


async def evaluate_example(example: JSONType) -> JSONType:
    try:
        example = await _evaluate_example(example)
    except tenacity.RetryError:
        return None

    return example


async def main() -> None:
    # Load and evaluates the quality of responses in the training set
    dataset = []
    with open(TRAINING_SET, "r") as f:
        for l in f:
            j = json.loads(l)
            dataset.append(j)

    tasks = []
    for example in dataset:
        task = evaluate_example(example)
        tasks.append(task)
    results = await tqdm_asyncio.gather(*tasks, total=len(tasks))
    results = [r for r in results if r]

    with open("server/datasets/training_set_4omini.jsonl", "w") as f:
        for datum in dataset:
            f.write(json.dumps(datum) + "\n")

    # Load and evaluates the quality of responses in the test set
    dataset = []
    with open(TEST_SET, "r") as f:
        for l in f:
            j = json.loads(l)
            dataset.append(j)

    # Evaluates the quality of responses in the test set
    tasks = []
    for example in dataset:
        task = evaluate_example(example)
        tasks.append(task)
    results = await tqdm_asyncio.gather(*tasks, total=len(tasks))
    results = [r for r in results if r]

    with open("server/datasets/test_set_4omini.jsonl", "w") as f:
        for datum in dataset:
            f.write(json.dumps(datum) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
