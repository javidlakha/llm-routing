import openai
import tenacity
import asyncio
import json
import os
from glob import glob
from pathlib import Path

from openai import OpenAI


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL = "gpt-4"
RATE_LIMIT = 3_000
RETRY_ATTEMPTS = 5

PROMPT = """Please evaluate the quality of the response provided by an AI model to an input query from the ELI5 (Explain Like I’m 5) dataset. Rate the response on a scale from 1 to 5 based on the criteria below. Your evaluation should be consistent across requests, allowing for objective comparisons. Focus on the following characteristics:

**Input Query:**  
{query}

**Model Response:**  
{response}

### Rating Scale and Criteria

- **5 (Excellent)**  
  - The response is exceptionally clear, concise, and complete, fully addressing the query in a way that is both accurate and easy to understand.
  - It effectively explains complex ideas in a simple, relatable manner that is ideal for the ELI5 format.
  - There are no significant factual errors, and the explanation is engaging, encouraging further curiosity about the topic.

- **4 (Good)**  
  - The response is clear and mostly complete, with only minor omissions or areas that could benefit from further simplification.
  - It is mostly accurate and appropriately simplified for a general audience, though the explanation could be slightly more relatable or accessible.
  - The response maintains coherence and engagement but may miss a small opportunity to fully capture the user’s interest.

- **3 (Adequate)**  
  - The response is moderately clear and covers the main points, though there may be areas where it could be simplified or rephrased for better understanding.
  - It is mostly accurate but may contain minor inaccuracies or lack sufficient detail, limiting the effectiveness of the explanation.
  - The response generally makes sense, though it might feel slightly off-tone for the ELI5 format, lacking full relatability or coherence.

- **2 (Poor)**  
  - The response is difficult to understand, incomplete, or somewhat inaccurate, causing potential confusion.
  - It may overuse technical language or fail to break down concepts sufficiently for the intended audience.
  - The response lacks coherence, and its structure or tone may feel disengaging or irrelevant to the ELI5 format.

- **1 (Very Poor)**  
  - The response is severely lacking in clarity, completeness, and accuracy, failing to address the query in a meaningful way.
  - It may contain significant factual errors or be entirely off-topic, confusing, or unhelpful.
  - The response is disengaging, incoherent, or overly technical, making it wholly unsuitable for the ELI5 format.

Please apply these criteria carefully and respond with a single number (1, 2, 3, 4, or 5) based on your assessment."""


semaphore = asyncio.Semaphore(RATE_LIMIT)


async def save(data, output_path):
    await asyncio.to_thread(Path(output_path).open("a").write, json.dumps(data) + "\n")


@tenacity.retry(
    wait=tenacity.wait_exponential(),
    stop=tenacity.stop_after_attempt(RETRY_ATTEMPTS),
    retry=tenacity.retry_if_exception_type(openai.OpenAIError),
)
async def evaluate_response(query: str, response: str) -> str:
    """
    Uses GPT-4 to evaluate the quality of a response to a query.
    """
    async with semaphore:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an evaluator of generated responses.",
                },
                {
                    "role": "user",
                    "content": PROMPT.format(query=query, response=response),
                },
            ],
        )
        return response.choices[0].message.content


async def process_jsonl(file_path: str, output_path: str):
    """
    Reads a JSONL file, evaluates the quality of each 'result' entry,
    and writes to a new JSONL file with evaluations.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(file_path, "r") as f:
        records = []
        for line in f:
            data = json.loads(line)
            records.append(data)

    tasks = []
    for record in records:
        query = record["query"]
        response = record["result"]
        evaluation = evaluate_response(query, response)
        tasks.append((record, evaluation))

    for record, evaluation in tasks:
        evaluation = await evaluation
        record["evaluation"] = evaluation
        await save(record, output_path)


if __name__ == "__main__":
    results = {}
    for input_path in glob("./outputs/*/results*.jsonl"):
        output_path = input_path.replace("outputs", "evaluations")
        if os.path.exists(output_path):
            continue
        asyncio.run(process_jsonl(input_path, output_path))
