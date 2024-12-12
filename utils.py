import json
from typing import Any, Iterator

import numpy as np
import torch

JSONType = dict[str, Any] | list[Any] | str | int | float | bool | None


def serialize_json(j: JSONType) -> str:
    """
    Serializes a JSON object into a consistent string format with sorted keys
    and compact separators.
    """
    return json.dumps(j, sort_keys=True, separators=(",", ":"))


def load_jsonl(file_path: str) -> Iterator[dict[str, Any]]:
    """Loads a .jsonl file into an iterator of JSON objects."""
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def save_jsonl(data: list[JSONType], file_path: str) -> None:
    """Saves a list of JSON objects into a .jsonl file."""
    with open(file_path, "w", encoding="utf-8") as file:
        for json_obj in data:
            json_str = serialize_json(json_obj)
            file.write(json_str + "\n")


def convert_to_json(o: Any) -> JSONType:
    """Recursively serializes an object, converting it into a JSON-compatible format."""
    # Primitives
    if isinstance(o, (str, int, float, bool, type(None))):
        return o

    # NumPy arrays
    elif isinstance(o, np.ndarray):
        return o.tolist()

    # PyTorch tensors
    elif torch.is_tensor(o):
        return o.detach().cpu().numpy().tolist()

    # Lists
    elif isinstance(o, list):
        return [convert_to_json(item) for item in o]

    # Dictionaries
    elif isinstance(o, dict):
        return {key: convert_to_json(value) for key, value in o.items()}

    # Objects with a __dict__ attribute
    elif hasattr(o, "__dict__"):
        return {key: convert_to_json(value) for key, value in o.__dict__.items()}

    # Named tuples
    elif hasattr(o, "_asdict"):
        return convert_to_json(o._asdict())

    # Iterables
    elif hasattr(o, "__iter__") and not isinstance(o, str):
        return [convert_to_json(item) for item in o]

    # Use the string representation for unsupported types
    else:
        return str(o)
