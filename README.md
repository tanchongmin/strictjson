# StrictJSON v6.2.0 — `parse_yaml`

## Structured Output Parser for Large Language Models (LLMs)

`parse_yaml` helps you get clean, structured output from LLMs like ChatGPT. It tells the LLM the required output format and data types, to ensure strict conformity to the output format.

> Old versions (e.g. `strict_json`) that used JSON still work, but `parse_yaml` is the new and improved method.

---

## What Makes It Useful

* YAML structure avoids the need for backslash escaping quotation marks or brackets, simplifying output and parsing compared to JSON.
* Shorter output context size with YAML than JSON.
* Creates / uses a Pydantic model for type checking, ensuring robustness of output.
* Fixes mistakes automatically (tries again up to three times by default — configurable via `num_tries`).
* Works with any LLM model (ChatGPT, Claude, Gemini, etc.) with adequate YAML understanding capabilities.

---

## How to Install

```bash
pip install strictjson
```

---

## Tutorial

[Basics](Tutorial%20-%20parse_yaml.ipynb): Teaches how to create LLM functions with `system_prompt`, `user_prompt` as input parameters, and output a string as output. Also covers both sync and async modes of `parse_yaml`

[Multimodal Input](Tutorial%20-%20Multimodal%20Inputs.ipynb): Teaches how to use the decorator `image_parser` and `image_parser_async` to get LLM to view images

---

## Quick Example

Here’s a simple program that asks the AI for a blog post idea:

```python
from strictjson import parse_yaml

def llm(system_prompt: str, user_prompt: str, **kwargs) -> str:
    from openai import OpenAI
    client = OpenAI()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=messages
    )
    return resp.choices[0].message.content

result = parse_yaml(
  system_prompt="You are a friendly assistant.",
  user_prompt="Write a blog post idea about AI in education",
  output_format={
    "title": "str",
    "tags": "List[str]",
    "published": "bool"
  },
  llm=llm
)

print(result)
# {'title': 'Adaptive Tutors', 'tags': ['ai', 'edtech', 'personalization'], 'published': False}
```

---

## Data Types You Can Use

You can use simple data types like these:

| Type             | Example Value (in Python)                      | Meaning / Notes                                            |
| ---------------- | ---------------------------------------------- | ---------------------------------------------------------- |
| `str`            | `"Hello"`                                      | A string of text.                                          |
| `int`            | `5`                                            | A whole number (integer).                                  |
| `float`          | `3.14`                                         | A floating-point number (decimal).                         |
| `bool`           | `True` / `False`                               | A boolean value — yes/no, true/false.                      |
| `List[int]`      | `[1, 2, 3]`                                    | A list (array-like) of integers. From `typing`.            |
| `Dict[str, Any]` | `{"a": 1, "b": 2}`                             | A dictionary (key–value pairs). From `typing`.             |
| `date`           | `date(2025, 5, 9)`                             | A calendar date (`from datetime import date`).             |
| `datetime`       | `datetime(2025, 5, 9, 14, 30)`                 | A date and time (`from datetime import datetime`).         |
| `UUID`           | `UUID("550e8400-e29b-41d4-a716-446655440000")` | A universally unique identifier (`from uuid import UUID`). |
| `Decimal`        | `Decimal("12.50")`                             | An exact decimal number (`from decimal import Decimal`).   |
| `None`           | `None`                                         | The null or "no value" object in Python.                   |

A datatype `Any` refers to any datatype.

All of the datatypes in `parse_yaml` are similar to Python type hints.

If you write `list` or `dict` without brackets, we interpret them as `List[Any]` and `Dict[str, Any]`.

If you write `List[{'key1': 'str', 'key2': 'int'}]`, we will ensure the output to be a list of dictionaries with the desired key names and output types for each key. This is useful if you need a list of a particular schema.

You can constrain a field to a fixed set of values using Enum:

> `Enum['A','B','C']` — limits output to one of these values. (Note: Use `Enum` instead of the `Literal` type hint in Python)

You can use `Optional` for fields that can return null values, e.g. `Optional[str]` returns either a string or a null value.

You can use `Union` for fields that can return more than one datatype, e.g. `Union[str, int]` returns either a string or integer.

We also support [PEP 604](https://peps.python.org/pep-0604/) syntax using `|`:

* `str | int` → `Union[str, int]`
* `str | None` → `Optional[str]`

`List[...]` and `Dict[...]` are **type hints** — the actual runtime values are `list` and `dict`. Likewise, `date`, `datetime`, `UUID`, and `Decimal` appear as proper Python objects in the result.

---

## More Examples

### 1. Creative ideas

```python
parse_yaml(
  system_prompt="Be imaginative but safe",
  user_prompt="Give me 5 names on a weather theme",
  output_format={
    "Names": "5 cool weather names, List[str]",
    "Meanings": "name -> meaning, Dict[str, str]"
  },
  llm=llm,
)
```

**Example Output:**

```python
{
  "Names": ["Storm Whisper", "Cloud Dancer", "Thunder Heart", "Rain Song", "Sun Chaser"],
  "Meanings": {
    "Storm Whisper": "Someone calm even in chaos",
    "Cloud Dancer": "Light and free-spirited",
    "Thunder Heart": "Strong and passionate",
    "Rain Song": "Gentle and emotional",
    "Sun Chaser": "Always positive"
  }
}
```

---

### 2. Finding facts

```python
parse_yaml(
  system_prompt="Extract information",
  user_prompt=article_about_nasa,
  output_format={
    "Entities": "organisations only, List[str]",
    "Sentiment": "Enum['Happy','Sad','Neutral']",
    "Summary": "str"
  },
  llm=llm,
)
```

**Example Output:**

```python
{
  "Entities": ["NASA", "SpaceX", "Boeing"],
  "Sentiment": "Happy",
  "Summary": "NASA announced a new partnership with SpaceX to develop next-generation spacecraft. The article is optimistic about future space exploration."
}
```

---

### 3. Nested information

```python
parse_yaml(
  system_prompt="Plan a party",
  user_prompt="Generate a kids party for Alex",
  output_format={
    "name": "str",
    "date": "date",
    "participants": "List[Dict[str, Any]]"
  },
  llm=llm,
)
```

**Example Output:**

```python
{
  "name": "Alex's Birthday Party",
  "date": date(2026, 3, 12),
  "participants": [
    {"name": "Ava", "age": 8, "gift": "Lego set"},
    {"name": "Ben", "age": 9, "gift": "Book"},
    {"name": "Liam", "age": 8, "gift": "Art kit"}
  ]
}
```

---

## Using Your Own Models

You can also define your own data rules using Pydantic models. This is for developers who want strong type checking.

```python
from typing import List
from pydantic import BaseModel, Field
from datetime import date as Date

class Participant(BaseModel):
    Name: str = Field(..., pattern=r"^A.*$")
    Age: int = Field(..., ge=5, le=12)

class CalendarEvent(BaseModel):
    name: str
    date: Date
    participants: List[Participant]

result = parse_yaml(
    system_prompt="You are a helpful assistant",
    user_prompt="Generate a birthday event for Alex",
    pydantic_model=CalendarEvent,
    llm=llm
)
```

---

## Async Example (For Advanced Users)

If you are writing an async app, use this version:

```python
from openai import AsyncOpenAI

async def llm_async(system_prompt: str, user_prompt: str, **kwargs):
    client = AsyncOpenAI()
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return resp.choices[0].message.content

result = await parse_yaml_async(
  system_prompt="Summarise this research abstract",
  user_prompt=abstract,
  output_format={"Summary": "<= 3 sentences, str"},
  llm=llm_async,
)
# {'Summary': 'The research explores how artificial intelligence can personalize learning for students. It highlights improved engagement and adaptive feedback as key benefits. The study concludes that AI can complement teachers by automating routine assessments.'}
```

---

## Multimodal Inputs (For Advanced Users)

### Image Reading (Vision)

`parse_yaml` can process images using the decorator `@image_parser`.

`parse_yaml_async` can process images using `@image_parser_async`.

They convert any `<<image_url_or_path>>` in your prompt into structured multimodal input for vision-capable models (like Gemini, Claude, or GPT-4o).
You need to use the OpenAI-compatible interface to use this parser.

#### Example (Sync)

```python
from strictjson import parse_yaml
from strictjson.utils import image_parser
import os

@image_parser
def llm(system_prompt, user_prompt, **kwargs):
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"])
    messages = []
    if system_prompt:
        messages.append({"role": "system",
                         "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": user_prompt})
    resp = client.chat.completions.create(
        model="google/gemini-2.5-flash", messages=messages)
    return resp.choices[0].message.content

img1 = "https://.../map1.png"
img2 = "https://.../map2.png"

res = parse_yaml(
  system_prompt="Describe the game states",
  user_prompt=f"State1: <<{img1}>>, State2: <<{img2}>>",
  output_format={
    "State 1": "str",
    "State 2": "str",
    "Thoughts": "str"
  },
  llm=llm
)
```

---

#### Example (Async)

```python
from strictjson import parse_yaml_async
from strictjson.utils import image_parser_async
import os

@image_parser_async
async def llm_async(system_prompt, user_prompt, **kwargs):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                         api_key=os.environ["OPENROUTER_API_KEY"])
    messages = []
    if system_prompt:
        messages.append({"role": "system",
                         "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": user_prompt})
    resp = await client.chat.completions.create(
        model="google/gemini-2.5-flash", messages=messages)
    return resp.choices[0].message.content

filepath = "ai_electricity.png"
res = await parse_yaml_async(
  system_prompt="Extract text and describe the image",
  user_prompt=f"Image: <<{filepath}>>",
  output_format={
    "Image Text": "str",
    "Image Description": "str"
  },
  llm=llm_async
)
```

---

## Project Info

* Version: 6.2.0
* Updated: 7 Nov 2025
* Created: 7 Apr 2023
* Tested with: Pydantic 2.12.4
* Community: [John's AI Group on Discord](https://discord.gg/bzp87AHJy5)
