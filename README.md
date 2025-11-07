# StrictJSON v6.2.0 — `parse_yaml`

## Make AI give answers in a neat, tidy format

`parse_yaml` helps you get clean, structured answers from Large Language Models (LLMs) like ChatGPT. It tells the LLM exactly what shape the answer should be, so you don’t have to fix it later.

It uses YAML, a simple way of writing lists and information, kind of like an easier version of JSON. It also checks the answer automatically to make sure it makes sense.

> Old versions (e.g. `strict_json`) that used JSON still work, but `parse_yaml` is the new and improved method.

---

## What Makes It Useful

* Easy to read — YAML looks like tidy notes, not messy code.
* Checks answers — makes sure the LLM gives the right types (like numbers, dates, or words).
* Understands many data types.
* Fixes mistakes automatically (tries again up to three times by default - changeable via `num_tries` parameter).
* Works with any LLM model (ChatGPT, Claude, Gemini, etc.).

---

## How to Install

```
pip install strictjson
```

---

## Quick Example

Here is a small program that asks the AI for a blog post idea:

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
  system_prompt="You are concise",
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

| Type             | Example                                | Meaning                          |
| ---------------- | -------------------------------------- | -------------------------------- |
| `str`            | "Hello"                                | text or words                    |
| `int`            | 5                                      | whole number                     |
| `float`          | 3.14                                   | number with decimals             |
| `bool`           | True / False                           | yes or no                        |
| `List[int]`      | [1, 2, 3]                              | a list of numbers                |
| `Dict[str, Any]` | {"a": 1, "b": 2}                       | a dictionary (key-value pairs)   |
| `date`           | 2025-05-09                             | a calendar date                  |
| `datetime`       | 2025-05-09T14:30                       | date and time                    |
| `UUID`           | "550e8400-e29b-41d4-a716-446655440000" | a unique ID                      |
| `Decimal`        | 12.50                                  | an exact number (good for money) |

A datatype `Any` refers to any datatype.

A datatype `list` or `dict` used without [], will be defaulted to `List[Any]` or `Dict[str, Any]`.

---

## What Happens Behind the Scenes

1. You tell `parse_yaml` what kind of answer you want.
2. It builds a simple template for the LLM to follow.
3. The LLM fills in that template in YAML format.
4. `parse_yaml` checks if the result is correct. If not, it fixes or retries it.
5. You get a clean Python dictionary at the end.

---

## More Examples

**1. Creative ideas**

```python
parse_yaml(
  system_prompt="Be imaginative but safe",
  user_prompt="Give me 5 names on a weather theme",
  output_format={
    "Names": "5 cool weather names, List[str]",
    "Meanings": "name → meaning, Dict[str, str]"
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

**2. Finding facts**

```python
parse_yaml(
  system_prompt="Extract information",
  user_prompt="Article about NASA",
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

**3. Nested information**

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
  "date": "2026-03-12",
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
#{"Summary": "The research explores how artificial intelligence can personalize learning for students. It highlights improved engagement and adaptive feedback as key benefits. The study concludes that AI can complement teachers by automating routine assessments."}
```

---

## Multimodal Inputs (For Advanced Users)

## Image Reading (Vision)

`parse_yaml` can process images with decorator `@image_parser` 
`parse_yaml_async` can process images with decorator `@image_parser_async`.
They convert any `<<image_url_or_path>>` in your prompt into structured multimodal input for vision-capable models (like Gemini, Claude, or GPT-4o). You need to use the OpenAI interface to use this parser.

### Example (Sync)

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

### Example (Async)

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
* Updated: 6 Nov 2025
* Created: 7 Apr 2023
* Tested with: Pydantic 2.12.4
* Community: [John's AI Group on Discord](https://discord.gg/bzp87AHJy5)