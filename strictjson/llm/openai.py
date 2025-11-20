import logging
from typing import List, Union, Optional, Type

import openai
from openai import OpenAI, AsyncOpenAI, APIError
from pydantic import BaseModel, ValidationError
from strictjson import convert_schema_to_openai_pydantic
from strictjson.utils import openai_image_parser_sync, openai_image_parser_async

@openai_image_parser_sync
def openai_sync(
    system_prompt: str,
    user_prompt: Union[str, List],
    **kwargs,
):
    """
    Synchronous helper using the OpenAI Responses API + structured outputs.

    Behaviour mirrors your `gemini(...)` helper:

    - If `output_format` (JSON schema) is provided, it is converted to a
      Pydantic model and used as `text_format`.
    - If `pydantic_model` is provided directly, it is used as `text_format`.
    - Otherwise, returns the raw `response.output_text` (string).
    """

    # put OPENAI_API_KEY and (optionally) OPENAI_BASE_URL in your .env
    client = OpenAI()

    # --- pull structured-output config out of kwargs ---
    output_format = kwargs.pop("output_format", None)
    pydantic_model: Optional[Type[BaseModel]] = kwargs.pop("pydantic_model", None)

    # --- pull out model type from kwargs --- 
    model = kwargs.pop("model", "gpt-5.1")

    if output_format is not None:
        # convert JSON schema -> Pydantic model
        pydantic_model = convert_schema_to_openai_pydantic(output_format)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # --- structured output path (Pydantic model) ---
        if pydantic_model is not None:
            response = client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=pydantic_model,
                **kwargs,
            )

            # `output_parsed` is an instance of `pydantic_model`
            return response.choices[0].message.parsed.model_dump()

        # --- plain-text path ---
        response = client.chat.completions.parse(
            model=model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

    except (ValidationError, ValueError, TypeError) as e:
        # ValidationError from Pydantic, Value/TypeError from bad kwargs, etc.
        raise ValueError(
            "Failed to parse OpenAI structured output.\n"
            f"Error message: {e}"
        ) from e

    except APIError as e:
        logging.warning(f"OpenAI request failed: {e}")
        raise

@openai_image_parser_async
async def openai_async(
    system_prompt: str,
    user_prompt: Union[str, List],
    **kwargs,
):
    """
    Async version of the helper, using AsyncOpenAI + structured outputs.

    Same semantics as `openai_sync`:
    - `output_format` (JSON schema) → converted to Pydantic.
    - `pydantic_model` → used directly.
    - Otherwise, returns the raw text (from the first choice message).
    """

    async_client = AsyncOpenAI()

    # --- pull structured-output config out of kwargs ---
    output_format = kwargs.pop("output_format", None)
    pydantic_model: Optional[Type[BaseModel]] = kwargs.pop("pydantic_model", None)

    # --- pull out model type from kwargs --- 
    model = kwargs.pop("model", "gpt-5.1")

    if output_format is not None:
        # convert JSON schema -> Pydantic model
        pydantic_model = convert_schema_to_openai_pydantic(output_format)

    # mirror the sync helper's message format
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # --- structured output path (Pydantic model) ---
        if pydantic_model is not None:
            response = await async_client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=pydantic_model,
                **kwargs,
            )

            # `parsed` is an instance of `pydantic_model`
            return response.choices[0].message.parsed.model_dump()

        # --- plain-text path ---
        response = await async_client.chat.completions.parse(
            model=model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

    except (ValidationError, ValueError, TypeError) as e:
        # ValidationError from Pydantic, Value/TypeError from bad kwargs, etc.
        raise ValueError(
            "Failed to parse OpenAI structured output.\n"
            f"Error message: {e}"
        ) from e

    except APIError as e:
        logging.warning(f"OpenAI request failed: {e}")
        raise
