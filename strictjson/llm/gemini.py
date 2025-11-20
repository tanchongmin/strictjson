from google.genai import types
from google import genai
from google.api_core.exceptions import GoogleAPICallError

import logging
from typing import List, Union

from strictjson import convert_schema_to_pydantic
from pydantic import ValidationError

from strictjson.utils import gemini_image_parser_sync, gemini_image_parser_async

@gemini_image_parser_sync
def gemini_sync(
    system_prompt: str,
    user_prompt: Union[str, List],
    **kwargs,
):
    """
    Synchronous version of the Gemini API helper with native structured output.

    - If `output_format` (JSON schema) is provided, it is converted to a
      Pydantic model and used as the response_json_schema.
    - If `pydantic_model` is provided directly, it is used as the schema.
    - Otherwise, returns the raw `response.text`.
    """

    # --- pull structured-output config out of kwargs ---
    output_format = kwargs.pop("output_format", None)
    pydantic_model: Optional[Type[BaseModel]] = kwargs.pop("pydantic_model", None)

    # --- pull out model type from kwargs --- 
    model = kwargs.pop("model", "gemini-2.5-flash")

    # add in optional pydantic model generation
    if output_format is not None:
        pydantic_model = convert_schema_to_pydantic(output_format)

        additional_config = {
            "response_mime_type": "application/json",
            "response_json_schema": pydantic_model.model_json_schema(),
        }

    elif pydantic_model is not None:
        additional_config = {
            "response_mime_type": "application/json",
            "response_json_schema": pydantic_model.model_json_schema(),
        }

    else:
        pydantic_model = None
        additional_config = {}

    contents = user_prompt if isinstance(user_prompt, list) else [user_prompt]

    try:
        # place your GEMINI_API_KEY inside .env
        client = genai.Client()

        # Synchronous call of LLM
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                **kwargs,
                **additional_config,
            ),
        )

        try:
            # If user passed in a concrete Pydantic model
            if pydantic_model is not None:
                output = pydantic_model.model_validate_json(
                    response.text
                ).model_dump()
                return output

            # Otherwise just return the raw text
            else:
                return response.text

        except (ValidationError, ValueError, TypeError) as e:
            raise ValueError(
                "Failed to parse Gemini structured output.\n"
                f"Response string: {response.text}\nError message: {e}"
            ) from e

    except (GoogleAPICallError, RuntimeError) as e:
        logging.warning(f"Gemini request failed: {e}")
        raise

@gemini_image_parser_async
async def gemini_async(
    system_prompt: str,
    user_prompt: Union[str, List],
    **kwargs,
):
    """
    Asynchronous version of the Gemini API helper with native structured output.

    - If `output_format` (JSON schema) is provided, it is converted to a
      Pydantic model and used as the response_json_schema.
    - If `pydantic_model` is provided directly, it is used as the schema.
    - Otherwise, returns the raw `response.text`.
    """

    # --- pull structured-output config out of kwargs ---
    output_format = kwargs.pop("output_format", None)
    pydantic_model: Optional[Type[BaseModel]] = kwargs.pop("pydantic_model", None)

    # --- pull out model type from kwargs --- 
    model = kwargs.pop("model", "gemini-2.5-flash")

    # add in optional pydantic model generation
    if output_format is not None:
        pydantic_model = convert_schema_to_pydantic(output_format)

        additional_config = {
            "response_mime_type": "application/json",
            "response_json_schema": pydantic_model.model_json_schema(),
        }

    elif pydantic_model is not None:
        additional_config = {
            "response_mime_type": "application/json",
            "response_json_schema": pydantic_model.model_json_schema(),
        }

    else:
        additional_config = {}

    contents = user_prompt if isinstance(user_prompt, list) else [user_prompt]

    try:
        # place your GEMINI_API_KEY inside .env
        client = genai.Client()

        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                **kwargs,
                **additional_config,
            ),
        )

        try:
            if pydantic_model is not None:
                output = pydantic_model.model_validate_json(
                    response.text
                ).model_dump()
                return output
                
            else:
                return response.text

        except (ValidationError, ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to parse Gemini structured output.\n"
                f"Response string: {response.text}\nError message: {e}"
            ) from e

    except (GoogleAPICallError, RuntimeError) as e:
        logging.warning(f"Gemini request failed: {e}")
        raise
