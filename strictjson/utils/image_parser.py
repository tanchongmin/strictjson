import os
import re
import base64
import mimetypes
from functools import wraps
from typing import Any, Callable, Awaitable, Dict, List

def _parse_prompt_to_contents(user_prompt: str) -> List[Dict[str, Any]]:
    """
    Turn a user prompt with <<markers>> into a list of text/image parts.
    Shared by sync and async decorators to keep behavior identical.
    """
    if '<<' in user_prompt and '>>' in user_prompt:
        parts = re.split(r'<<(.*?)>>', user_prompt)  # odd-indexed = markers
        new_contents: List[Dict[str, Any]] = []
        for idx, part in enumerate(parts):
            if idx % 2 == 1:
                marker = part.strip()
                if marker.startswith(("http://", "https://")):
                    new_contents.append({"type": "image_url", "image_url": {"url": marker}})
                elif os.path.isfile(marker):
                    mime_type, _ = mimetypes.guess_type(marker)
                    if mime_type is None:
                        mime_type = "application/octet-stream"
                    with open(marker, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode("utf-8")
                    data_url = f"data:{mime_type};base64,{encoded}"
                    new_contents.append({"type": "image_url", "image_url": {"url": data_url}})
                else:
                    new_contents.append({"type": "text", "text": f"<<{marker}>>"})
            else:
                if part:
                    new_contents.append({"type": "text", "text": part})
        return new_contents
    else:
        return [{"type": "text", "text": user_prompt}]

def image_parser(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Sync decorator version (unchanged behavior vs your original, but DRY via helper).
    """
    @wraps(func)
    def wrapper(system_prompt: str, user_prompt: str):
        contents = _parse_prompt_to_contents(user_prompt)
        return func(system_prompt, contents)
    return wrapper

def image_parser_async(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """
    Async decorator that scans the user prompt for markers in the form <<marker>>.
    Behavior matches the sync version, but awaits the wrapped coroutine function.
    """
    @wraps(func)
    async def wrapper(system_prompt: str, user_prompt: str):
        contents = _parse_prompt_to_contents(user_prompt)
        return await func(system_prompt, contents)
    return wrapper
