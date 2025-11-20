import re
import asyncio
from typing import List, Union


def _split_prompt(user_prompt: str) -> List[str]:
    """Split on <<marker>> segments, keeping the markers' contents."""
    return re.split(r'<<(.*?)>>', user_prompt)


async def _fetch_url_image_async(session, url: str, types):
    """Fetch an image over HTTP(S) asynchronously and return a Gemini Part."""
    async with session.get(url) as resp:
        resp.raise_for_status()
        data = await resp.read()
        mime = resp.headers.get("Content-Type") or "image/jpeg"
        return types.Part.from_bytes(data=data, mime_type=mime)


async def _open_local_image_async(path: str):
    """Open a local image asynchronously using PIL.Image."""
    from PIL import Image  # local import to keep dependencies minimal
    return await asyncio.to_thread(Image.open, path)


def _fetch_url_image_sync(url: str, types):
    """Fetch an image over HTTP(S) synchronously and return a Gemini Part."""
    from urllib.request import urlopen
    from urllib.error import URLError

    try:
        with urlopen(url) as resp:
            data = resp.read()
            mime = resp.headers.get("Content-Type") or "image/jpeg"
    except URLError:
        # Propagate so caller can decide to keep the marker as text
        raise
    return types.Part.from_bytes(data=data, mime_type=mime)


def _open_local_image_sync(path: str):
    """Open a local image synchronously using PIL.Image."""
    from PIL import Image  # local import to keep dependencies minimal
    return Image.open(path)


def gemini_image_parser_async(func):
    """
    Async decorator that scans the user prompt for markers in the form <<filename>>.
    - HTTP(S) images are fetched with aiohttp and wrapped as types.Part.from_bytes
    - Local images are opened via PIL.Image using asyncio.to_thread
    The resulting list (mixing text and images) is passed to the wrapped async function.
    """

    async def wrapper(system_prompt: str, user_prompt: str, **kwargs):
        import aiohttp
        from google.genai import types

        # If already pre-processed, just pass through
        if not isinstance(user_prompt, str):
            return await func(system_prompt, user_prompt, **kwargs)

        parts = _split_prompt(user_prompt)
        new_contents: List[Union[str, object]] = []

        async with aiohttp.ClientSession() as session:
            for idx, part in enumerate(parts):
                if idx % 2 == 1:
                    image_source = part.strip()

                    # HTTP(S) source
                    if image_source.startswith("http://") or image_source.startswith("https://"):
                        try:
                            img_part = await _fetch_url_image_async(session, image_source, types)
                            new_contents.append(img_part)
                        except (aiohttp.ClientError, asyncio.TimeoutError):
                            # If fetching fails, leave the marker as text
                            new_contents.append(f"<<{part}>>")
                    else:
                        # Local file path
                        try:
                            img = await _open_local_image_async(image_source)
                            new_contents.append(img)
                        except (FileNotFoundError, OSError):
                            new_contents.append(f"<<{part}>>")
                else:
                    if part:
                        new_contents.append(part)

        return await func(system_prompt, new_contents, **kwargs)

    return wrapper


def gemini_image_parser_sync(func):
    """
    Sync decorator that scans the user prompt for markers in the form <<filename>>.
    - HTTP(S) images are fetched with urllib and wrapped as types.Part.from_bytes
    - Local images are opened via PIL.Image
    The resulting list (mixing text and images) is passed to the wrapped function.
    """

    def wrapper(system_prompt: str, user_prompt: str, **kwargs):
        from google.genai import types
        from urllib.error import URLError

        # If already pre-processed, just pass through
        if not isinstance(user_prompt, str):
            return func(system_prompt, user_prompt, **kwargs)

        parts = _split_prompt(user_prompt)
        new_contents: List[Union[str, object]] = []

        for idx, part in enumerate(parts):
            if idx % 2 == 1:
                image_source = part.strip()

                # HTTP(S) source
                if image_source.startswith("http://") or image_source.startswith("https://"):
                    try:
                        img_part = _fetch_url_image_sync(image_source, types)
                        new_contents.append(img_part)
                    except (URLError, OSError):
                        # If fetching fails, leave the marker as text
                        new_contents.append(f"<<{part}>>")
                else:
                    # Local file path
                    try:
                        img = _open_local_image_sync(image_source)
                        new_contents.append(img)
                    except (FileNotFoundError, OSError):
                        new_contents.append(f"<<{part}>>")
            else:
                if part:
                    new_contents.append(part)

        return func(system_prompt, new_contents, **kwargs)

    return wrapper