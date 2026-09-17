import io
from base64 import b64encode
from typing import Any

from langchain.tools import tool
from langchain_core.messages import ImageContentBlock
from langchain_core.messages.content import create_image_block
from PIL.Image import Image

from .memory import LongTermMemory


def convert_to_img_block(img: Image) -> ImageContentBlock:
    # Save image data into a buffer
    buf = io.BytesIO()
    img.save(buf, format="JPEG", optimize=True)
    img_bytes = buf.getvalue()

    return create_image_block(
        base64=b64encode(img_bytes).decode("utf-8"),
        mime_type="image/jpeg",
        detail="original",
    )


@tool
async def search_long_term_memory(query: str) -> list[dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
    """
    Semantically search long-term memory for facts about past projects and
    decisions from before today. Today's activity is already in Short-Term
    Memory in the system prompt — search this only when that isn't enough.
    Contains no facts about the user.

    Args:
        query: A specific, descriptive query — a topic, project, or
            decision, not the user's raw message.

    Returns:
        Matching memories, most relevant first. Each item carries the
        memory text under "memory" plus metadata such as "score" and
        timestamps — heavier than a plain string, so search selectively.
    """

    result = await LongTermMemory.client.search(
        query, filters={"app_id": "com.rehatsingh.marc"}
    )
    memories = result["results"]
    return memories
