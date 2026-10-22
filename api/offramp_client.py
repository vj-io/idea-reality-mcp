"""NVIDIA Nemotron on Nebius Token Factory (added by Offramp).

Token Factory is OpenAI-compatible: migrated call sites use the OpenAI SDK
with the client below. Nemotron reasons by default, so every call passes
extra_body={"chat_template_kwargs": {"enable_thinking": <bool>}}.
"""

import os

from openai import AsyncOpenAI, OpenAI

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"

# Nemotron tiers on Token Factory, cheapest first (ids from GET /v1/models).
LIGHTNING = "nvidia/Nemotron-3_5-Lightning"
SUPER = "nvidia/nemotron-3-super-120b-a12b"
ULTRA = "nvidia/Nemotron-3-Ultra-550b-a55b"


def token_factory_client() -> OpenAI:
    """Token Factory client: reads NEBIUS_API_KEY (and NEBIUS_BASE_URL)."""
    return OpenAI(
        api_key=os.environ["NEBIUS_API_KEY"],
        base_url=os.environ.get("NEBIUS_BASE_URL", NEBIUS_BASE_URL),
    )


def token_factory_async_client(**options) -> AsyncOpenAI:
    """Async Token Factory client for the call sites that were async (they
    still await the call): reads NEBIUS_API_KEY (and NEBIUS_BASE_URL).
    `options` are the replaced client's own, such as timeout=.

    A test fakes it the way it faked the old SDK client: patch
    `token_factory_async_client` where a module imports it by name, or, for a
    call site that imports it inside the function, put a module holding a
    fake one in sys.modules["offramp_client"] (patch.dict).
    """
    return AsyncOpenAI(
        api_key=os.environ["NEBIUS_API_KEY"],
        base_url=os.environ.get("NEBIUS_BASE_URL", NEBIUS_BASE_URL),
        **options,
    )
