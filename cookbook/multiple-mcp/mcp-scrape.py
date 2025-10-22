# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx",
#     "mcp",
#     "starlette",
# ]
# ///
import asyncio
import json
import logging
import os
from typing import Any, Dict

import httpx
from mcp.server.fastmcp import Context, FastMCP

# Configure logging
log_level = os.getenv("VERL_LOGGING_LEVEL", "WARNING")
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Web Scrape with LLM Summary")

SUMMARY_LLM_BASE_URL = os.environ.get("SUMMARY_LLM_BASE_URL")
SUMMARY_LLM_MODEL_NAME = os.environ.get("SUMMARY_LLM_MODEL_NAME")
SUMMARY_LLM_API_KEY = os.environ.get("SUMMARY_LLM_API_KEY")


@mcp.tool()
async def scrape_and_extract_info(
    context: Context, url: str, info_to_extract: str
) -> Dict[str, Any]:
    """
    Scrape content from a URL and extract specific types of information using LLM.

    Args:
        url (str): The URL to scrape content from
        info_to_extract (str): The specific types of information to extract (usually a question)

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Whether the operation was successful
            - url (str): The original URL
            - extracted_info (str): The extracted information
            - error (str): Error message if the operation failed
            - scrape_stats (Dict): Statistics about the scraped content
            - model_used (str): The model used for summarization
            - tokens_used (int): Number of tokens used (if available)
    """
    # Extract LLM configuration from custom headers or use defaults
    llm_base_url = None
    llm_model_name = None
    llm_api_key = None
    config_overrides = []
    request = context.request_context.request
    request_headers = request.headers if request else None

    if request_headers:
        llm_base_url = request_headers.get("X-Summary-LLM-Base-URL")
        llm_model_name = request_headers.get("X-Summary-LLM-Model-Name")
        llm_api_key = request_headers.get("X-Summary-LLM-API-Key")

        if llm_base_url:
            config_overrides.append(f"SUMMARY_LLM_BASE_URL: {llm_base_url}")
        if llm_model_name:
            config_overrides.append(f"SUMMARY_LLM_MODEL_NAME: {llm_model_name}")
        if llm_api_key:
            config_overrides.append(f"SUMMARY_LLM_API_KEY: ***{llm_api_key[-4:]}")

    # Use defaults if not overridden
    llm_base_url = llm_base_url or SUMMARY_LLM_BASE_URL or ""
    llm_model_name = llm_model_name or SUMMARY_LLM_MODEL_NAME or ""
    llm_api_key = llm_api_key or SUMMARY_LLM_API_KEY or ""

    # Print configuration information
    if config_overrides:
        print(f"[Config] Tool called with overrides: {', '.join(config_overrides)}")
    else:
        print(
            f"[Config] Tool called with default config: SUMMARY_LLM_BASE_URL={llm_base_url}, SUMMARY_LLM_MODEL_NAME={llm_model_name}, SUMMARY_LLM_API_KEY=***{llm_api_key[-8:] if llm_api_key else 'None'}"
        )

    if _is_huggingface_dataset_or_space_url(url):
        return {
            "success": False,
            "url": url,
            "extracted_info": "",
            "error": "You are trying to scrape a Hugging Face dataset for answers, please do not use the scrape tool for this purpose.",
            "scrape_stats": {},
            "tokens_used": 0,
        }
    scrape_result = await scrape_url_with_jina(url)

    if not scrape_result["success"]:
        print(
            f"Jina Scrape and Extract Info: Scraping failed: {scrape_result['error']}"
        )
        return {
            "success": False,
            "url": url,
            "extracted_info": "",
            "error": f"Scraping failed: {scrape_result['error']}",
            "scrape_stats": {},
            "tokens_used": 0,
        }

    # Then, summarize the content
    extracted_result = await extract_info_with_llm(
        url=url,
        content=scrape_result["content"],
        info_to_extract=info_to_extract,
        model=llm_model_name,
        max_tokens=8192,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
    )
    # Combine results
    return {
        "success": extracted_result["success"],
        "url": url,
        "extracted_info": extracted_result["extracted_info"],
        "error": extracted_result["error"],
        "scrape_stats": {
            "line_count": scrape_result["line_count"],
            "char_count": scrape_result["char_count"],
            "last_char_line": scrape_result["last_char_line"],
            "all_content_displayed": scrape_result["all_content_displayed"],
        },
        "model_used": extracted_result["model_used"],
        "tokens_used": extracted_result["tokens_used"],
    }


def _is_huggingface_dataset_or_space_url(url):
    """
    Check if the URL is a Hugging Face dataset or space URL.
    :param url: The URL to check
    :return: True if it's a HuggingFace dataset or space URL, False otherwise
    """
    if not url:
        return False
    return "huggingface.co/datasets" in url or "huggingface.co/spaces" in url


async def scrape_url_with_jina(url: str, max_chars: int = 102400 * 4) -> Dict[str, Any]:
    """
    Scrape content from a URL and save to a temporary file. Need to read the content from the temporary file.


    Args:
        url (str): The URL to scrape content from
        context (Context): The context of the request
        max_chars (int): Maximum number of characters to reserve for the scraped content

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Whether the operation was successful
            - filename (str): Absolute path to the temporary file containing the scraped content
            - content (str): The scraped content of the first 40k characters
            - error (str): Error message if the operation failed
            - line_count (int): Number of lines in the scraped content
            - char_count (int): Number of characters in the scraped content
            - last_char_line (int): Line number where the last displayed character is located
            - all_content_displayed (bool): Signal indicating if all content was displayed (True if content <= 40k chars)


    """

    # Validate input
    if not url or not url.strip():
        return {
            "success": False,
            "filename": "",
            "content": "",
            "error": "URL cannot be empty",
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    # Get API key from environment
    jina_api_key = os.getenv("JINA_API_KEY")
    if not jina_api_key:
        return {
            "success": False,
            "filename": "",
            "content": "",
            "error": "JINA_API_KEY environment variable is not set",
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    jina_base_url = os.getenv("JINA_BASE_URL", "https://miro-api.miromind.site/jina")

    # Construct the Jina.ai API URL
    jina_url = f"{jina_base_url}/{url}"

    response = None
    try:
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {jina_api_key}",
        }

        # Retry configuration
        retry_delays = [1, 2, 4, 8]

        for attempt, delay in enumerate(retry_delays, 1):
            try:
                # Make the request using httpx library
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        jina_url,
                        headers=headers,
                        timeout=httpx.Timeout(None, connect=20, read=60),
                        follow_redirects=True,  # Follow redirects (equivalent to curl -L)
                    )

                # Check if request was successful
                response.raise_for_status()
                break  # Success, exit retry loop

            except httpx.ConnectTimeout as e:
                # connection timeout, retry
                if attempt < len(retry_delays):
                    delay = retry_delays[attempt]
                    print(
                        f"Jina Scrape: Connection timeout, {delay}s before next attempt (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(
                        f"Jina Scrape: Connection retry attempts exhausted, url: {url}"
                    )
                    raise e

            except httpx.ConnectError as e:
                # connection error, retry
                if attempt < len(retry_delays):
                    delay = retry_delays[attempt]
                    print(
                        f"Jina Scrape: Connection error: {e}, {delay}s before next attempt"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(
                        f"Jina Scrape: Connection retry attempts exhausted, url: {url}"
                    )
                    raise e

            except httpx.ReadTimeout as e:
                # read timeout, retry
                if attempt < len(retry_delays):
                    delay = retry_delays[attempt]
                    print(
                        f"Jina Scrape: Read timeout, {delay}s before next attempt (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(
                        f"Jina Scrape: Read timeout retry attempts exhausted, url: {url}"
                    )
                    raise e

            except httpx.HTTPStatusError as e:
                if attempt < len(retry_delays):
                    if response is not None:
                        print(
                            f"Jina Scrape: HTTP error: {e}, response.text: {response.text}, url: {url}, {delay}s before next attempt (attempt {attempt + 1})"
                        )
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(
                        f"Jina Scrape: HTTP error retry attempts exhausted, url: {url}"
                    )
                    raise e

            except httpx.RequestError as e:
                if attempt < len(retry_delays):
                    print(
                        f"Jina Scrape: Unknown request exception: {e}, url: {url}, {delay}s before next attempt (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(
                        f"Jina Scrape: Unknown request exception retry attempts exhausted, url: {url}"
                    )
                    raise e

    except Exception as e:
        error_msg = f"Jina Scrape: Unexpected error occurred: {str(e)}"
        print(error_msg)
        return {
            "success": False,
            "filename": "",
            "content": "",
            "error": error_msg,
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    # Get the scraped content
    content = response.text if response else ""

    if not content:
        return {
            "success": False,
            "filename": "",
            "content": "",
            "error": "No content returned from Jina.ai API",
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    # handle insufficient balance error
    try:
        content_dict = json.loads(content)
    except json.JSONDecodeError:
        content_dict = None
    if (
        isinstance(content_dict, dict)
        and content_dict.get("name") == "InsufficientBalanceError"
    ):
        return {
            "success": False,
            "filename": "",
            "content": "",
            "error": "Insufficient balance",
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    # Get content statistics
    total_char_count = len(content)
    total_line_count = content.count("\n") + 1 if content else 0

    # Extract first max_chars characters
    displayed_content = content[:max_chars]
    all_content_displayed = total_char_count <= max_chars

    # Calculate the line number of the last character displayed
    if displayed_content:
        # Count newlines up to the last displayed character
        last_char_line = displayed_content.count("\n") + 1
    else:
        last_char_line = 0

    return {
        "success": True,
        "content": displayed_content,
        "error": "",
        "line_count": total_line_count,
        "char_count": total_char_count,
        "last_char_line": last_char_line,
        "all_content_displayed": all_content_displayed,
    }


EXTRACT_INFO_PROMPT = """You are given a piece of content and the requirement of information to extract. Your task is to extract the information specifically requested. Be precise and focus exclusively on the requested information.

INFORMATION TO EXTRACT:
{}

INSTRUCTIONS:
1. Extract the information relevant to the focus above.
2. If the exact information is not found, extract the most closely related details.
3. Be specific and include exact details when available.
4. Clearly organize the extracted information for easy understanding.
5. Do not include general summaries or unrelated content.

CONTENT TO ANALYZE:
{}

EXTRACTED INFORMATION:"""


def get_prompt_with_truncation(
    info_to_extract: str, content: str, truncate_last_num_chars: int = -1
) -> str:
    if truncate_last_num_chars > 0:
        content = content[:-truncate_last_num_chars] + "[...truncated]"

    # Prepare the prompt
    prompt = EXTRACT_INFO_PROMPT.format(info_to_extract, content)
    return prompt


async def extract_info_with_llm(
    url: str,
    content: str,
    info_to_extract: str,
    model: str = "LLM",
    max_tokens: int = 4096,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
) -> Dict[str, Any]:
    """
    Summarize content using an LLM API.

    Args:
        content (str): The content to summarize
        info_to_extract (str): The specific types of information to extract (usually a question)
        model (str): The model to use for summarization
        max_tokens (int): Maximum tokens for the response
        llm_base_url (str): LLM API base URL (optional, uses default if not provided)
        llm_api_key (str): LLM API key (optional, uses default if not provided)

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Whether the operation was successful
            - extracted_info (str): The extracted information
            - error (str): Error message if the operation failed
            - model_used (str): The model used for summarization
            - tokens_used (int): Number of tokens used (if available)
    """

    # Use provided values or fall back to defaults
    base_url = llm_base_url or SUMMARY_LLM_BASE_URL or ""
    api_key = llm_api_key or SUMMARY_LLM_API_KEY or ""

    # Validate input
    if not content or not content.strip():
        return {
            "success": False,
            "extracted_info": "",
            "error": "Content cannot be empty",
            "model_used": model,
            "tokens_used": 0,
        }

    prompt = get_prompt_with_truncation(info_to_extract, content)

    # Determine API type based on URL ending
    is_chat_completions_api = base_url.rstrip("/").endswith("/chat/completions")

    # Prepare the payload based on API type
    if is_chat_completions_api:
        # OpenAI chat/completions API format
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            # "temperature": 0.7,
            # "top_p": 0.8,
        }
    else:
        # Custom responses API format
        payload = {
            "model": model,
            "max_output_tokens": max_tokens,
            "input": [
                {"role": "user", "content": prompt},
            ],
            # "temperature": 0.7,
            # "top_p": 0.8,
            # "top_k": 20,
        }

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = None
    try:
        # Retry configuration
        connect_retry_delays = [1, 2, 4, 8]

        for attempt, delay in enumerate(connect_retry_delays, 1):
            try:
                # Make the API request using httpx
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        base_url,
                        headers=headers,
                        json=payload,
                        timeout=httpx.Timeout(None, connect=30, read=300),
                    )

                # Check if the request was successful
                if (
                    "Requested token count exceeds the model's maximum context length"
                    in response.text
                    or "longer than the model's context length" in response.text
                ):
                    prompt = get_prompt_with_truncation(
                        info_to_extract,
                        content,
                        truncate_last_num_chars=40960 * attempt,
                    )  # remove 40k * num_attempts chars from the end of the content
                    if is_chat_completions_api:
                        payload["messages"][0]["content"] = prompt
                    else:
                        payload["input"][0]["content"] = prompt
                    continue  # no need to raise error here, just try again

                response.raise_for_status()
                break  # Success, exit retry loop

            except httpx.ConnectTimeout as e:
                # connection timeout, retry
                if attempt < len(connect_retry_delays):
                    delay = connect_retry_delays[attempt]
                    print(
                        f"Jina Scrape and Extract Info: Connection timeout, {delay}s before next attempt (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(
                        "Jina Scrape and Extract Info: Connection retry attempts exhausted"
                    )
                    raise e

            except httpx.ConnectError as e:
                # connection error, retry
                if attempt < len(connect_retry_delays):
                    delay = connect_retry_delays[attempt]
                    print(
                        f"Jina Scrape and Extract Info: Connection error: {e}, {delay}s before next attempt"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(
                        "Jina Scrape and Extract Info: Connection retry attempts exhausted"
                    )
                    raise e

            except httpx.ReadTimeout as e:
                # read timeout, LLM API is too slow, no need to retry
                if attempt < len(connect_retry_delays):
                    print(
                        f"Jina Scrape and Extract Info: LLM API attempt {attempt} read timeout"
                    )
                    continue
                else:
                    print(
                        f"Jina Scrape and Extract Info: LLM API read timeout retry attempts exhausted, please check the request complexity, information to extract: {info_to_extract}, length of content: {len(content)}, url: {url}"
                    )
                    raise e

            except httpx.HTTPStatusError as e:
                text = response.text if response else ""
                print(
                    f"Jina Scrape and Extract Info: HTTP error for LLM API: {e}, response.text: {text}"
                )
                raise httpx.HTTPStatusError(
                    f"response.text: {text}",
                    request=e.request,
                    response=e.response,
                ) from e

            except httpx.RequestError as e:
                print(f"Jina Scrape and Extract Info: Unknown request exception: {e}")
                raise e
    except Exception as e:
        error_msg = f"Jina Scrape and Extract Info: Unexpected error during LLM API call: {str(e)}"
        print(error_msg)
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": model,
            "tokens_used": 0,
        }

    # Parse the response
    response_data = {}
    try:
        if response is not None:
            response_data = response.json()

    except json.JSONDecodeError as e:
        error_msg = (
            f"Jina Scrape and Extract Info: Failed to parse LLM API response: {str(e)}"
        )
        print(error_msg)
        if response is not None:
            print(f"Raw response: {response.text}")
        else:
            print("No response from LLM API")
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": model,
            "tokens_used": 0,
        }

    print(
        f"Jina Scrape and Extract Info: Info to extract: {info_to_extract}, LLM Response data: {response_data}"
    )

    # Extract summary from response based on API type
    summary = None

    if is_chat_completions_api:
        # Parse chat/completions API response
        if "choices" in response_data and len(response_data["choices"]) > 0:
            try:
                summary = response_data["choices"][0]["message"]["content"]
            except Exception as e:
                error_msg = f"Jina Scrape and Extract Info: Failed to get summary from chat/completions API response: {str(e)}"
                print(error_msg)
                return {
                    "success": False,
                    "extracted_info": "",
                    "error": error_msg,
                    "model_used": model,
                    "tokens_used": 0,
                }
    else:
        # Parse responses API response
        if "output" in response_data and len(response_data["output"]) > 0:
            try:
                summary = response_data["output"][0]["content"][0]["text"]
            except Exception as e:
                error_msg = f"Jina Scrape and Extract Info: Failed to get summary from responses API response: {str(e)}"
                print(error_msg)
                return {
                    "success": False,
                    "extracted_info": "",
                    "error": error_msg,
                    "model_used": model,
                    "tokens_used": 0,
                }

    if summary is not None:
        # Extract token usage if available
        tokens_used = 0
        if "usage" in response_data:
            tokens_used = response_data["usage"].get("total_tokens", 0)

        return {
            "success": True,
            "extracted_info": summary,
            "error": "",
            "model_used": model,
            "tokens_used": tokens_used,
        }
    elif "error" in response_data:
        error_msg = (
            f"Jina Scrape and Extract Info: LLM API error: {response_data['error']}"
        )
        print(error_msg)
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": model,
            "tokens_used": 0,
        }
    else:
        error_msg = "Jina Scrape and Extract Info: No valid response from LLM API"
        print(error_msg)
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": model,
            "tokens_used": 0,
        }


if __name__ == "__main__":
    import uvicorn
    from starlette.responses import JSONResponse

    port = int(os.getenv("PORT", "8000"))

    # Health check endpoint
    async def health_check(request):
        """
        Health check endpoint for readiness probe.
        """
        return JSONResponse({"status": "ok"})

    mcp.settings.json_response = True
    mcp.settings.stateless_http = True

    asgi_app = mcp.streamable_http_app()
    asgi_app.add_route("/health", health_check)
    uvicorn.run(asgi_app, host="0.0.0.0", port=port)
