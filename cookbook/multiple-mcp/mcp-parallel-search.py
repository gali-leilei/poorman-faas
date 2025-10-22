# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx",
#     "mcp",
#     "starlette",
#     "tenacity",
# ]
# ///

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
SERPER_BASE_URL = os.environ.get(
    "SERPER_BASE_URL", "https://miro-api.miromind.site/serper"
)

# Google search result filtering environment variables
REMOVE_SNIPPETS = os.environ.get("REMOVE_SNIPPETS", "").lower() in ("true", "1", "yes")
REMOVE_KNOWLEDGE_GRAPH = os.environ.get("REMOVE_KNOWLEDGE_GRAPH", "").lower() in (
    "true",
    "1",
    "yes",
)
REMOVE_ANSWER_BOX = os.environ.get("REMOVE_ANSWER_BOX", "").lower() in (
    "true",
    "1",
    "yes",
)


async def smart_request(url: str, params: Optional[Dict] = None) -> str:
    """Make a smart HTTP request with retries."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.text


def request_to_json(content: str) -> Dict[str, Any]:
    """Parse JSON from request content."""
    return json.loads(content)


def decode_percent_encoded_urls(content: str, input_type: str = "text") -> str:
    """Decode percent-encoded URLs in content."""
    # Simple implementation - just return content as-is
    # Can be enhanced to actually decode URLs if needed
    return content


def _is_huggingface_dataset_or_space_url(url):
    """
    Check if the URL is a Hugging Face dataset or space URL.
    :param url: The URL to check
    :return: True if it's a HuggingFace dataset or space URL, False otherwise
    """
    if not url:
        return False
    return "huggingface.co/datasets" in url or "huggingface.co/spaces" in url


# Initialize FastMCP server
mcp = FastMCP("searching-mcp-server")


def filter_google_search_result(result_content: str) -> str:
    """Filter google search result content based on environment variables.

    Args:
        result_content: The JSON string result from google search

    Returns:
        Filtered JSON string result
    """
    try:
        # Parse JSON
        data = json.loads(result_content)

        # Remove knowledgeGraph if requested
        if REMOVE_KNOWLEDGE_GRAPH and "knowledgeGraph" in data:
            del data["knowledgeGraph"]

        # Remove answerBox if requested
        if REMOVE_ANSWER_BOX and "answerBox" in data:
            del data["answerBox"]

        # Remove snippets if requested
        if REMOVE_SNIPPETS:
            # Remove snippets from organic results
            if "organic" in data:
                for item in data["organic"]:
                    if "snippet" in item:
                        del item["snippet"]

            # Remove snippets from peopleAlsoAsk
            if "peopleAlsoAsk" in data:
                for item in data["peopleAlsoAsk"]:
                    if "snippet" in item:
                        del item["snippet"]

        # Return filtered JSON
        return json.dumps(data, ensure_ascii=False, indent=2)

    except (json.JSONDecodeError, Exception):
        # If filtering fails, return original content
        return result_content


def _url_matches_blocked_pattern(page_url: str, blocked_url: str) -> bool:
    """
    Check if a page URL matches a blocked URL pattern.

    Uses intelligent matching to avoid false positives:
    - Exact match: page_url == blocked_url
    - Suffix match: page_url ends with blocked_url
    - Path segment match: blocked_url matches complete path segments

    Args:
        page_url: The actual page URL
        blocked_url: The blocked URL pattern

    Returns:
        True if the page should be blocked, False otherwise
    """
    # Exact match
    if page_url == blocked_url:
        return True

    # If blocked_url is a suffix of page_url
    if page_url.endswith(blocked_url):
        return True

    # Check if blocked_url matches a complete path segment
    # For example: "wiki/Apple" should match "wiki/Apple" but not "wiki/Apple_Inc."
    if blocked_url in page_url:
        # Find the position of blocked_url in page_url
        index = page_url.find(blocked_url)
        if index != -1:
            # Check character after the match
            end_index = index + len(blocked_url)
            if end_index < len(page_url):
                next_char = page_url[end_index]
                # If next character is not alphanumeric or underscore, it's a valid match
                if not (next_char.isalnum() or next_char == "_"):
                    return True
            else:
                # blocked_url is at the end of page_url
                return True

    return False


def filter_blocked_urls(search_results: str, blocked_urls: list) -> str:
    """Filter out search results containing blocked URLs.

    This function provides comprehensive filtering across all search result types:
    - Organic results (main links and sitelinks)
    - News results
    - Images results (both image URLs and source URLs)
    - Videos results
    - Shopping results
    - Knowledge graph
    - Answer box
    - Related searches

    Note: This function is used internally by both public and filtered search functions.

    Args:
        search_results: The raw search results string from Serper API (JSON format)
        blocked_urls: List of URLs or URL patterns to exclude (substring matching)

    Returns:
        Filtered search results with blocked URLs removed, maintaining original JSON structure
    """
    logger.debug(f"🔍 Starting URL filtering with blocked_urls: {blocked_urls}")

    if not blocked_urls or not search_results:
        logger.debug("⚠️ No blocked URLs or empty search results, returning original")
        return search_results

    import json

    try:
        # Try to parse as JSON first (Serper typically returns JSON)
        data = json.loads(search_results)
        logger.debug(
            f"✅ Successfully parsed JSON, found sections: {list(data.keys())}"
        )

        # Filter organic results
        if "organic" in data:
            logger.debug(f"📋 Found {len(data['organic'])} organic results to check")
            filtered_organic = []
            for result in data["organic"]:
                url = result.get("link", "")

                # Check main URL
                blocked = False
                for blocked_url in blocked_urls:
                    if _url_matches_blocked_pattern(url, blocked_url):
                        logger.debug(f"🚫 Blocking URL: {url} (matches: {blocked_url})")
                        blocked = True
                        break

                if blocked:
                    continue  # Skip this result

                # Check and filter sitelinks if present
                if "sitelinks" in result:
                    filtered_sitelinks = []
                    for sitelink in result["sitelinks"]:
                        sitelink_url = sitelink.get("link", "")
                        if not any(
                            _url_matches_blocked_pattern(sitelink_url, blocked_url)
                            for blocked_url in blocked_urls
                        ):
                            filtered_sitelinks.append(sitelink)
                        else:
                            logger.debug(
                                f"Skipping blocked sitelinkURL: {sitelink_url}"
                            )
                    result["sitelinks"] = filtered_sitelinks

                filtered_organic.append(result)
            data["organic"] = filtered_organic

        # Filter knowledge graph results if present
        if "knowledgeGraph" in data and "source" in data["knowledgeGraph"]:
            kg_url = data["knowledgeGraph"]["source"].get("link", "")
            if any(
                _url_matches_blocked_pattern(kg_url, blocked_url)
                for blocked_url in blocked_urls
            ):
                data.pop("knowledgeGraph", None)

        # Filter news results if present
        if "news" in data:
            filtered_news = []
            for news in data["news"]:
                news_url = news.get("link", "")
                if not any(
                    _url_matches_blocked_pattern(news_url, blocked_url)
                    for blocked_url in blocked_urls
                ):
                    filtered_news.append(news)
            data["news"] = filtered_news

        # Filter images results if present
        if "images" in data:
            filtered_images = []
            for image in data["images"]:
                # Check both image URL and source URL
                image_url = image.get("imageUrl", "")
                source_url = image.get("link", "")
                if not any(
                    _url_matches_blocked_pattern(image_url, blocked_url)
                    for blocked_url in blocked_urls
                ) and not any(
                    _url_matches_blocked_pattern(source_url, blocked_url)
                    for blocked_url in blocked_urls
                ):
                    filtered_images.append(image)
            data["images"] = filtered_images

        # Filter videos results if present
        if "videos" in data:
            filtered_videos = []
            for video in data["videos"]:
                video_url = video.get("link", "")
                if not any(
                    _url_matches_blocked_pattern(video_url, blocked_url)
                    for blocked_url in blocked_urls
                ):
                    filtered_videos.append(video)
            data["videos"] = filtered_videos

        # Filter shopping results if present
        if "shopping" in data:
            filtered_shopping = []
            for item in data["shopping"]:
                item_url = item.get("link", "")
                if not any(
                    _url_matches_blocked_pattern(item_url, blocked_url)
                    for blocked_url in blocked_urls
                ):
                    filtered_shopping.append(item)
            data["shopping"] = filtered_shopping

        # Filter answer box if present
        if "answerBox" in data:
            answer_box = data["answerBox"]
            if "link" in answer_box:
                answer_url = answer_box.get("link", "")
                if any(
                    _url_matches_blocked_pattern(answer_url, blocked_url)
                    for blocked_url in blocked_urls
                ):
                    data.pop("answerBox", None)

        # Filter related searches if they contain blocked URLs
        if "relatedSearches" in data:
            filtered_related = []
            for search in data["relatedSearches"]:
                query = search.get("query", "")
                # Simple check - you might want to enhance this
                if not any(
                    blocked_url in query.lower() for blocked_url in blocked_urls
                ):
                    filtered_related.append(search)
            data["relatedSearches"] = filtered_related

        # Count final results
        final_organic_count = len(data.get("organic", []))
        logger.debug(
            f"🎯 Filtering complete. Final organic results: {final_organic_count}"
        )

        return json.dumps(data, ensure_ascii=False, indent=2)

    except json.JSONDecodeError:
        # If not JSON, treat as plain text and filter line by line
        lines = search_results.split("\n")
        filtered_lines = []

        for line in lines:
            # Check if line contains any blocked URLs
            if not any(
                _url_matches_blocked_pattern(line, blocked_url)
                for blocked_url in blocked_urls
            ):
                filtered_lines.append(line)

        return "\n".join(filtered_lines)

    except Exception as e:
        # Log the error but don't let filtering failures break the search
        logger.error(f"❌ URL filtering failed: {e}", exc_info=True)
        logger.debug("⚠️ Returning original unfiltered results due to error")
        return search_results


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError)
    ),
)
async def make_serper_request(
    payload: Dict[str, Any], headers: Dict[str, str]
) -> httpx.Response:
    """Make HTTP request to Serper API with retry logic."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SERPER_BASE_URL}/search",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        return response


async def _google_search_internal(
    q: str,
    gl: str = "us",
    hl: str = "en",
    location: str | None = None,
    tbs: str | None = None,
    page: int = 1,
    autocorrect: bool | None = None,
    blocked_urls: Optional[List[str]] = None,
) -> dict:
    """Internal function to perform google searches via Serper API with optional URL filtering.

    Args:
        q: Search query string.
        gl: Country context for search (e.g., 'us' for United States, 'cn' for China, 'uk' for United Kingdom). Influences regional results priority. Default is 'us'.
        hl: Google interface language (e.g., 'en' for English, 'zh' for Chinese, 'es' for Spanish). Affects snippet language preference. Default is 'en'.
        location: City-level location for search results (e.g., 'SoHo, New York, United States', 'California, United States').
        tbs: Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, 'qdr:w' for past week, 'qdr:m' for past month, 'qdr:y' for past year).
        page: The page number of results to return (default: 1).
        autocorrect: Whether to autocorrect spelling in query
        blocked_urls: List of URL patterns to exclude from search results (optional, internal use only).

    Returns:
        Dict with searchParameters and organic results, or error information.
    """

    # 检测查询中是否包含中文字符
    def contains_chinese(text):
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                return True
        return False

    # 如果查询包含中文，自动设置gl和hl参数
    if contains_chinese(q):
        gl = "cn"
        hl = "zh-cn"

    # Build payload first for consistent searchParameters
    payload = {
        "q": q.strip() if q else "",
        "gl": gl,
        "hl": hl,
        "num": 10,  # default value
        "page": page,
        "autocorrect": autocorrect,
    }
    if location:
        payload["location"] = location
    if tbs:
        payload["tbs"] = tbs

    if SERPER_API_KEY == "":
        return {
            "searchParameters": payload,
            "organic": [],
            "error": "SERPER_API_KEY is not set, google_search tool is not available.",
        }

    # Validate required parameter
    if not q or not q.strip():
        return {
            "searchParameters": payload,
            "organic": [],
            "error": "Search query 'q' is required and cannot be empty",
        }

    try:
        # Set up headers
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        }

        # Make the API request
        response = await make_serper_request(payload, headers)
        data = response.json()

        # filter out huggingface dataset or space urls
        organic_results = []
        if "organic" in data:
            for item in data["organic"]:
                if _is_huggingface_dataset_or_space_url(item.get("link", "")):
                    continue
                organic_results.append(item)
        # Build comprehensive response

        response_data = {
            "searchParameters": data.get("searchParameters", payload),
            "organic": organic_results,
        }

        # Apply URL filtering if blocked URLs are provided
        if blocked_urls:
            result_content_str = json.dumps(response_data, ensure_ascii=False, indent=2)
            result_content_str = filter_blocked_urls(result_content_str, blocked_urls)
            # Parse back to dict
            response_data = json.loads(result_content_str)

        return response_data
    except Exception as error:
        return {
            "searchParameters": payload,
            "organic": [],
            "error": f"[ERROR]: google_search tool execution failed: {str(error)}",
        }


@mcp.tool()
async def parallel_google_search(
    queries: List[str],
    gl: str = "us",
    hl: str = "en",
    location: str | None = None,
    tbs: str | None = None,
    page: int = 1,
    autocorrect: bool | None = None,
) -> str:
    """Perform multiple google searches in parallel via Serper API and retrieve rich results.
    It is able to retrieve organic search results, people also ask, related searches, and knowledge graph.

    Args:
        queries: List of search query strings to execute in parallel.
        gl: Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')
        hl: Optional language code for search results in ISO 639-1 format (e.g., 'en')
        location: Location for search results (e.g., 'SoHo, New York, United States', 'California, United States').
        num: The number of results to return (default: 10).
        tbs: Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, 'qdr:w' for past week, 'qdr:m' for past month, 'qdr:y' for past year).
        page: The page number of results to return (default: 1).
        autocorrect: Whether to autocorrect spelling in query

    Returns:
        The search results.
    """

    # Define single search function
    async def single_google_search(q: str) -> dict:
        """Execute a single google search and return dict."""
        return await _google_search_internal(
            q=q,
            gl=gl,
            hl=hl,
            location=location,
            tbs=tbs,
            page=page,
            autocorrect=autocorrect,
            blocked_urls=None,
        )

    # Execute all searches in parallel
    tasks = [single_google_search(q) for q in queries]
    results = await asyncio.gather(*tasks)

    # Add query index to each result
    results_with_index = [
        {"queryIndex": i, **result} for i, result in enumerate(results)
    ]

    # Count successes and errors
    success_count = sum(1 for r in results_with_index if "error" not in r)
    error_count = sum(1 for r in results_with_index if "error" in r)

    # Build organized response
    organized_response = {
        "summary": {
            "totalQueries": len(queries),
            "successCount": success_count,
            "errorCount": error_count,
        },
        "results": results_with_index,
    }

    return json.dumps(organized_response, ensure_ascii=False, indent=2)


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
