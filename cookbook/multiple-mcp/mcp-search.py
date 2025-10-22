# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx",
#     "mcp",
#     "requests",
#     "starlette",
#     "tenacity",
#     "wikipedia",
# ]
# ///
import asyncio
import calendar
import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
import requests
import wikipedia
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
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
JINA_BASE_URL = os.environ.get("JINA_BASE_URL", "https://miro-api.miromind.site/jina")

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


def _filter_search_results_by_blocked_urls(
    search_results: List[str], blocked_urls: Optional[List[str]], context: str = ""
) -> List[str]:
    """
    Filter search results to exclude pages that match blocked URL patterns.

    Args:
        search_results: List of search result titles to filter
        blocked_urls: List of URL patterns to exclude (optional)
        context: Context string for debug logging (e.g., "disambiguation", "PageError")

    Returns:
        List of filtered search results that don't match blocked patterns
    """
    if not blocked_urls:
        # No filtering needed
        return search_results

    filtered_results = []
    for result in search_results:
        try:
            # Check if this search result would lead to a blocked page
            test_page = wikipedia.page(title=result, auto_suggest=False)
            page_url = test_page.url

            # Check if the page URL matches any blocked URLs
            is_blocked = any(
                _url_matches_blocked_pattern(page_url, blocked_url)
                for blocked_url in blocked_urls
            )

            if not is_blocked:
                filtered_results.append(result)
                logger.debug(f"✅ Keeping {context} search result: {result}")
            else:
                logger.debug(
                    f"🚫 Filtering {context} search result: {result} (URL: {page_url} matches blocked URLs)"
                )

        except wikipedia.exceptions.PageError:
            # If page doesn't exist, skip it (it's effectively blocked)
            logger.debug(
                f"🚫 Filtering {context} search result: {result} (page not found)"
            )
        except Exception as e:
            # For other exceptions, include the result to be safe
            logger.debug(
                f"⚠️ Cannot check {context} search result: {result} ({str(e)}), keeping it"
            )
            filtered_results.append(result)

    return filtered_results


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
    location: str = None,
    num: int = 10,
    tbs: str = None,
    page: int = 1,
    autocorrect: bool = None,
    blocked_urls: Optional[List[str]] = None,
) -> str:
    """Internal function to perform google searches via Serper API with optional URL filtering.

    Args:
        q: Search query string.
        gl: Country context for search (e.g., 'us' for United States, 'cn' for China, 'uk' for United Kingdom). Influences regional results priority. Default is 'us'.
        hl: Google interface language (e.g., 'en' for English, 'zh' for Chinese, 'es' for Spanish). Affects snippet language preference. Default is 'en'.
        location: City-level location for search results (e.g., 'SoHo, New York, United States', 'California, United States').
        num: The number of results to return (default: 10).
        tbs: Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, 'qdr:w' for past week, 'qdr:m' for past month, 'qdr:y' for past year).
        page: The page number of results to return (default: 1).
        autocorrect: Whether to autocorrect spelling in query
        blocked_urls: List of URL patterns to exclude from search results (optional, internal use only).

    Returns:
        The search results, optionally filtered.
    """
    if SERPER_API_KEY == "":
        return "SERPER_API_KEY is not set, google_search tool is not available."

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

    # Validate required parameter
    if not q or not q.strip():
        return {
            "success": False,
            "error": "Search query 'q' is required and cannot be empty",
            "results": [],
        }
    try:
        # tool_name = "google_search"
        payload = {
            "q": q.strip(),
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
            "searchParameters": data.get("searchParameters", {}),
            "organic": organic_results,
        }
        result_content = json.dumps(response_data, ensure_ascii=False, indent=2)

        # Apply URL filtering if blocked URLs are provided
        if blocked_urls:
            result_content = filter_blocked_urls(result_content, blocked_urls)

        return decode_percent_encoded_urls(
            result_content, input_type="text"
        )  # Success, exit retry loop
    except Exception as error:
        return f"[ERROR]: google_search tool execution failed: {str(error)}"

    return "[ERROR]: Unknown error occurred in google_search tool, please try again."


# @mcp.tool()
async def google_search_internal_filtered(
    q: str,
    blocked_urls: List[str],
    gl: str = "us",
    hl: str = "en",
    location: str = None,
    num: int = 10,
    tbs: str = None,
    page: int = 1,
) -> str:
    """🔒 INTERNAL: Google search with URL filtering via MCP. Hidden from model prompts.

    This tool is registered in the MCP server but blacklisted in ToolManager
    to prevent exposure to the model while allowing internal MCP calls.

    Args:
        q: Search query string.
        blocked_urls_json: JSON string of URL patterns to exclude from search results.
        gl: Geographic location code (default: "us").
        hl: Language code (default: "en").
        location: Location for search results.
        num: The number of results to return (default: 10).
        tbs: Time-based search filter.
        page: The page number of results to return (default: 1).

    Returns:
        The filtered search results.
    """
    return await _google_search_internal(
        q=q,
        gl=gl,
        hl=hl,
        location=location,
        num=num,
        tbs=tbs,
        page=page,
        blocked_urls=blocked_urls,
    )


@mcp.tool()
async def google_search(
    q: str,
    gl: str = "us",
    hl: str = "en",
    location: str = None,
    num: int = 10,
    tbs: str = None,
    page: int = 1,
    autocorrect: bool = None,
) -> str:
    """Perform google searches via Serper API and retrieve rich results.
    It is able to retrieve organic search results, people also ask, related searches, and knowledge graph.

    Args:
        q: Search query string.
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
    return await _google_search_internal(
        q=q,
        gl=gl,
        hl=hl,
        location=location,
        num=num,
        tbs=tbs,
        page=page,
        autocorrect=autocorrect,
        blocked_urls=None,
    )


async def _wiki_get_page_content_internal(
    entity: str, first_sentences: int = 10, blocked_urls: Optional[List[str]] = None
) -> str:
    """Internal function to get Wikipedia page content with optional URL filtering.

    Args:
        entity: The entity to search for in Wikipedia.
        first_sentences: Number of first sentences to return from the page. Set to 0 to return full content. Defaults to 10.
        blocked_urls: List of URL patterns to exclude from search results (optional, internal use only).

    Returns:
        str: Formatted search results containing title, first sentences/full content, and URL.
             Returns error message if page not found or other issues occur.
    """
    try:
        # Try to get the Wikipedia page directly
        page = wikipedia.page(title=entity, auto_suggest=False)

        # Check if the page URL is blocked
        if blocked_urls:
            for blocked_url in blocked_urls:
                if _url_matches_blocked_pattern(page.url, blocked_url):
                    logger.debug(
                        f"🚫 Blocking Wikipedia page: {page.url} (matches {blocked_url})"
                    )
                    # Return as if page not found
                    raise wikipedia.exceptions.PageError(f"Page blocked: {entity}")

        # Prepare the result
        result_parts = [f"Page Title: {page.title}"]

        if first_sentences > 0:
            # Get summary with specified number of sentences
            try:
                summary = wikipedia.summary(
                    entity, sentences=first_sentences, auto_suggest=False
                )
                result_parts.append(
                    f"First {first_sentences} sentences (introduction): {summary}"
                )
            except Exception:
                # Fallback to page summary if direct summary fails
                content_sentences = page.content.split(". ")[:first_sentences]
                summary = (
                    ". ".join(content_sentences) + "."
                    if content_sentences
                    else page.content[:5000] + "..."
                )
                result_parts.append(
                    f"First {first_sentences} sentences (introduction): {summary}"
                )
        else:
            # Return full content if first_sentences is 0
            # TODO: Context Engineering Needed
            result_parts.append(f"Content: {page.content}")

        result_parts.append(f"URL: {page.url}")

        return "\n\n".join(result_parts)

    except wikipedia.exceptions.DisambiguationError as e:
        # Filter out blocked options if blocked_urls is provided
        available_options = e.options[:10]  # Limit to first 10

        # Filter disambiguation options to exclude blocked URLs
        available_options = _filter_search_results_by_blocked_urls(
            available_options, blocked_urls, "disambiguation option"
        )

        # If no options remain after filtering, return as page not found
        if not available_options:
            try:
                search_results = wikipedia.search(entity, results=5)
                if search_results:
                    # Filter search results to exclude blocked URLs
                    filtered_search_results = _filter_search_results_by_blocked_urls(
                        search_results[:5],
                        blocked_urls,
                        "disambiguation search suggestion",
                    )

                    if filtered_search_results:
                        suggestion_list = "\n".join(
                            [f"- {result}" for result in filtered_search_results]
                        )
                        return (
                            f"[ERROR]: Page Not Found: No Wikipedia page found for '{entity}'.\n\n"
                            f"Similar pages found:\n{suggestion_list}\n\n"
                            f"Try searching for one of these suggestions instead."
                        )
                    else:
                        # All search suggestions were filtered out
                        return (
                            f"[ERROR]: Page Not Found: No Wikipedia page found for '{entity}'. "
                            f"Some similar pages were found but are not accessible."
                        )
                else:
                    return (
                        f"[ERROR]: Page Not Found: No Wikipedia page found for '{entity}' "
                        f"and no similar pages were found. Please try a different search term."
                    )
            except Exception as search_error:
                return (
                    f"[ERROR]: Page Not Found: No Wikipedia page found for '{entity}'. "
                    f"Search for alternatives also failed: {str(search_error)}"
                )

        options_list = "\n".join([f"- {option}" for option in available_options])
        output = (
            f"[ERROR]: Disambiguation Error: Multiple pages found for '{entity}'.\n\n"
            f"Available options:\n{options_list}\n\n"
            f"Please be more specific in your search query."
        )

        try:
            search_results = wikipedia.search(entity, results=5)
            if search_results:
                # Filter search results to exclude blocked URLs for disambiguation additional search
                filtered_search_results = _filter_search_results_by_blocked_urls(
                    search_results[:5], blocked_urls, "disambiguation additional search"
                )

                if filtered_search_results:
                    output += f"Try to search {entity} in Wikipedia: {filtered_search_results}"
            return output
        except Exception:
            pass

        return output

    except wikipedia.exceptions.PageError:
        # Try a search if direct page lookup fails
        try:
            search_results = wikipedia.search(entity, results=5)
            if search_results:
                # Filter search results to exclude blocked URLs
                filtered_search_results = _filter_search_results_by_blocked_urls(
                    search_results[:5], blocked_urls, "PageError search suggestion"
                )

                if filtered_search_results:
                    suggestion_list = "\n".join(
                        [f"- {result}" for result in filtered_search_results]
                    )
                    return (
                        f"[ERROR]: Page Not Found: No Wikipedia page found for '{entity}'.\n\n"
                        f"Similar pages found:\n{suggestion_list}\n\n"
                        f"Try searching for one of these suggestions instead."
                    )
                else:
                    # All search suggestions were filtered out
                    return (
                        f"[ERROR]: Page Not Found: No Wikipedia page found for '{entity}'. "
                        f"Some similar pages were found but are not accessible."
                    )
            else:
                return (
                    f"[ERROR]: Page Not Found: No Wikipedia page found for '{entity}' "
                    f"and no similar pages were found. Please try a different search term."
                )
        except Exception as search_error:
            return (
                f"[ERROR]: Page Not Found: No Wikipedia page found for '{entity}'. "
                f"Search for alternatives also failed: {str(search_error)}"
            )

    except wikipedia.exceptions.RedirectError:
        return f"[ERROR]: Redirect Error: Failed to follow redirect for '{entity}'"

    except requests.exceptions.RequestException as e:
        return f"[ERROR]: Network Error: Failed to connect to Wikipedia: {str(e)}"

    except wikipedia.exceptions.WikipediaException as e:
        return f"[ERROR]: Wikipedia Error: An error occurred while searching Wikipedia: {str(e)}"

    except Exception as e:
        return f"[ERROR]: Unexpected Error: An unexpected error occurred: {str(e)}"


# @mcp.tool()
async def wiki_get_page_content_internal_filtered(
    entity: str, blocked_urls: List[str], first_sentences: int = 10
) -> str:
    """🔒 INTERNAL: Wikipedia page content with URL filtering via MCP. Hidden from model prompts.

    This tool is registered in the MCP server but blacklisted in ToolManager
    to prevent exposure to the model while allowing internal MCP calls.

    Args:
        entity: The entity to search for in Wikipedia.
        blocked_urls_json: JSON string of URL patterns to exclude from search results.
        first_sentences: Number of first sentences to return from the page. Set to 0 to return full content. Defaults to 10.

    Returns:
        str: Formatted search results containing title, first sentences/full content, and URL.
             Returns error message if page not found or other issues occur.
    """
    return await _wiki_get_page_content_internal(entity, first_sentences, blocked_urls)


# @mcp.tool()
async def wiki_get_page_content(entity: str, first_sentences: int = 10) -> str:
    """Get specific Wikipedia page content for the specific entity (people, places, concepts, events) and return structured information.

    This tool searches Wikipedia for the given entity and returns either the first few sentences
    (which typically contain the summary/introduction) or full page content based on parameters.
    It handles disambiguation pages and provides clean, structured output.

    Args:
        entity: The entity to search for in Wikipedia.
        first_sentences: Number of first sentences to return from the page. Set to 0 to return full content. Defaults to 10.

    Returns:
        str: Formatted search results containing title, first sentences/full content, and URL.
             Returns error message if page not found or other issues occur.
    """
    return await _wiki_get_page_content_internal(
        entity, first_sentences, blocked_urls=None
    )


# @mcp.tool()
async def search_wiki_revision(
    entity: str, year: int, month: int, max_revisions: int = 50
) -> str:
    """Search for an entity in Wikipedia and return the revision history for a specific month.

    Args:
        entity: The entity to search for in Wikipedia.
        year: The year of the revision (e.g. 2024).
        month: The month of the revision (1-12).
        max_revisions: Maximum number of revisions to return. Defaults to 50.

    Returns:
        str: Formatted revision history with timestamps, revision IDs, and URLs.
             Returns error message if page not found or other issues occur.
    """
    # Auto-adjust date values and track changes
    adjustments = []
    original_year, original_month = year, month
    current_year = datetime.datetime.now().year

    # Adjust year to valid range
    if year < 2000:
        year = 2000
        adjustments.append(
            f"Year adjusted from {original_year} to 2000 (minimum supported)"
        )
    elif year > current_year:
        year = current_year
        adjustments.append(
            f"Year adjusted from {original_year} to {current_year} (current year)"
        )

    # Adjust month to valid range
    if month < 1:
        month = 1
        adjustments.append(f"Month adjusted from {original_month} to 1")
    elif month > 12:
        month = 12
        adjustments.append(f"Month adjusted from {original_month} to 12")

    # Prepare adjustment message if any changes were made
    if adjustments:
        adjustment_msg = (
            "Date auto-adjusted: "
            + "; ".join(adjustments)
            + f". Using {year}-{month:02d} instead.\n\n"
        )
    else:
        adjustment_msg = ""

    base_url = "https://en.wikipedia.org/w/api.php"

    try:
        # Construct the time range
        start_date = datetime.datetime(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]
        end_date = datetime.datetime(year, month, last_day, 23, 59, 59)

        # Convert to ISO format (UTC time)
        start_iso = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        # API parameters configuration
        params = {
            "action": "query",
            "format": "json",
            "titles": entity,
            "prop": "revisions",
            "rvlimit": min(max_revisions, 500),  # Wikipedia API limit
            "rvstart": start_iso,
            "rvend": end_iso,
            "rvdir": "newer",
            "rvprop": "timestamp|ids",
        }

        content = await smart_request(url=base_url, params=params)
        data = request_to_json(content)

        # Check for API errors
        if "error" in data:
            return f"[ERROR]: Wikipedia API Error: {data['error'].get('info', 'Unknown error')}"

        # Process the response
        pages = data.get("query", {}).get("pages", {})

        if not pages:
            return f"[ERROR]: No results found for entity '{entity}'"

        # Check if page exists
        page_id = list(pages.keys())[0]
        if page_id == "-1":
            return f"[ERROR]: Page Not Found: No Wikipedia page found for '{entity}'"

        page_info = pages[page_id]
        page_title = page_info.get("title", entity)

        if "revisions" not in page_info or not page_info["revisions"]:
            return (
                adjustment_msg + f"Page Title: {page_title}\n\n"
                f"No revisions found for '{entity}' in {year}-{month:02d}.\n\n"
                f"The page may not have been edited during this time period."
            )

        # Format the results
        result_parts = [
            f"Page Title: {page_title}",
            f"Revision Period: {year}-{month:02d}",
            f"Total Revisions Found: {len(page_info['revisions'])}",
        ]

        # Add revision details
        revisions_details = []
        for i, rev in enumerate(page_info["revisions"], 1):
            revision_id = rev["revid"]
            timestamp = rev["timestamp"]

            # Format timestamp for better readability
            try:
                dt = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                formatted_time = timestamp

            # Construct revision URL
            rev_url = f"https://en.wikipedia.org/w/index.php?title={entity}&oldid={revision_id}"

            revisions_details.append(
                f"{i}. Revision ID: {revision_id}\n"
                f"   Timestamp: {formatted_time}\n"
                f"   URL: {rev_url}"
            )

        if revisions_details:
            result_parts.append("Revisions:\n" + "\n\n".join(revisions_details))

        return (
            adjustment_msg
            + "\n\n".join(result_parts)
            + "\n\nHint: You can use the `scrape_website` tool to get the webpage content of a URL."
        )

    except requests.exceptions.Timeout:
        return f"[ERROR]: Network Error: Request timed out while fetching revision history for '{entity}'"

    except requests.exceptions.RequestException as e:
        return f"[ERROR]: Network Error: Failed to connect to Wikipedia: {str(e)}"

    except ValueError as e:
        return f"[ERROR]: Date Error: Invalid date values - {str(e)}"

    except Exception as e:
        return f"[ERROR]: Unexpected Error: An unexpected error occurred: {str(e)}"


# @mcp.tool()
async def search_archived_webpage(url: str, year: int, month: int, day: int) -> str:
    """Search the Wayback Machine (archive.org) for archived versions of a webpage, optionally for a specific date.

    Args:
        url: The URL to search for in the Wayback Machine.
        year: The target year (e.g., 2023).
        month: The target month (1-12).
        day: The target day (1-31).

    Returns:
        str: Formatted archive information including archived URL, timestamp, and status.
             Returns error message if URL not found or other issues occur.
    """
    # Handle empty URL
    if not url:
        return f"[ERROR]: Invalid URL: '{url}'. URL cannot be empty."

    # Auto-add https:// if no protocol is specified
    protocol_hint = ""
    if not url.startswith(("http://", "https://")):
        original_url = url
        url = f"https://{url}"
        protocol_hint = f"[NOTE]: Automatically added 'https://' to URL '{original_url}' -> '{url}'\n\n"

    hint_message = ""
    if ".wikipedia.org" in url:
        hint_message = "Note: You are trying to search a Wikipedia page, you can also use the `search_wiki_revision` tool to get the revision content of a Wikipedia page.\n\n"

    # Check if specific date is requested
    date = ""
    adjustment_msg = ""
    if year > 0 and month > 0:
        # Auto-adjust date values and track changes
        adjustments = []
        original_year, original_month, original_day = year, month, day
        current_year = datetime.datetime.now().year

        # Adjust year to valid range
        if year < 1995:
            year = 1995
            adjustments.append(
                f"Year adjusted from {original_year} to 1995 (minimum supported)"
            )
        elif year > current_year:
            year = current_year
            adjustments.append(
                f"Year adjusted from {original_year} to {current_year} (current year)"
            )

        # Adjust month to valid range
        if month < 1:
            month = 1
            adjustments.append(f"Month adjusted from {original_month} to 1")
        elif month > 12:
            month = 12
            adjustments.append(f"Month adjusted from {original_month} to 12")

        # Adjust day to valid range for the given month/year
        max_day = calendar.monthrange(year, month)[1]
        if day < 1:
            day = 1
            adjustments.append(f"Day adjusted from {original_day} to 1")
        elif day > max_day:
            day = max_day
            adjustments.append(
                f"Day adjusted from {original_day} to {max_day} (max for {year}-{month:02d})"
            )

        # Update the date string with adjusted values
        date = f"{year:04d}{month:02d}{day:02d}"

        try:
            # Validate the final adjusted date
            datetime.datetime(year, month, day)
        except ValueError as e:
            return f"[ERROR]: Invalid date: {year}-{month:02d}-{day:02d}. {str(e)}"

        # Prepare adjustment message if any changes were made
        if adjustments:
            adjustment_msg = (
                "Date auto-adjusted: "
                + "; ".join(adjustments)
                + f". Using {date} instead.\n\n"
            )

    try:
        base_url = "https://archive.org/wayback/available"
        # Search with specific date if provided
        if date:
            retry_count = 0
            # retry 5 times if the response is not valid
            while retry_count < 5:
                content = await smart_request(
                    url=base_url, params={"url": url, "timestamp": date}
                )
                data = request_to_json(content)
                if (
                    "archived_snapshots" in data
                    and "closest" in data["archived_snapshots"]
                ):
                    break
                retry_count += 1
                await asyncio.sleep(min(2**retry_count, 60))

            if "archived_snapshots" in data and "closest" in data["archived_snapshots"]:
                closest = data["archived_snapshots"]["closest"]
                archived_url = closest["url"]
                archived_timestamp = closest["timestamp"]
                available = closest.get("available", True)

                if not available:
                    return (
                        hint_message
                        + adjustment_msg
                        + (
                            f"Archive Status: Snapshot exists but is not available\n\n"
                            f"Original URL: {url}\n"
                            f"Requested Date: {year:04d}-{month:02d}-{day:02d}\n"
                            f"Closest Snapshot: {archived_timestamp}\n\n"
                            f"Try a different date"
                        )
                    )

                # Format timestamp for better readability
                try:
                    dt = datetime.datetime.strptime(archived_timestamp, "%Y%m%d%H%M%S")
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                except Exception:
                    formatted_time = archived_timestamp

                return (
                    protocol_hint
                    + hint_message
                    + adjustment_msg
                    + (
                        f"Archive Found: Archived version located\n\n"
                        f"Original URL: {url}\n"
                        f"Requested Date: {year:04d}-{month:02d}-{day:02d}\n"
                        f"Archived URL: {archived_url}\n"
                        f"Archived Timestamp: {formatted_time}\n"
                    )
                    + "\n\nHint: You can also use the `scrape_website` tool to get the webpage content of a URL."
                )

        # Search without specific date (most recent)
        retry_count = 0
        # retry 5 times if the response is not valid
        while retry_count < 5:
            content = await smart_request(url=base_url, params={"url": url})
            data = request_to_json(content)
            if "archived_snapshots" in data and "closest" in data["archived_snapshots"]:
                break
            retry_count += 1
            await asyncio.sleep(min(2**retry_count, 60))

        if "archived_snapshots" in data and "closest" in data["archived_snapshots"]:
            closest = data["archived_snapshots"]["closest"]
            archived_url = closest["url"]
            archived_timestamp = closest["timestamp"]
            available = closest.get("available", True)

            if not available:
                return (
                    protocol_hint
                    + hint_message
                    + (
                        f"Archive Status: Most recent snapshot exists but is not available\n\n"
                        f"Original URL: {url}\n"
                        f"Most Recent Snapshot: {archived_timestamp}\n\n"
                        f"The URL may have been archived but access is restricted"
                    )
                )

            # Format timestamp for better readability
            try:
                dt = datetime.datetime.strptime(archived_timestamp, "%Y%m%d%H%M%S")
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                formatted_time = archived_timestamp

            return (
                protocol_hint
                + hint_message
                + (
                    f"Archive Found: Most recent archived version\n\n"
                    f"Original URL: {url}\n"
                    f"Archived URL: {archived_url}\n"
                    f"Archived Timestamp: {formatted_time}\n"
                )
                + "\n\nHint: You can also use the `scrape_website` tool to get the webpage content of a URL."
            )
        else:
            return (
                protocol_hint
                + hint_message
                + (
                    f"Archive Not Found: No archived versions available\n\n"
                    f"Original URL: {url}\n\n"
                    f"The URL '{url}' has not been archived by the Wayback Machine.\n"
                    f"You may want to:\n"
                    f"- Check if the URL is correct\n"
                    f"- Try a different URL and date\n"
                )
            )

    except requests.exceptions.RequestException as e:
        return f"[ERROR]: Network Error: Failed to connect to Wayback Machine: {str(e)}"

    except ValueError as e:
        return f"[ERROR]: Data Error: Failed to parse response from Wayback Machine: {str(e)}"

    except Exception as e:
        return f"[ERROR]: Unexpected Error: An unexpected error occurred: {str(e)}"


if __name__ == "__main__":
    import uvicorn
    from starlette.responses import JSONResponse

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
    uvicorn.run(asgi_app, host="0.0.0.0", port=8000)
