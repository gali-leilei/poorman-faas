import asyncio
from pathlib import Path

from client import AsyncFaas, UploadRequest
from proxy import Proxy, StreamableHttpConfig


async def test_upload():
    script_path = Path(__file__).parent
    scripts = [
        UploadRequest.from_file(
            name="scrape",
            script_file=script_path / "mcp-scrape.py",
            dot_file=script_path / ".env",
        ),
        UploadRequest.from_file(
            name="search",
            script_file=script_path / "mcp-search.py",
            dot_file=script_path / ".env",
        ),
    ]
    faas = AsyncFaas(BaseURL="https://faas.miromind.site/faas")  # type: ignore

    proxy = await Proxy.from_uploaded_scripts(faas, scripts)

    # test the proxy
    result = await proxy.execute_tool_call(
        server_name="scrape",
        tool_name="scrape",
        arguments={"url": "https://www.google.com"},
    )

    print(result)


class TestLocalServers:
    @staticmethod
    async def test_scrape():
        server_dict = {
            "scrape": StreamableHttpConfig(
                name="scrape",
                kind="streamable_http",
                url="http://localhost:8000/mcp",
            )
        }
        proxy = await Proxy.from_local_servers(server_dict)

        result = await proxy.execute_tool_call(
            server_name="scrape",
            tool_name="scrape_and_extract_info",
            arguments={
                "url": "https://news.ycombinator.com/",
                "info_to_extract": "What are the top stories on Hacker News?",
            },
        )

        print(result)


if __name__ == "__main__":
    # asyncio.run(test_upload())
    asyncio.run(TestLocalServers.test_scrape())
