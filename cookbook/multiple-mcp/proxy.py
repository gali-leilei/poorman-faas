from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, cast

from client import AsyncFaas, UploadRequest
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import TextContent
from pydantic import BaseModel, HttpUrl


class ProxyProtocol(Protocol):
    """Proxy for multiple MCP servers."""

    async def get_all_tool_definitions(self) -> Any: ...
    async def execute_tool_call(
        self, *, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> Any: ...


class ConfigBase(BaseModel):
    name: str


class StdIOConfig(ConfigBase):
    kind: Literal["stdio"]
    params: StdioServerParameters


class SSEConfig(ConfigBase):
    kind: Literal["sse"]
    url: HttpUrl


class StreamableHttpConfig(ConfigBase):
    kind: Literal["streamable_http"]
    url: HttpUrl


Config = StdIOConfig | SSEConfig | StreamableHttpConfig


@asynccontextmanager
async def connect(cfg: Config):
    """
    Returns a `mcp.ClientSession` instance, depending on `Config`.
    """
    async with AsyncExitStack() as stack:
        read, write = None, None
        if cfg.kind == "stdio":
            cfg = cast(StdIOConfig, cfg)
            read, write = await stack.enter_async_context(stdio_client(cfg.params))
        elif cfg.kind == "sse":
            cfg = cast(SSEConfig, cfg)
            read, write = await stack.enter_async_context(sse_client(str(cfg.url)))
        elif cfg.kind == "streamable_http":
            cfg = cast(StreamableHttpConfig, cfg)
            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(str(cfg.url))
            )
        else:  # type: ignore
            raise TypeError("unknown kind {} in cfg".format(cfg.kind))
        if read is not None and write is not None:
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            yield session


class LoggingMixin:
    """
    add logging instance (.task_log) and helper functions (info(), error()) to any class.
    """

    task_log: Any = None

    def add_log(self, logger: Any):
        self.task_log = logger

    def _log(self, level: str, step_name: str, message: str, metadata=None):
        """Helper method to log using task_log if available, otherwise skip logging."""
        if self.task_log:
            self.task_log.log_step(level, step_name, message, metadata)

    def info(self, step_name: str, message: str):
        self._log("info", f"ToolManagerV2 | {step_name}", message)

    def error(self, step_name: str, message: str):
        self._log("error", f"ToolManagerV2 | {step_name}", message)


@dataclass
class Proxy(ProxyProtocol, LoggingMixin):
    """
    A local proxy for multiple MCP servers.
    Supports all transport types: stdio, sse, streamable_http.
    """

    server_dict: Mapping[str, Config]

    @classmethod
    async def from_uploaded_scripts(
        cls, client: AsyncFaas, scripts: list[UploadRequest]
    ) -> "Proxy":
        """
        Create a Proxy from a dictionary of PEP 723 python scripts.
        :param client: AsyncClient
        :param scripts: Dictionary of UploadRequests
        :return: Proxy
        """

        responses = await client.upload_many(scripts)
        server_dict = {
            response.name: StreamableHttpConfig(
                name=response.name,
                kind="streamable_http",
                url=response.url,
            )
            for response in responses
        }

        return cls(server_dict=server_dict)

    @classmethod
    async def from_local_servers(cls, server_dict: Mapping[str, Config]) -> "Proxy":
        """
        Create a Proxy from a dictionary of local servers.
        :param server_dict: Dictionary of server configurations
        :return: Proxy
        """
        return cls(server_dict=server_dict)

    async def get_all_tool_definitions(self):
        """
        Connect to all configured servers and get their tool definitions.
        Returns a list suitable for passing to the Prompt generator.
        """

        async def inner_list_tools(session: ClientSession):
            """helper function to reduce indentation level"""
            try:
                response = await session.list_tools()
                return response, None
            except Exception as e:
                return None, e

        final = []
        # Process remote server tools
        for name, config in self.server_dict.items():
            self.info(
                "Get Tool Definitions",
                f"Getting tool definitions for server '{name}'...",
            )
            curr = {"name": name, "tools": []}
            try:
                async with connect(config) as session:
                    response, error = await inner_list_tools(session)
                    if error is not None:
                        self.error(
                            "List Tools Error",
                            f"Unable to connect or get tools from server '{name}': {str(error)}",
                        )
                        curr["tools"] = [
                            {"error": f"Unable to fetch tools: {str(error)}"}
                        ]
                    if response is not None:
                        for tool in response.tools:
                            curr["tools"].append(
                                {
                                    "name": tool.name,  # type: ignore
                                    "description": tool.description,
                                    "schema": tool.inputSchema,
                                }
                            )
            except Exception as e:
                self.error("MCP session Error", f"MCP session error: {str(e)}")
                curr["tools"] = [{"error": f"MCP session error: {str(e)}"}]
            finally:
                final.append(curr)

        return final

    async def execute_tool_call(
        self, *, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        """
        Execute a single tool call.
        :param server_name: Server name
        :param tool_name: Tool name
        :param arguments: Tool arguments dictionary
        :return: Dictionary containing result or error
        """

        def rv(*, exc: str | None = None, res: str | None = None):
            common = {"server_name": server_name, "tool_name": tool_name}
            depends = {"error": exc} if exc is not None else {"result": res}
            return common | depends

        async def inner_call_tool(sess: ClientSession, name: str, args: dict[str, Any]):
            """helper function to reduce indentation level"""
            try:
                tool_result = await sess.call_tool(name, arguments=args)
                final = ""
                if tool_result is not None:
                    if (
                        getattr(tool_result, "content", None) is not None
                        and len(tool_result.content) > 0
                    ):
                        block = tool_result.content[-1]
                        if isinstance(block, TextContent):
                            final = block.text
                return final, None
            except Exception as e:
                return None, e

        config = self.server_dict.get(server_name, None)
        if config is None:
            self.error(
                "Server Not Found",
                f"Attempting to call server '{server_name}' not found",
            )
            return rv(exc=f"Server '{server_name}' not found.")

        self.info(
            "Tool Call Start",
            f"Connecting to server '{server_name}' to call tool '{tool_name}'",
        )
        try:
            async with connect(config) as session:
                res, exc = await inner_call_tool(session, tool_name, arguments)
                if exc is not None:
                    self.error(
                        "Tool Execution Error",
                        f"Tool execution error: {exc}",
                    )
                    return rv(exc=f"Tool execution failed: {str(exc)}")
                if res is not None:
                    return rv(res=res)
        except Exception as e:
            self.error(
                "MCP Session Error",
                f"MCP session error: {e}",
            )
            return rv(exc=f"MCP session error: {str(e)}")
