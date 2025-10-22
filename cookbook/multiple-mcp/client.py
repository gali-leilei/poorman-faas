import asyncio
import base64
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, FilePath, HttpUrl


class UploadRequest(BaseModel):
    name: str
    # base64 encoded string
    dot_file: str
    script: str

    @classmethod
    def from_file(
        cls, name: str, script_file: FilePath, dot_file: FilePath
    ) -> "UploadRequest":
        with open(script_file, "rb") as f:
            script_content = f.read()
            b64_script = base64.b64encode(script_content).decode("utf-8")
        with open(dot_file, "rb") as g:
            dot_file_content = g.read()
            b64_dot_file = base64.b64encode(dot_file_content).decode("utf-8")
        return cls(name=name, dot_file=b64_dot_file, script=b64_script)


class UploadResponse(BaseModel):
    name: str
    url: HttpUrl
    code: int
    message: str

    @classmethod
    def from_response(cls, name: str, response: httpx.Response) -> "UploadResponse":
        payload = response.json()
        # ending with slash is important:
        # >>> from urllib.parse import urljoin
        # >>> x = 'https://faas.miromind.site/faas/gateway/service-2d607\
        # b7-183a-408e-a0af-56060fe5bc8f'
        # >>> urljoin(x, "mcp")
        # 'https://faas.miromind.site/faas/gateway/mcp'
        # >>> urljoin(x, "/mcp")
        # 'https://faas.miromind.site/mcp'
        # >>> urljoin(x, "./mcp")
        # 'https://faas.miromind.site/faas/gateway/mcp'
        # >>> y = x + "/"
        # >>> y
        # 'https://faas.miromind.site/faas/gateway/service-2d607b7-183a-408e-a0af-56060fe5bc8f/'
        # >>> urljoin(y, "mcp")
        # 'https://faas.miromind.site/faas/gateway/service-2d607b7-183a-408e-a0af-56060fe5bc8f/mcp'
        # >>> urljoin(y, "/mcp")
        # 'https://faas.miromind.site/mcp'
        # >>> urljoin(y, "./mcp")
        # 'https://faas.miromind.site/faas/gateway/service-2d607b7-183a-408e-a0af-56060fe5bc8f/mcp'
        base_url = payload["url"]
        if not base_url.endswith("/"):
            base_url += "/"
        final_url = urljoin(base_url, "./mcp")
        return cls(
            name=name,
            url=HttpUrl(final_url),
            code=payload["code"],
            message=payload["message"],
        )


class AsyncFaas(BaseModel):
    BaseURL: HttpUrl
    # no need for now
    # ApiKey: str

    def upload_url(self) -> str:
        return urljoin(str(self.BaseURL), "faas/admin/python")

    async def upload_one(self, script: UploadRequest) -> UploadResponse:
        # Set a longer timeout for uploads (60 seconds for connection, 300 seconds for read)
        timeout = httpx.Timeout(10.0, read=300.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url=self.upload_url(),
                json=script.model_dump(),
            )
            response.raise_for_status()
            return UploadResponse.from_response(script.name, response)

    async def upload_many(self, scripts: list[UploadRequest]) -> list[UploadResponse]:
        async with asyncio.TaskGroup() as tg:
            responses = []
            for script in scripts:
                response = await tg.create_task(self.upload_one(script))
                responses.append(response)
            return responses
