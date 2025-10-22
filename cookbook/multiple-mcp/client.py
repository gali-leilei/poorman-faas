import asyncio
import base64

import httpx
from pydantic import BaseModel, FilePath, HttpUrl


class UploadRequest(BaseModel):
    name: str
    dot_file: bytes
    script: bytes

    @classmethod
    def from_file(
        cls, name: str, script_file: FilePath, dot_file: FilePath
    ) -> "UploadRequest":
        with open(script_file, "rb") as f:
            script_content = f.read()
            b64_script = base64.b64encode(script_content)
        with open(dot_file, "rb") as g:
            dot_file_content = g.read()
            b64_dot_file = base64.b64encode(dot_file_content)
        return cls(name=name, dot_file=b64_dot_file, script=b64_script)


class UploadResponse(BaseModel):
    name: str
    url: HttpUrl
    code: int
    message: str

    @classmethod
    def from_response(cls, response: httpx.Response) -> "UploadResponse":
        return cls(
            name=response.json()["name"],
            url=response.json()["url"],
            code=response.status_code,
            message=response.text,
        )


class AsyncFaas(BaseModel):
    BaseURL: HttpUrl
    # no need for now
    # ApiKey: str

    def upload_url(self) -> str:
        return f"{self.BaseURL}/faas/admin/python"

    async def upload_one(self, script: UploadRequest) -> UploadResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=self.upload_url(),
                json=script.model_dump(),
            )
            response.raise_for_status()
            return UploadResponse.from_response(response)

    async def upload_many(self, scripts: list[UploadRequest]) -> list[UploadResponse]:
        async with asyncio.TaskGroup() as tg:
            responses = []
            for script in scripts:
                response = await tg.create_task(self.upload_one(script))
                responses.append(response)
            return responses
