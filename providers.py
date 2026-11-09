"""HTTP provider adapter (OpenAI-compatible). Maps transport/HTTP failures onto router error types."""
from __future__ import annotations

from typing import Any

import httpx

from router import CompletionRequest, ProviderConnectionError, ProviderHTTPError, ProviderTimeout


class HttpProvider:
    def __init__(self, name: str, url: str, api_key: str | None = None, *, model: str | None = None,
                 client: httpx.AsyncClient | None = None) -> None:
        self.name, self.url, self.api_key, self.model = name, url, api_key, model
        # Per-request deadline is enforced by the router; this is only a backstop for hung sockets.
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))

    async def complete(self, request: CompletionRequest) -> dict[str, Any]:
        body = {**request.raw, "stream": False}
        if self.model:
            body["model"] = self.model
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = await self.client.post(self.url, json=body, headers=headers)
        except httpx.TimeoutException as e:
            raise ProviderTimeout(repr(e)) from e
        except httpx.TransportError as e:
            raise ProviderConnectionError(repr(e)) from e
        if r.status_code >= 400:
            ra = r.headers.get("retry-after")
            raise ProviderHTTPError(r.status_code, body=r.text,
                                    retry_after=float(ra) if ra and ra.replace(".", "", 1).isdigit() else None)
        try:
            data = r.json()
        except ValueError as e:
            raise ProviderHTTPError(502, body="non-JSON upstream response") from e
        if not isinstance(data, dict):
            raise ProviderHTTPError(502, body="unexpected upstream payload")
        return data
