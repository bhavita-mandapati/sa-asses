"""Thin HTTP front for the router: POST /v1/completions  (Authorization: Bearer <tenant key>)."""
from __future__ import annotations

import json
import logging
import math
import sys

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from router import GatewayError, ModelRouter, new_request_id

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
MAX_BODY = 1_000_000


def create_app(router: ModelRouter) -> Starlette:
    def err(e: GatewayError) -> JSONResponse:
        e.request_id = e.request_id or new_request_id()   # errors raised here (bad JSON, too large) get ids too
        headers = {"X-Request-Id": e.request_id} if e.request_id else {}
        if e.retry_after is not None:
            headers["Retry-After"] = str(max(1, math.ceil(e.retry_after)))
        return JSONResponse(e.to_payload(), status_code=e.status, headers=headers)

    async def completions(request: Request):
        auth = request.headers.get("authorization", "")
        scheme, _, key = auth.partition(" ")
        api_key = key.strip() if scheme.lower() == "bearer" else None
        body = await request.body()
        if len(body) > MAX_BODY:
            return err(GatewayError(413, "request_too_large", "Request body too large."))
        try:
            payload = json.loads(body)
        except ValueError:
            return err(GatewayError(400, "invalid_request", "Request body must be valid JSON."))
        try:
            result = await router.complete(api_key, payload)
        except GatewayError as e:
            return err(e)
        return JSONResponse(result, headers={"X-Request-Id": result["gateway"]["request_id"]})

    return Starlette(routes=[Route("/v1/completions", completions, methods=["POST"])])
