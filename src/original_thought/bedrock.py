"""Thin wrappers around Bedrock: text generation via Converse, embeddings via Titan v2."""

import json
import time

import boto3
from botocore.config import Config

REGION = "us-east-1"
EMBED_MODEL = "amazon.titan-embed-text-v2:0"

_client = None


def client():
    global _client
    if _client is None:
        _client = boto3.client(
            "bedrock-runtime",
            region_name=REGION,
            config=Config(retries={"max_attempts": 10, "mode": "adaptive"}, read_timeout=300),
        )
    return _client


def generate(model_id: str, prompt: str, *, temperature: float = 1.0,
             top_p: float | None = 1.0, max_tokens: int = 600) -> dict:
    """One completion via the Converse API. Returns text plus token usage.

    Claude 4.5+ rejects temperature and topP together; pass top_p=None to omit it.
    """
    inference_config = {"temperature": temperature, "maxTokens": max_tokens}
    if top_p is not None and not model_id.startswith(("us.anthropic.", "global.anthropic.")):
        inference_config["topP"] = top_p
    for attempt in range(6):
        try:
            resp = client().converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig=inference_config,
            )
            return {
                "text": resp["output"]["message"]["content"][0]["text"],
                "usage": resp["usage"],
                "stop_reason": resp["stopReason"],
            }
        except client().exceptions.ThrottlingException:
            time.sleep(2**attempt)
    raise RuntimeError(f"throttled out after retries: {model_id}")


def embed(text: str, dimensions: int = 512) -> list[float]:
    """Titan v2 embedding (normalized)."""
    resp = client().invoke_model(
        modelId=EMBED_MODEL,
        body=json.dumps({"inputText": text[:8000], "dimensions": dimensions, "normalize": True}),
    )
    return json.loads(resp["body"].read())["embedding"]
