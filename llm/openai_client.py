"""
OpenAI-uyumlu ince istemci — OpenRouter, vLLM, ya da herhangi bir
OpenAI-uyumlu endpoint için TEK sarmalayıcı.

Sağlayıcı değiştirmek = sadece base_url + model + api_key ortam değişkeni.
Kod değişmez. (OpenRouter bugün, vLLM yarın — ikisi de aynı API.)

API key ASLA koda/log'a girmez; yalnızca ortam değişkeninden okunur.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key_env: str,
                 default_headers: Optional[dict] = None) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"'{api_key_env}' ortam değişkeni yok. Key'i .env dosyasına ekle "
                f"(örn. {api_key_env}=sk-...). Key koda girmemeli.")
        self.client = OpenAI(base_url=base_url, api_key=api_key,
                             default_headers=default_headers or {})

    def chat(self, model: str, system: str, user: str, temperature: float = 0.7,
             force_json: bool = True, max_tokens: int = 4000,
             web_search: bool = False) -> LLMResponse:
        kwargs: dict = dict(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
        if web_search:
            # OpenRouter web_search aracı (hoca örneği). SDK doğrulamasını atlamak
            # için extra_body ile gönderilir; OpenRouter aramayı kendisi yürütür.
            kwargs["extra_body"] = {"tools": [
                {"type": "openrouter:web_search",
                 "parameters": {"engine": "auto", "max_results": 5}}]}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception:
            # Bazı modeller response_format desteklemez; JSON zorlamadan tekrar dene.
            if force_json:
                kwargs.pop("response_format", None)
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=resp.model or model,
        )
