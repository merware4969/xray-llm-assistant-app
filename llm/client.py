# client.py (drop-in replacement)

import os
import json
from typing import Optional
from dataclasses import dataclass
import requests

from config import (
    LLM_PROVIDER,
    LLM_MODEL,
    OPENAI_API_KEY,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    OLLAMA_HOST,
    REQUEST_TIMEOUT,      # int (초)
    # ★ Gemini
    GEMINI_API_KEY, 
    GEMINI_API_BASE
    # 선택: config에 있다면 사용. 없다면 아래에서 os.environ로도 fallback 처리함.
    # LLM_TEMPERATURE,
)

@dataclass
class LLMResponse:
    text: str


# ------------------------------
# 공통: HTTP 래퍼 (정밀 오류 메시지)
# ------------------------------
def _http_post(url: str, **kw) -> requests.Response:
    try:
        r = requests.post(url, **kw)
        r.raise_for_status()
        return r
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "N/A"
        body = ""
        if e.response is not None:
            # 에러 본문을 깔끔히 요약
            try:
                j = e.response.json()
                body = json.dumps(
                    {
                        "type": j.get("error", {}).get("type"),
                        "code": j.get("error", {}).get("code"),
                        "message": j.get("error", {}).get("message", "")[:600],
                    },
                    ensure_ascii=False,
                )
            except Exception:
                body = (e.response.text or "")[:800]
        raise RuntimeError(f"LLM HTTP {status}: {body}") from e
    except requests.exceptions.Timeout:
        raise RuntimeError("LLM 요청이 타임아웃되었습니다. REQUEST_TIMEOUT을 늘려보세요.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("LLM 서버에 연결할 수 없습니다. 네트워크/방화벽/호스트 설정을 확인하세요.")


def _build_messages(system: Optional[str], prompt: str):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


# ------------------------------
# Base
# ------------------------------
class BaseLLMClient:
    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        raise NotImplementedError


# ------------------------------
# OpenAI
# ------------------------------
class OpenAIClient(BaseLLMClient):
    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

        # 프록시/게이트웨이를 쓰는 환경을 위해 BASE 커스터마이즈 허용
        base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com").rstrip("/")
        url = f"{base}/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        temperature = _get_float_env("LLM_TEMPERATURE", 0.2)
        max_tokens = _get_int_env("LLM_MAX_TOKENS", 512)

        payload = {
            "model": LLM_MODEL,
            "messages": _build_messages(system, prompt),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        r = _http_post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        data = r.json()

        # 방어적 파싱
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            content = json.dumps(data, ensure_ascii=False)[:1500]
        return LLMResponse(text=content)


# ------------------------------
# Azure OpenAI
# ------------------------------
class AzureOpenAIClient(BaseLLMClient):
    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and deployment):
            raise RuntimeError(
                "Azure 설정 누락: AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / AZURE_OPENAI_DEPLOYMENT를 확인하세요."
            )

        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        url = f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"

        headers = {
            "api-key": AZURE_OPENAI_API_KEY,
            "Content-Type": "application/json",
        }

        temperature = _get_float_env("LLM_TEMPERATURE", 0.2)
        max_tokens = _get_int_env("LLM_MAX_TOKENS", 512)

        payload = {
            "messages": _build_messages(system, prompt),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        r = _http_post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        data = r.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            content = json.dumps(data, ensure_ascii=False)[:1500]
        return LLMResponse(text=content)


# ------------------------------
# Ollama
# ------------------------------
class OllamaClient(BaseLLMClient):
    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        host = (OLLAMA_HOST or "http://localhost:11434").rstrip("/")
        url = f"{host}/api/chat"

        temperature = _get_float_env("LLM_TEMPERATURE", 0.2)

        payload = {
            "model": LLM_MODEL,
            "messages": _build_messages(system, prompt),
            "options": {"temperature": temperature},
            # "stream": False  # 기본 비스트리밍
        }

        r = _http_post(url, json=payload, timeout=max(REQUEST_TIMEOUT, 120))
        data = r.json()

        # Ollama 응답 파서 (버전에 따라 달라질 수 있음)
        # 일반적으로 {"message": {"role":"assistant","content":"..."}, "done":true}
        txt = ""
        try:
            if isinstance(data, dict) and "message" in data:
                txt = data["message"].get("content", "")
            elif isinstance(data, dict) and "messages" in data and data["messages"]:
                txt = data["messages"][-1].get("content", "")
            else:
                txt = json.dumps(data, ensure_ascii=False)[:1500]
        except Exception:
            txt = json.dumps(data, ensure_ascii=False)[:1500]
        return LLMResponse(text=txt)

# ★★★★★ Gemini Client 추가 ★★★★★
class GeminiClient(BaseLLMClient):
    """
    Google Generative Language API v1beta
    POST {BASE}/v1beta/models/{model}:generateContent?key=API_KEY
    """
    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
        base = (GEMINI_API_BASE or "https://generativelanguage.googleapis.com").rstrip("/")
        model = os.environ.get("GEMINI_MODEL", LLM_MODEL)  # LLM_MODEL을 그대로 써도 됨
        url = f"{base}/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"

        # Gemini는 'contents' 형식. system을 붙일 땐 간단히 프리펜드
        user_text = (system + "\n\n" + prompt) if system else prompt
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": user_text}]}
            ],
            "generationConfig": {
                "temperature": _get_float_env("LLM_TEMPERATURE", 0.2),
                "maxOutputTokens": _get_int_env("LLM_MAX_TOKENS", 512)
            }
            # 필요 시 safetySettings 추가 가능
        }

        headers = {"Content-Type": "application/json"}
        r = _http_post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        data = r.json()

        # 응답 파싱
        # 보통: data["candidates"][0]["content"]["parts"][0]["text"]
        try:
            cands = data.get("candidates", [])
            if cands:
                parts = cands[0].get("content", {}).get("parts", [])
                if parts and "text" in parts[0]:
                    return LLMResponse(text=parts[0]["text"])
            # 방어적 폴백
            return LLMResponse(text=json.dumps(data, ensure_ascii=False)[:1500])
        except Exception:
            return LLMResponse(text=json.dumps(data, ensure_ascii=False)[:1500])

# ------------------------------
# Factory
# ------------------------------
# --- 팩토리 업데이트 ---
def get_llm_client() -> BaseLLMClient:
    provider = (LLM_PROVIDER or "none").lower()
    if provider == "openai" and OPENAI_API_KEY:
        return OpenAIClient()
    if provider == "azure" and AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
        return AzureOpenAIClient()
    if provider == "ollama":
        return OllamaClient()
    if provider == "gemini" and GEMINI_API_KEY:
        return GeminiClient()

    class NoopClient(BaseLLMClient):
        def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
            return LLMResponse(text="[LLM 비활성화: 환경변수 설정 필요]")
    return NoopClient()
