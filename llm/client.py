import os
from typing import Optional
from dataclasses import dataclass
import requests
from config import LLM_PROVIDER, OPENAI_API_KEY, AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, OLLAMA_HOST, LLM_MODEL

@dataclass
class LLMResponse:
    text: str

class BaseLLMClient:
    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        raise NotImplementedError

class OpenAIClient(BaseLLMClient):
    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        # 최신 SDK가 환경마다 달라 호환 이슈가 잦습니다.
        # 여기서는 HTTP 형태의 예시를 남깁니다(필요 시 공식 SDK로 교체).
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": LLM_MODEL, "messages": messages, "temperature": 0.2}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return LLMResponse(text=data["choices"][0]["message"]["content"])

class AzureOpenAIClient(BaseLLMClient):
    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        # 배포명 등 환경마다 상이 — 실제 값에 맞춰 수정
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", LLM_MODEL)
        url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{deployment}/chat/completions?api-version=2024-02-15-preview"
        headers = {"api-key": AZURE_OPENAI_API_KEY, "Content-Type": "application/json"}
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"messages": messages, "temperature": 0.2}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return LLMResponse(text=data["choices"][0]["message"]["content"])

class OllamaClient(BaseLLMClient):
    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        url = f"{OLLAMA_HOST}/api/chat"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": LLM_MODEL, "messages": messages, "options": {"temperature": 0.2}}
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        # ollama는 스트리밍도 제공. 여기선 단발 응답 가정
        txt = r.json().get("message", {}).get("content", "")
        return LLMResponse(text=txt)

def get_llm_client() -> BaseLLMClient:
    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        return OpenAIClient()
    if LLM_PROVIDER == "azure" and AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
        return AzureOpenAIClient()
    if LLM_PROVIDER == "ollama":
        return OllamaClient()
    class NoopClient(BaseLLMClient):
        def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
            return LLMResponse(text="[LLM 비활성화: 환경변수 설정 필요]")
    return NoopClient()