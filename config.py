import os
from pathlib import Path

# ---- .env 로드 (최상단) ----
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True), override=True)
except Exception:
    # dotenv 미설치여도 치명적이지 않게 통과
    pass

# 기본 경로
BASE_DIR = Path(__file__).resolve().parent
MODEL_WEIGHTS = BASE_DIR / "models" / "best_resnet34_addval.pth"
FONT_PATH = BASE_DIR / "assets" / "KoPubDotumMedium.ttf"

# 런타임
DEVICE = "cuda" if os.environ.get("FORCE_CPU", "0") != "1" and \
                  __import__("torch").cuda.is_available() else "cpu"
SEED = int(os.environ.get("SEED", 42))

# LLM
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "none")  # openai|azure|ollama|none
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# 선택적 튜닝 파라미터
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", 0.2))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", 60))  # seconds

# UI
APP_TITLE = "X-ray Pneumonia Classifier with LLM Report"
IMAGE_MAX_SIDE = int(os.environ.get("IMAGE_MAX_SIDE", 1024))

# ---- 경로/설정 유효성 점검(조기 실패) ----
def _warn(msg: str):
    print(f"[config] {msg}")

if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
    _warn("OPENAI_API_KEY가 없습니다. .env 또는 OS 환경변수에 설정하세요.")

if LLM_PROVIDER == "azure":
    if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY):
        _warn("Azure 사용 설정이지만 AZURE_OPENAI_ENDPOINT 또는 AZURE_OPENAI_API_KEY 누락.")

if not MODEL_WEIGHTS.exists():
    _warn(f"가중치 파일이 없습니다: {MODEL_WEIGHTS}")

if not FONT_PATH.exists():
    _warn(f"폰트 파일이 없습니다(무시 가능): {FONT_PATH}")
