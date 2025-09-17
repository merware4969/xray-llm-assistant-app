# X-ray LLM Assistant (DL + Gemini 2.0 Flash)

흉부 X-ray 이미지에 대해 딥러닝 분류(ResNet34)와 Grad-CAM 시각화, 그리고 **Gemini 2.0 Flash**(Generative Language API)를 이용한 교육용 설명 리포트를 제공하는 Gradio 앱입니다.

> 주의: 본 도구는 **교육용 보조 도구**이며 의료 전문가의 진단을 대체하지 않습니다.

---

## 주요 기능

- 흉부 X-ray 이미지 업로드
- 딥러닝 이진 분류: NORMAL / PNEUMONIA (+ 확률)
- Grad-CAM 히트맵 오버레이
- **Gemini 2.0 Flash** 기반 해설 리포트(교육용 안내, 한계/주의 포함)
- 모듈/클래스 분리 설계(유지보수/확장 용이)

---

## 📸 시연 화면
![Image](https://github.com/user-attachments/assets/d21d506c-810d-4be7-be63-568f7eeb5c26)

---

## 디렉터리 구조

```
xray_llm_app/
├── app.py                          # Gradio 진입점(UI, 이벤트 바인딩)
├── config.py                       # 공통 설정/경로/환경변수 로드
├── requirements.txt                # 의존성
├── .env                            # 런타임 환경설정 (개인키)
├── models/
│   └── best_resnet34_addval.pth    # 학습된 가중치
├── assets/
│   └── KoPubDotumMedium.ttf        # (선택) 폰트/리소스
├── classifiers/
│   └── image_classifier.py         # ImageClassifier/Prediction
├── explainers/
│   └── gradcam.py                  # GradCAMGenerator
├── llm/
│   ├── client.py                   # LLM 클라이언트(추상 + Gemini 구현)
│   └── prompt.py                   # PromptBuilder/ReportInputs
├── services/
│   └── analyzer.py                 # Analyzer(오케스트레이션)
└── utils/
    └── imaging.py                  # 공용 이미지 유틸
```

---

## 동작 흐름

1. 사용자가 Gradio UI에서 X-ray 업로드 → `btn.click(infer)`
2. `Analyzer.run()`:
   - `ImageClassifier.predict()` → 클래스/확률
   - `GradCAMGenerator.generate()` → CAM 생성
   - `overlay_on_image()` → 오버레이 이미지 생성
   - `PromptBuilder.build_report_prompt()` → 프롬프트 텍스트 생성
   - `LLMClient.generate()`(Gemini 2.0 Flash) → 리포트 텍스트 생성
3. UI에 오버레이 이미지 + 예측 요약 + LLM 리포트 출력

---

## 모듈/클래스 개요

### `config.py`
- 경로/디바이스/하이퍼파라미터/LLM 키 등 런타임 설정 단일화.
- 주요 항목
  - 모델/디바이스: `MODEL_WEIGHTS`, `DEVICE`, `SEED`, `IMAGE_MAX_SIDE`
  - LLM: `LLM_PROVIDER=gemini`, `LLM_MODEL=gemini-2.0-flash`, `REQUEST_TIMEOUT`
  - Gemini: `GEMINI_API_KEY`, `GEMINI_API_BASE(선택)`

### `classifiers/image_classifier.py`
- `ImageClassifier`: ResNet34 이진 분류(헤드 교체, 가중치 로드, 224×224 전처리).
- `Prediction`: `pred_idx`, `probs`, `class_names=("NORMAL", "PNEUMONIA")`.

### `explainers/gradcam.py`
- `GradCAMGenerator`: target layer hook으로 활성맵/그라디언트 캡처 → CAM 생성.
- `overlay_on_image`: heatmap을 원본 이미지에 합성.

### `llm/prompt.py`
- `ReportInputs`: 분류/확률/CAM 요약 입력.
- `PromptBuilder.build_report_prompt`: 교육용·안전 고지·해석 주의·추가 권고 포함 프롬프트 구성.

### `llm/client.py`
- `BaseLLMClient` 추상화, `GeminiClient` 구현체.
  - REST: `POST {GEMINI_API_BASE}/v1beta/models/{LLM_MODEL}:generateContent?key=...`
  - `generationConfig.temperature`, `maxOutputTokens`를 `.env`로 제어.
  - `finishReason`/`usageMetadata` 기반 진단 메시지 확장 가능.
- 공통 HTTP 래퍼로 상태코드/오류 메시지 가시화(401/403/404/429/timeout 등).

### `services/analyzer.py`
- `Analyzer`: 분류 → CAM → 오버레이 → 프롬프트 → LLM 호출 오케스트레이션.
- 예외 처리:
  - LLM 오류 시 메시지 UI 표기.
  - 필요 시 로컬 규칙 기반 리포트로 폴백(운영 연속성).

### `utils/imaging.py`
- `ensure_rgb_pil`: 입력을 PIL RGB로 보정.

### `app.py`
- `.env` 로드, 시드 고정, 모델/Analyzer 싱글톤 초기화.
- Gradio Blocks UI와 이벤트 바인딩(`concurrency_limit` 활용).
- 실행: `demo.launch(server_name="127.0.0.1", server_port=7861, max_threads=2)`.

---

## 설치 및 실행

### 0) 사전 요구
- Python 3.10+ 권장.
- GPU 사용 시: CUDA 호환 PyTorch.

### 1) 의존성 설치
```bash
pip install -r requirements.txt
```

### 2) .env 설정

```dotenv
# LLM (Gemini 2.0 Flash만 사용)
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash
GEMINI_API_KEY=AIzaSy...

# 출력/안정성 튜닝
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=1024
REQUEST_TIMEOUT=120

# 기타(선택)
SEED=42
FORCE_CPU=0
IMAGE_MAX_SIDE=1024
```

### 3) 실행
```bash
python app.py
```
콘솔 출력 예시:
```
[ENV] PROVIDER= gemini
[ENV] MODEL   = gemini-2.0-flash
[ENV] KEY_SET = True
```

---

## 권장 의존성 조합

`requirements.txt` 예시(안정화 조합):
```txt
# Core DL
torch>=2.1
torchvision>=0.16
pillow
opencv-python-headless

# UI
gradio==4.44.1
gradio-client==1.3.0
fastapi>=0.112,<0.113
starlette>=0.38,<0.39
uvicorn>=0.30

# Misc
numpy
requests
python-dotenv

# (중요) Pydantic 호환
pydantic<2.11
pydantic-core<2.26
```

> 메모: 일부 Gradio 버전은 pydantic 2.11대와 스키마 충돌이 있으므로 위 조합을 권장합니다.

---

## 트러블슈팅

- **Gradio 오류: `TypeError: argument of type 'bool' is not iterable`**
  - 원인: 특정 Gradio/Client 버전과 pydantic 2.11대의 스키마 파서 충돌.
  - 해결: `pydantic<2.11`, `pydantic-core<2.26` 고정 + `gradio 4.44.1 / client 1.3.0` 조합.

- **UI 팝업: `No API found`**
  - 원인: 오래된 Service Worker 캐시의 `/info`와 서버 엔드포인트 불일치.
  - 해결: `api_name` 제거, 시크릿 창 접속 또는 캐시 삭제, 새 포트(예: 7861)로 실행.

- **LLM 리포트가 중간에 끊김**
  - 원인: `finishReason != STOP`(특히 `MAX_TOKENS` 도달).
  - 해결: `.env`의 `LLM_MAX_TOKENS` 상향, 필요 시 이어쓰기 로직 추가.

- **Gemini 관련 권한/한도 오류**
  - 403/404: 모델 접근 권한/이름 확인(`LLM_MODEL=gemini-2.0-flash`).
  - 429: 쿼터/요율 한도 초과. 과금/한도 복구 후 재시도.

---

## 추가 제안 사항

- 응답 길이 제어: 프롬프트에 “6~8문장, 간결하게”와 같은 길이 힌트를 포함.
- 비용 제어: `LLM_MAX_TOKENS`/프롬프트 길이 최적화.
- 로깅: LLM 예외 메시지는 UI와 콘솔 모두에 표기하여 진단 시간 단축.

---

## 고지

- 본 프로젝트는 교육/연구 목적의 보조 도구입니다.
- 임상 의사결정은 반드시 의료 전문가의 진단과 책임 하에 이루어져야 합니다.
