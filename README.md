# LLM_chest_xray_project
딥러닝으로 진행한 프로젝트에 LLM을 접목시켜 고도화시키고 Gradio로 구현하는 프로젝트

xray_llm_app/
├── app.py                          # Gradio 진입점
├── config.py                       # 설정/경로/하이퍼파라미터
├── requirements.txt
├── models/
│   └── best_resnet34_addval.pth    # 기존 가중치
├── assets/
│   └── KoPubDotumMedium.ttf
├── classifiers/
│   └── image_classifier.py         # ResNet34 래퍼 클래스
├── explainers/
│   └── gradcam.py                  # Grad-CAM 생성기
├── llm/
│   ├── client.py                   # LLM 추상/구현 클라이언트
│   └── prompt.py                   # 프롬프트 템플릿/후처리
├── services/
│   └── analyzer.py                 # DL 결과 + Grad-CAM + LLM 리포트 오케스트레이션
└── utils/
    └── imaging.py                  # 공용 전처리/시각화 유틸
