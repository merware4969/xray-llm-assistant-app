from dataclasses import dataclass
from typing import Tuple

@dataclass
class ReportInputs:
    class_names: Tuple[str, str]
    pred_idx: int
    probs: Tuple[float, float]
    cam_notes: str  # 예: "오른쪽 하엽 주변에 주목 영역 집중"

class PromptBuilder:
    @staticmethod
    def build_report_prompt(x: ReportInputs) -> str:
        label = x.class_names[x.pred_idx]
        confidence = x.probs[x.pred_idx] * 100
        other = x.class_names[1 - x.pred_idx]
        other_conf = x.probs[1 - x.pred_idx] * 100

        return f"""당신은 영상의학과 임상보조 설명가입니다. 아래 모델 결과를 토대로 '교육용' 안내문을 작성하세요.
- 의료진의 진단을 대체하지 않음을 명확히 고지
- 모델의 한계와 Grad-CAM 해석 주의사항 포함
- 용어는 쉬운 설명 먼저, 괄호로 전문어 병기
- 6~10문장 내외로 간결히

[모델 결과]
- 예측: {label} (신뢰도 {confidence:.2f}%)
- 대안 클래스: {other} (신뢰도 {other_conf:.2f}%)
- Grad-CAM 메모: {x.cam_notes}

[요청 출력 섹션]
1) 요약 판단(교육용, 비진단 고지 포함)
2) 모델이 주목한 영상 특징(Grad-CAM 기반 해석 주의)
3) 추가 확인/추가 촬영 또는 임상증상 확인 권고
4) 촬영 품질/라벨링 등 주의(필요 시)
"""
