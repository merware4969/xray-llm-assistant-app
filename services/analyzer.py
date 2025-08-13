from dataclasses import dataclass
from PIL import Image
import torch
from classifiers.image_classifier import ImageClassifier, Prediction
from explainers.gradcam import GradCAMGenerator
from llm.client import get_llm_client
from llm.prompt import PromptBuilder, ReportInputs

@dataclass
class AnalysisResult:
    prediction: Prediction
    overlay_image: Image.Image
    llm_report: str

class Analyzer:
    def __init__(self, classifier: ImageClassifier):
        self.classifier = classifier
        self.gradcam = GradCAMGenerator(self.classifier.model)
        self.llm = get_llm_client()

    def run(self, image: Image.Image) -> AnalysisResult:
        # 1) 예측
        pred = self.classifier.predict(image)

        # 2) Grad-CAM
        x = self.classifier.t(image.convert("RGB")).unsqueeze(0).to(self.classifier.device)
        with torch.enable_grad():
            x.requires_grad_(True)
            cam = self.gradcam.generate(x, pred.pred_idx)
        overlay = self.gradcam.overlay_on_image(cam, image, alpha=0.5)

        # 3) CAM 간단 요약(정교화 원하면 여기서 contour/영역 위치 추정)
        cam_notes = "히트맵 고강도 영역에 집중(정확한 병변 위치와 일치하지 않을 수 있음)."

        # 4) LLM 리포트
        prompt = PromptBuilder.build_report_prompt(
            ReportInputs(class_names=pred.class_names, pred_idx=pred.pred_idx, probs=pred.probs, cam_notes=cam_notes)
        )
        report = "[LLM 비활성화 또는 오류]"
        try:
            report = self.llm.generate(
                prompt, system="당신은 의료영상 보조설명가입니다. 안전하고 책임있는 안내를 제공합니다."
            ).text
        except Exception as e:
            # 콘솔에도 남겨서 서버 로그로 확인 가능
            import traceback
            print("[LLM ERROR]", repr(e))
            traceback.print_exc()
            # UI에는 메시지까지 보여준다
            report = f"[LLM 오류] {e}"

        return AnalysisResult(prediction=pred, overlay_image=overlay, llm_report=report)
