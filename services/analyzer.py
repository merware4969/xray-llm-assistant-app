# from dataclasses import dataclass
# from PIL import Image
# import torch
# from classifiers.image_classifier import ImageClassifier, Prediction
# from explainers.gradcam import GradCAMGenerator
# from llm.client import get_llm_client
# from llm.prompt import PromptBuilder, ReportInputs

# @dataclass
# class AnalysisResult:
#     prediction: Prediction
#     overlay_image: Image.Image
#     llm_report: str

# class Analyzer:
#     def __init__(self, classifier: ImageClassifier):
#         self.classifier = classifier
#         self.gradcam = GradCAMGenerator(self.classifier.model)
#         self.llm = get_llm_client()

#     def run(self, image: Image.Image) -> AnalysisResult:
#         # 1) 예측
#         pred = self.classifier.predict(image)

#         # 2) Grad-CAM
#         x = self.classifier.t(image.convert("RGB")).unsqueeze(0).to(self.classifier.device)
#         with torch.enable_grad():
#             x.requires_grad_(True)
#             cam = self.gradcam.generate(x, pred.pred_idx)
#         overlay = self.gradcam.overlay_on_image(cam, image, alpha=0.5)

#         # 3) CAM 간단 요약(정교화 원하면 여기서 contour/영역 위치 추정)
#         cam_notes = "히트맵 고강도 영역에 집중(정확한 병변 위치와 일치하지 않을 수 있음)."

#         # 4) LLM 리포트
#         prompt = PromptBuilder.build_report_prompt(
#             ReportInputs(class_names=pred.class_names, pred_idx=pred.pred_idx, probs=pred.probs, cam_notes=cam_notes)
#         )
#         report = "[LLM 비활성화 또는 오류]"
#         try:
#             report = self.llm.generate(
#                 prompt, system="당신은 의료영상 보조설명가입니다. 안전하고 책임있는 안내를 제공합니다."
#             ).text
#         except Exception as e:
#             # 콘솔에도 남겨서 서버 로그로 확인 가능
#             import traceback
#             print("[LLM ERROR]", repr(e))
#             traceback.print_exc()
#             # UI에는 메시지까지 보여준다
#             report = f"[LLM 오류] {e}"

#         return AnalysisResult(prediction=pred, overlay_image=overlay, llm_report=report)

from dataclasses import dataclass
from typing import Tuple, List
from PIL import Image
import torch
import numpy as np
import cv2

from classifiers.image_classifier import ImageClassifier, Prediction
from explainers.gradcam import GradCAMGenerator
from llm.client import get_llm_client
from llm.prompt import PromptBuilder, ReportInputs


@dataclass
class AnalysisResult:
    prediction: Prediction
    overlay_image: Image.Image
    llm_report: str
    cam_notes: str  # 선택: UI에서도 CAM 요약을 함께 노출하고 싶다면 추가


def summarize_cam(cam: np.ndarray, img_h: int, img_w: int,
                  thr: float = 0.6, topk: int = 3) -> str:
    """
    cam: (H, W) in [0,1] (정규화 필요)
    img_h, img_w: 원본 이미지 크기(서술용)
    thr: 고강도 영역 임계값
    topk: 상위 핫스팟 개수
    """
    H, W = cam.shape[:2]
    cam_bin = (cam >= thr).astype(np.uint8)

    cam_mean = float(cam.mean())
    cam_max = float(cam.max())
    high_ratio = float(cam_bin.sum()) / max(1, (H * W))

    # 연결성분 분석(고강도 영역)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cam_bin, connectivity=8)
    comps = []
    for lbl in range(1, num_labels):
        x, y, w, h, area = stats[lbl]
        cx, cy = centroids[lbl]
        mask = (labels == lbl)
        comp_mean = float(cam[mask].mean())
        comp_max = float(cam[mask].max())
        comps.append({
            "area": int(area),
            "bbox": (int(x), int(y), int(w), int(h)),
            "centroid": (float(cx), float(cy)),
            "mean": comp_mean,
            "max": comp_max,
        })
    comps.sort(key=lambda d: d["area"], reverse=True)
    major = comps[0] if comps else None

    def zone_of(y, x):
        lr = "좌측" if x < W/2 else "우측"
        band = y / max(1, H)
        if band < 1/3:
            tb = "상부"
        elif band < 2/3:
            tb = "중부"
        else:
            tb = "하부"
        return lr, tb

    # 상위 k개 핫스팟 좌표
    flat_idx = np.argsort(cam.ravel())[::-1][:max(topk, 1)]
    ys, xs = np.unravel_index(flat_idx, (H, W))
    hotspots = [(int(x), int(y), float(cam[y, x])) for y, x in zip(ys, xs)]

    lines: List[str] = []
    lines.append(f"CAM 평균 {cam_mean:.2f}, 최대 {cam_max:.2f}. 고강도(≥{thr:.2f}) 면적 비율 {high_ratio*100:.1f}%.")

    if major:
        mx, my = int(major["centroid"][0]), int(major["centroid"][1])
        lr, tb = zone_of(my, mx)
        lines.append(
            f"주요 고강도 영역은 {lr} {tb}에 위치하며, 면적 {major['area']}px, 평균 {major['mean']:.2f}, 최대 {major['max']:.2f}."
        )
    else:
        lines.append("임계값 이상 고강도 연결영역이 뚜렷하지 않습니다.")

    if len(comps) >= 2:
        lines.append(f"고강도 연결영역이 {len(comps)}개로 산재하는 양상.")
    elif len(comps) == 1:
        lines.append("단일 주요 영역에 집중되는 양상.")
    else:
        lines.append("고강도 영역 분포가 희박하여 명확한 집중 부위를 특정하기 어려움.")

    hs_desc = []
    for i, (x, y, v) in enumerate(hotspots, 1):
        lr, tb = zone_of(y, x)
        hs_desc.append(f"{i}위: {lr} {tb} (값 {v:.2f})")
    lines.append("핫스팟: " + "; ".join(hs_desc))

    # 품질/주의 플래그
    if high_ratio > 0.5:
        lines.append("주의: 히트맵이 과도하게 넓게 분포하여 비특이적 주목 가능성.")
    if cam_max < 0.2:
        lines.append("주의: 전반적으로 CAM 세기가 약해 해석 신뢰도가 낮을 수 있음.")

    left_mean = float(cam[:, :W//2].mean())
    right_mean = float(cam[:, W//2:].mean())
    if abs(left_mean - right_mean) > 0.05:
        dom = "좌측" if left_mean > right_mean else "우측"
        lines.append(f"{dom} 편측 주목 경향 관찰.")

    return " ".join(lines)


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
            cam = self.gradcam.generate(x, pred.pred_idx)  # (H, W) 또는 Tensor

        # 2-1) 오버레이 이미지는 종전과 동일
        overlay = self.gradcam.overlay_on_image(cam, image, alpha=0.5)

        # 3) CAM 정량 요약 생성(정규화 포함)
        if isinstance(cam, torch.Tensor):
            cam_np = cam.detach().float().cpu().numpy()
        else:
            cam_np = cam
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)  # [0,1]

        cam_notes = summarize_cam(cam_np, image.height, image.width, thr=0.6, topk=3)

        # 4) LLM 리포트
        prompt = PromptBuilder.build_report_prompt(
            ReportInputs(
                class_names=pred.class_names,
                pred_idx=pred.pred_idx,
                probs=pred.probs,
                cam_notes=cam_notes
            )
        )
        report = "[LLM 비활성화 또는 오류]"
        try:
            report = self.llm.generate(
                prompt,
                system="당신은 의료영상 보조설명가입니다. 안전하고 책임있는 안내를 제공합니다. 숫자값은 수정하지 마세요."
            ).text
        except Exception as e:
            import traceback
            print("[LLM ERROR]", repr(e))
            traceback.print_exc()
            report = f"[LLM 오류] {e}"

        return AnalysisResult(
            prediction=pred,
            overlay_image=overlay,
            llm_report=report,
            cam_notes=cam_notes  # UI에 병행 표기하고 싶지 않다면 이 필드는 제거해도 됩니다.
        )
