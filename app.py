import gradio as gr
from PIL import Image
import torch, random, numpy as np
from config import APP_TITLE, MODEL_WEIGHTS, DEVICE, SEED, IMAGE_MAX_SIDE
from classifiers.image_classifier import ImageClassifier
from services.analyzer import Analyzer

# 재현성(선택)
def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
set_seed(SEED)

# 모델 구성
_classifier = ImageClassifier(MODEL_WEIGHTS, device=DEVICE)
_analyzer = Analyzer(_classifier)

def limit_size(img: Image.Image, max_side: int = IMAGE_MAX_SIDE) -> Image.Image:
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        return img.resize((int(w*scale), int(h*scale)))
    return img

def infer(image: Image.Image):
    if image is None:
        return None, "이미지를 업로드하세요.", ""
    image = limit_size(image)
    result = _analyzer.run(image)

    label = result.prediction.class_names[result.prediction.pred_idx]
    prob = result.prediction.probs[result.prediction.pred_idx] * 100
    summary = f"예측: {label} | 신뢰도: {prob:.2f}%"
    return result.overlay_image, summary, result.llm_report

with gr.Blocks(title=APP_TITLE, css="footer {visibility: hidden}") as demo:
    gr.Markdown(f"## {APP_TITLE}\n본 도구는 교육용 보조도구입니다. 의료진의 진단을 대체하지 않습니다.")
    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="X-ray 이미지 업로드")
            btn = gr.Button("분석 실행", variant="primary")
        with gr.Column(scale=1):
            img_out = gr.Image(type="pil", label="Grad-CAM Overlay")
            pred_text = gr.Markdown()
    with gr.Row():
        llm_out = gr.Markdown(label="LLM 설명 리포트")

    btn.click(fn=infer, inputs=[img_in], outputs=[img_out, pred_text, llm_out])
    img_in.change(fn=lambda x: (None, "", ""), inputs=img_in, outputs=[img_out, pred_text, llm_out])

if __name__ == "__main__":
    demo.queue(concurrency_count=2).launch()