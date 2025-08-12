# streamlit_app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.metrics import auc
from streamlit_option_menu import option_menu
from glob import glob
import matplotlib.font_manager as fm

# -----------------------------
# 결정론적 설정 (Grad-CAM 일관성 유지)
# -----------------------------
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# 한글 폰트 설정 (KoPub + HTML 스타일 적용까지)
# -----------------------------
font_path = "./KoPubDotumMedium.ttf"
fontprop = fm.FontProperties(fname=font_path, size=14)
plt.rcParams['font.family'] = 'KoPubDotum'
plt.rcParams['axes.unicode_minus'] = False

with open(font_path, "rb") as f:
    base64_font = f.read().hex()

# HTML용 font-face 정의 추가
FONT_STYLE = f"""
<style>
@font-face {{
  font-family: 'KoPub';
  src: url(data:font/ttf;charset=utf-8;base64,{base64_font});
}}
body, div, h1, h2, h3, h4, h5, span, p {{
  font-family: 'KoPub', sans-serif;
}}
</style>
"""
st.markdown(FONT_STYLE, unsafe_allow_html=True)

# -----------------------------
# 모델 로딩
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('models/best_resnet34_addval.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

# -----------------------------
# 이미지 전처리
# -----------------------------
def preprocess_image(path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(path).convert('RGB')
    return transform(image).unsqueeze(0).to(device), image

def preprocess_uploaded(file):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(file).convert('RGB')
    return transform(image).unsqueeze(0).to(device), image

# -----------------------------
# 예측
# -----------------------------
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
    return pred, prob.squeeze().cpu().numpy()

# -----------------------------
# Grad-CAM
# -----------------------------
def generate_gradcam(model, image_tensor, class_idx):
    gradients = []
    activations = []

    def fw_hook(module, input, output):
        activations.append(output)

    def bw_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = model.layer4.register_forward_hook(fw_hook)
    h2 = model.layer4.register_full_backward_hook(bw_hook)

    output = model(image_tensor)
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0][class_idx] = 1
    output.backward(gradient=one_hot)

    acts = activations[0].squeeze(0).cpu()
    grads = gradients[0].squeeze(0).cpu()
    weights = grads.mean(dim=(1, 2))

    cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam.detach().numpy(), 0)
    cam /= np.max(cam)

    h1.remove()
    h2.remove()
    return cam


# -----------------------------
# Streamlit UI 구성
# -----------------------------
st.set_page_config(page_title="폐렴 분류 시각화", layout="centered")

with st.sidebar:
    selected = option_menu(
        menu_title="📌 메뉴",
        options=["📊 ROC 커브", "🔥 Grad-CAM 시각화", "🧪 추론"],
        icons=["bar-chart", "fire", "upload"],
        menu_icon="cast",
        default_index=0
    )

st.markdown("<h2 style='font-size:34px;'>ResNet34 기반 흉부 X-ray 폐렴 이진 분류</h2>", unsafe_allow_html=True)
model = load_model()
class_names = ['NORMAL', 'PNEUMONIA']

if selected == "📊 ROC 커브":
    st.subheader("📈 ROC Curve (Test Set 기준)")
    roc_image_path = "roc/roc_curve.png"

    if os.path.exists(roc_image_path):
        st.image(roc_image_path, caption="ROC Curve (Test Set)", use_column_width=True)
    else:
        st.warning("ROC 이미지 파일이 존재하지 않습니다. 먼저 ROC 커브를 생성하세요.")

elif selected == "🔥 Grad-CAM 시각화":
    st.header("🔥 Grad-CAM 시각화 (Test Set 이미지)")
    
    # test 하위 폴더 선택
    category = st.selectbox("클래스를 선택하세요", ["NORMAL", "PNEUMONIA"])
    folder_path = os.path.join("test", category)
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

    if image_files:
        img_name = st.selectbox("이미지를 선택하세요", image_files)
        path = os.path.join(folder_path, img_name)
        img_tensor, orig_img = preprocess_image(path)
        pred_class, _ = predict(model, img_tensor)
        cam = generate_gradcam(model, img_tensor, pred_class)
        cam_resized = cv2.resize(cam, orig_img.size)

        # 시각화
        fig, ax = plt.subplots()
        ax.imshow(orig_img)
        ax.imshow(cam_resized, cmap='jet', alpha=0.5, interpolation='bilinear')
        ax.set_title(f"Predicted Class: {class_names[pred_class]}")
        ax.axis('off')
        st.pyplot(fig)

        # 저장 버튼
        if st.button("🖼️ 이미지 저장하기"):
            save_dir = "saved_gradcam"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"gradcam_{category}_{img_name}")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            st.toast("✅ Grad-CAM 이미지가 저장되었습니다!", icon="🎉")
    else:
        st.warning(f"{category} 폴더에 이미지가 없습니다.")

elif selected == "🧪 추론":
    st.header("🧪 테스트 이미지 추론")
    uploaded = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    if uploaded:
        img_tensor, orig_img = preprocess_uploaded(uploaded)
        pred_class, probs = predict(model, img_tensor)
        st.image(orig_img, caption="업로드한 이미지", use_column_width=True)

        st.markdown(f"""
        <div style='padding: 1.5em; background-color: #f0f2f6; border-left: 6px solid #4b9cd3; font-size: 24px;'>
            <b>예측 결과:</b> {class_names[pred_class]}<br>
            <b>정확도:</b> {probs[pred_class] * 100:.2f}%
        </div>
        """, unsafe_allow_html=True)