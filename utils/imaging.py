from PIL import Image

def ensure_rgb_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.open(img).convert("RGB")