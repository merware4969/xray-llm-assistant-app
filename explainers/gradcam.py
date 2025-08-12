import torch
import numpy as np
import cv2
from typing import Optional
from PIL import Image

class GradCAMGenerator:
    def __init__(self, model, target_layer_name: str = "layer4"):
        self.model = model
        self.target_layer_name = target_layer_name

    def _get_target_layer(self):
        return getattr(self.model, self.target_layer_name)

    def generate(self, image_tensor: torch.Tensor, class_idx: Optional[int] = None):
        model = self.model
        model.eval()
        acts, grads = [], []

        def f_hook(module, inp, out):
            acts.append(out)

        def b_hook(module, gin, gout):
            grads.append(gout[0])

        layer = self._get_target_layer()
        h1 = layer.register_forward_hook(f_hook)
        h2 = layer.register_full_backward_hook(b_hook)

        out = model(image_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(out, dim=1).item())

        model.zero_grad(set_to_none=True)
        one_hot = torch.zeros_like(out)
        one_hot[0, class_idx] = 1.0
        out.backward(gradient=one_hot)

        A = acts[0].squeeze(0).detach().cpu().numpy()   # [C, H, W]
        G = grads[0].squeeze(0).detach().cpu().numpy()  # [C, H, W]
        weights = G.mean(axis=(1, 2))
        cam = np.maximum(np.tensordot(weights, A, axes=([0], [0])), 0)
        cam = cam / (cam.max() + 1e-8)

        h1.remove(); h2.remove()
        return cam

    @staticmethod
    def overlay_on_image(cam: np.ndarray, image: Image.Image, alpha: float = 0.5) -> Image.Image:
        cam_resized = cv2.resize(cam, image.size)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.array(image.convert("RGB")), 1 - alpha, heatmap, alpha, 0)
        return Image.fromarray(overlay)
