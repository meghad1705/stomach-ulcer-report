import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load trained model ---
model = models.efficientnet_b0(pretrained=False)
num_ftrs = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(num_ftrs, 2))
model.load_state_dict(torch.load("ulcer_model.pth", map_location=device))
model.eval().to(device)

# --- Grad-CAM ---
def grad_cam(img_path, target_class=1):
    img = Image.open(img_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    inp = tf(img).unsqueeze(0).to(device)

    activations, grads = None, None
    def f_hook(m, i, o): nonlocal activations; activations = o
    def b_hook(m, gi, go): nonlocal grads; grads = go[0]
    h1 = model.features[-1].register_forward_hook(f_hook)
    h2 = model.features[-1].register_backward_hook(b_hook)

    out = model(inp)
    score = out[0, target_class]
    model.zero_grad()
    score.backward()

    weights = grads.mean(dim=(2,3), keepdim=True)
    cam = (weights * activations).sum(dim=1).squeeze().cpu().detach().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img.width, img.height))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(img), 0.5, heatmap, 0.5, 0)

    h1.remove(); h2.remove()
    cv2.imwrite("gradcam_result.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("✅ Grad-CAM saved as gradcam_result.jpg")

# Example:
# grad_cam("data/test/ulcer/001.jpg", target_class=1)
