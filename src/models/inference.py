"""
Secure inference & Grad-CAM visualization for single frame prediction.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import hashlib
import argparse
from src.models.model import get_model

def verify_model(path, sha256_file):
    with open(path, "rb") as f:
        hash_val = hashlib.sha256(f.read()).hexdigest()
    with open(sha256_file, "r") as f:
        ref = f.read().strip()
    if hash_val != ref:
        raise RuntimeError("SHA256 mismatch for model checkpoint!")
    print("[INFO] Model file integrity verified.")

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return torch.tensor(img).unsqueeze(0)

def grad_cam(model, img_tensor, class_idx):
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fwd = model.base.layer4.register_forward_hook(forward_hook)
    handle_bwd = model.base.layer4.register_backward_hook(backward_hook)

    output = model(img_tensor)
    model.zero_grad()
    score = output[0, class_idx]
    score.backward()

    grads = gradients[0].detach().cpu().numpy()[0]
    fmap = features[0].detach().cpu().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.sum(fmap * weights[:, np.newaxis, np.newaxis], axis=0)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    handle_fwd.remove()
    handle_bwd.remove()
    return cam

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--checkpoint_path', required=True)
    args = parser.parse_args()

    sha256_file = args.checkpoint_path + '.sha256'
    verify_model(args.checkpoint_path, sha256_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()

    img_tensor = preprocess_image(args.image_path).to(device)
    output = model(img_tensor)
    probs = F.softmax(output, dim=1).cpu().numpy()[0]
    pred_class = np.argmax(probs)
    print(f"Prediction: {'Malignant' if pred_class else 'Benign'} (Confidence: {probs[pred_class]:.4f})")

    cam = grad_cam(model, img_tensor, pred_class)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    orig = cv2.imread(args.image_path)
    orig = cv2.resize(orig, (224, 224))
    overlay = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)
    cam_path = args.image_path.replace('.png', '_gradcam.png')
    cv2.imwrite(cam_path, overlay)
    print(f"Grad-CAM saved to {cam_path}")

if __name__ == "__main__":
    main()
