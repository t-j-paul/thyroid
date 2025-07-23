"""
Evaluate trained model: accuracy, ROC-AUC, F1, confusion matrix, ROC plot.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, accuracy_score, roc_curve, classification_report
import matplotlib.pyplot as plt
from src.models.model import get_model

def evaluate():
    # TODO: Implement dataset and DataLoader
    val_loader = ...

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(pretrained=False).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
    model.eval()

    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            # images, labels = batch
            # outputs = model(images.to(device))
            # probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            # preds = np.argmax(outputs.cpu().numpy(), axis=1)
            # y_true.extend(labels.numpy())
            # y_pred.extend(preds)
            # y_probs.extend(probs)
            pass

    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('outputs/roc_curve.png')
    plt.close()

    # Confusion matrix plot
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    evaluate()
