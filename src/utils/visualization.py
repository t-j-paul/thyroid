"""
Visualization utilities: ROC, confusion matrix, Grad-CAM, class distribution.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def plot_roc(y_true, y_probs, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC={auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_class_distribution(y, save_path=None):
    plt.figure()
    sns.countplot(x=y)
    plt.title('Class Distribution')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def overlay_gradcam(image, cam, alpha=0.5):
    import cv2
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    return overlay
