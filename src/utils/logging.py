"""
Logging utilities for user activity, predictions, model drift.
"""

import logging
import os
from datetime import datetime

def setup_logger(log_dir='logs/', name='activity', level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
    return logger

activity_logger = setup_logger(name='activity')
prediction_logger = setup_logger(name='predictions')
drift_logger = setup_logger(name='drift')

def log_user_action(user, action, details=""):
    activity_logger.info(f"User: {user} | Action: {action} | Details: {details}")

def log_prediction(user, image_id, prediction, confidence, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    prediction_logger.info(f"{timestamp} | User: {user} | Image: {image_id} | Prediction: {prediction} | Confidence: {confidence:.4f}")

def log_drift(metric, value, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    drift_logger.info(f"{timestamp} | Metric: {metric} | Value: {value}")
