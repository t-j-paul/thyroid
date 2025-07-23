"""
Extract frames from ultrasound cine-clips (video files).
Usage: python extract_frames.py --input_dir data/raw/ --output_dir data/processed/ [--frame_interval 5]
"""

import cv2
import os
import argparse
from tqdm import tqdm

def extract_and_save_frames(video_path, output_dir, frame_interval=5, resize_shape=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open {video_path}")
        return
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_resized = cv2.resize(frame, resize_shape)
            frame_norm = (frame_resized / 255.0).astype('float32')
            out_path = os.path.join(output_dir, f"{base_name}_frame{frame_count:04d}.png")
            cv2.imwrite(out_path, (frame_norm * 255).astype('uint8'))
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"[INFO] Extracted {saved_count} frames from {video_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--frame_interval', type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

    for video_file in tqdm(video_files):
        extract_and_save_frames(
            os.path.join(args.input_dir, video_file),
            args.output_dir,
            frame_interval=args.frame_interval
        )

if __name__ == '__main__':
    main()
