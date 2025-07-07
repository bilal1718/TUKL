import os
import cv2
from pathlib import Path

dataset_root = "AVEC2014"
output_root = "avec2014_frames"
splits = ["train", "dev", "test"]
tasks = ["Freeform", "Northwind"]

def extract_frames(video_path, output_dir):
    print(f"Reading video from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        if frame_count % 50 == 0:
            print(f"Saved frame {frame_count} to {frame_filename}")
        frame_count += 1
    cap.release()
    print(f"Done extracting {frame_count} frames to {output_dir}\n")

for split in splits:
    for task in tasks:
        input_dir = os.path.join(dataset_root, split, task)
        output_dir = os.path.join(output_root, split, task)
        print(f"Processing split: '{split}' | task: '{task}'")
        print(f"Looking in input directory: {input_dir}")
        for video in os.listdir(input_dir):
            if video.endswith(".mp4"):
                video_path = os.path.join(input_dir, video)
                video_name = Path(video).stem
                save_dir = os.path.join(output_dir, video_name)
                print(f"\n Extracting from video: {video}")
                print(f"Saving frames to: {save_dir}")
                extract_frames(video_path, save_dir)

print("\n Frame extraction complete for all videos!")
