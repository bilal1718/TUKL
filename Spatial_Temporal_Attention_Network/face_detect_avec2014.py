import os
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

dataset_path = "avec2014/avec2014_frames"
output_path = "avec2014/dataset/avec14/image"
crop_size = 112
splits = ["train", "dev", "test"]
tasks = ["Freeform", "Northwind"]
sample_interval = 3

def get_face(image, face_detection):
    """Detect and crop face using MediaPipe."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = image.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        padding = int(0.2 * max(width, height))
        x, y = max(0, x - padding), max(0, y - padding)
        width, height = width + 2 * padding, height + 2 * padding
        face = image[y:y+height, x:x+width]
        if face.size > 0:
            face = cv2.resize(face, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
            return face
    return None

def process_video(video, split, task):
    input_dir = os.path.join(dataset_path, split, task)
    output_dir = os.path.join(output_path, f"{split}_{task}")
    video_path = os.path.join(input_dir, video)
    save_path = os.path.join(output_dir, f"{video}_aligned")
    os.makedirs(save_path, exist_ok=True)
    
    face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    img_list = sorted(os.listdir(video_path))[::sample_interval]
    for img in tqdm(img_list, desc=f"Processing {split}/{task}/{video}"):
        if img.endswith(".jpg"):
            image = cv2.imread(os.path.join(video_path, img))
            if image is None:
                print(f"[WARN] Could not load image: {img}")
                continue
            face = get_face(image, face_detection)
            if face is not None:
                cv2.imwrite(os.path.join(save_path, img), face)
            else:
                height, width, _ = image.shape
                a = int(height/2 - crop_size/2)
                b = int(height/2 + crop_size/2)
                c = int(width/2 - crop_size/2)
                d = int(width/2 + crop_size/2)
                image = image[a:b, c:d]
                if image.shape[0] == crop_size and image.shape[1] == crop_size:
                    cv2.imwrite(os.path.join(save_path, img), image)
                else:
                    print(f"[WARN] Invalid crop for {img}")
    face_detection.close()

def align_dlib():
    mp_face_detection = mp.solutions.face_detection
    video_tasks = []
    for split in splits:
        for task in tasks:
            input_dir = os.path.join(dataset_path, split, task)
            for video in os.listdir(input_dir):
                if os.path.isdir(os.path.join(input_dir, video)):
                    video_tasks.append((video, split, task))
    
    print(f"Processing {len(video_tasks)} videos with {cpu_count()} workers")
    with Pool(processes=cpu_count()) as pool:
        pool.starmap(partial(process_video), video_tasks)

if __name__ == '__main__':
    align_dlib()