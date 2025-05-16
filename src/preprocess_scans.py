import cv2
import numpy as np
import os

def preprocess_scans(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    def preprocess_radar_scan(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.bilateralFilter(enhanced, 9, 75, 75)

    for filename in os.listdir(input_folder):
        output_path = os.path.join(output_folder, filename)

        if os.path.exists(output_path):
            continue

        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            processed_image = preprocess_radar_scan(image)

            cv2.imwrite(output_path, processed_image)
