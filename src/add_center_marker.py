import os
import cv2
import numpy as np

def add_center_marker(input_folder, output_folder, scale_factor=1.0):
    os.makedirs(output_folder, exist_ok=True)
    default_num_spokes = 250
    max_spoke_length = 256

    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith((".png", ".jpg"))])
    total_images = len(image_files)
    num_digits = 5  # Determine the number of digits for padding

    for idx, filename in enumerate(image_files, start=1):

        name, ext = os.path.splitext(filename)
        new_filename = f"radar_center_{idx:0{num_digits}d}{ext}"
        
        output_path = os.path.join(output_folder, new_filename)

        if os.path.exists(output_path):
            continue

        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        
        if scale_factor != 1.0:
            img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        
        h, w, _ = img.shape
        center = (w // 2, h // 2)
        
        cv2.circle(img, center, 1, (0, 0, 255), -1)
        
        # Generate new filename with leading zeros


        cv2.imwrite(output_path, img)

