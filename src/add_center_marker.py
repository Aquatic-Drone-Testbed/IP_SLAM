import os
import cv2

def add_center_marker(input_folder, output_folder, scale_factor=1.0):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            
            if scale_factor != 1.0:
                img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            
            h, w, _ = img.shape
            center = (w // 2, h // 2)
            
            cv2.circle(img, center, 1, (0, 0, 255), -1)
            
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, img)

