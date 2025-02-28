import cv2
import numpy as np
import os

def detect_red_dot(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None

def stitch_map(input_folder, transformations_file, output_map_path, map_steps_path):
    images = sorted([f for f in os.listdir(input_folder) if f.endswith(".png") or f.endswith(".jpg")])
    transformations = np.load(transformations_file, allow_pickle=True)
    
    print(f"Starting stitching process with {len(images)} images.")
    base_img = cv2.imread(os.path.join(input_folder, images[0]), cv2.IMREAD_COLOR)
    h, w, _ = base_img.shape
    
    stitched_map = np.zeros((h * 3, w * 3, 3), dtype=np.uint8)
    center_x, center_y = w * 1.5, h * 1.5
    
    os.makedirs(map_steps_path, exist_ok=True)
    os.makedirs(map_steps_path + "/matches", exist_ok=True)
    os.makedirs(map_steps_path + "/steps", exist_ok=True)

    print("Initializing stitched map with the first image.")
    stitched_map[h:w+h, h:w+h] = base_img
    red_dot_positions = []
    red_dot = detect_red_dot(base_img)
    if red_dot:
        red_dot_positions.append((red_dot[0] + h, red_dot[1] + w))
    cv2.imwrite(map_steps_path + "/steps/step_0.png", stitched_map)
    
    feature_detector = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    for i in range(1, len(images)):
        img_path = os.path.join(input_folder, images[i])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        print(f"Processing image {i+1}/{len(images)}: {images[i]}")
        
        stitched_gray = cv2.cvtColor(stitched_map, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        kp1, des1 = feature_detector.detectAndCompute(stitched_gray, None)
        kp2, des2 = feature_detector.detectAndCompute(img_gray, None)
        
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:55]
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        M, mask = cv2.estimateAffine2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=0.1)
        
        print(f"Applying transformation matrix for image {i}.")
        warped_img = cv2.warpAffine(img, M, (stitched_map.shape[1], stitched_map.shape[0]), borderMode=cv2.BORDER_TRANSPARENT)
        
        mask_warped = (warped_img > 0).astype(np.uint8)
        stitched_map = cv2.addWeighted(stitched_map, 0.5, warped_img, 0.5, 0, dtype=cv2.CV_8U)
        
        # Convert non-grayscale pixels to black
        gray_stitched = cv2.cvtColor(stitched_map, cv2.COLOR_BGR2GRAY)
        stitched_map[np.any(stitched_map != np.stack([gray_stitched]*3, axis=-1), axis=-1)] = [0, 0, 0]
        
        red_dot = detect_red_dot(img)
        if red_dot:
            transformed_dot = np.dot(M[:, :2], np.array([red_dot[0], red_dot[1]])) + M[:, 2]
            transformed_dot = (int(transformed_dot[0]), int(transformed_dot[1]))
            red_dot_positions.append(transformed_dot)
        
        cv2.imwrite(map_steps_path + f"/steps/step_{i}.png", stitched_map)
        
        match_img = cv2.drawMatches(stitched_gray, kp1, img_gray, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        match_path = os.path.join(map_steps_path + "/matches", f"match_{i:03d}.png")
        cv2.imwrite(match_path, match_img)
    
    stitched_map = cv2.resize(stitched_map, (stitched_map.shape[1] * 2, stitched_map.shape[0] * 2))
    for i in range(len(red_dot_positions)):
        scaled_dot = (red_dot_positions[i][0] * 2, red_dot_positions[i][1] * 2)
        cv2.circle(stitched_map, scaled_dot, 1, (0, 0, 255), -1)
        if i > 0:
            prev_scaled_dot = (red_dot_positions[i-1][0] * 2, red_dot_positions[i-1][1] * 2)
            cv2.line(stitched_map, prev_scaled_dot, scaled_dot, (0, 255, 0), 1)
    
    print(f"Saving final stitched map to {output_map_path}.")
    cv2.imwrite(output_map_path, stitched_map)
