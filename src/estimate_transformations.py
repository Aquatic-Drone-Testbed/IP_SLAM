import cv2
import numpy as np
import os

def estimate_transformations(input_folder, output_transformations_file, match_output_folder, overlay_output_folder, feature_detector_type="AKAZE", num_matches=55, nfeatures=5000):
    os.makedirs(match_output_folder, exist_ok=True)
    os.makedirs(overlay_output_folder, exist_ok=True)

    images = sorted([f for f in os.listdir(input_folder) if f.endswith(".png") or f.endswith(".jpg")])

    def create_feature_detector():
        if feature_detector_type == "SIFT":
            return cv2.SIFT_create()
        elif feature_detector_type == "ORB":
            return cv2.ORB_create(nfeatures)
        elif feature_detector_type == "AKAZE":
            return cv2.AKAZE_create()

    feature_detector = create_feature_detector()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) if feature_detector_type != "ORB" else cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    transformations = []

    for i in range(len(images) - 1):
        img1_path = os.path.join(input_folder, images[i])
        img2_path = os.path.join(input_folder, images[i + 1])

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        kp1, des1 = feature_detector.detectAndCompute(img1, None)
        kp2, des2 = feature_detector.detectAndCompute(img2, None)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        good_matches = matches[:num_matches]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        M, mask = cv2.estimateAffine2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=0.1)

        transformations.append(M)

        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        match_path = os.path.join(match_output_folder, f"match_{i:03d}.png")
        cv2.imwrite(match_path, match_img)

        h, w = img1.shape
        img2_transformed = cv2.warpAffine(img2, M, (w, h))

        overlay = cv2.addWeighted(img1, 0.5, img2_transformed, 0.5, 0)

        overlay_path = os.path.join(overlay_output_folder, f"overlay_{i:03d}.png")
        cv2.imwrite(overlay_path, overlay)

    np.save(output_transformations_file, transformations)
