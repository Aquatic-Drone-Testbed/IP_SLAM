import os
import cv2
import numpy as np
import imageio
from glob import glob

def stitch_map(aligned_folder, output_map_path, match_output_folder, gif_output_path, canvas_size=(500, 500), min_inliers=20):

    os.makedirs(match_output_folder, exist_ok=True)


    feature_threshhold = 20
    matches_threshhold = 10

    scan_paths = sorted(glob(os.path.join(aligned_folder, "*.png")))
    scans = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in scan_paths]
    scan_h, scan_w = scans[0].shape
    canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)

    akaze = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    poses = [np.eye(3)]
    gif_frames = []

    for i in range(1, len(scans)):
        ref_scan = scans[i - 1]
        curr_scan = scans[i]

        kp1, des1 = akaze.detectAndCompute(ref_scan, None)
        kp2, des2 = akaze.detectAndCompute(curr_scan, None)

        if des1 is None or des2 is None or len(kp1) < feature_threshhold or len(kp2) < feature_threshhold:
            poses.append(poses[-1])
            continue

        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) < matches_threshhold:
            poses.append(poses[-1])
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        matrix, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC)

        if matrix is None or inliers is None or np.sum(inliers) < min_inliers:
            poses.append(poses[-1])
            continue

        affine_matrix = np.vstack([matrix, [0, 0, 1]])
        composed_pose = poses[-1] @ affine_matrix
        poses.append(composed_pose)

        match_img = cv2.drawMatches(ref_scan, kp1, curr_scan, kp2, good_matches, None, flags=2)
        match_path = os.path.join(match_output_folder, f"match_{i:03d}.png")
        cv2.imwrite(match_path, match_img)

    for i, (scan_gray, pose) in enumerate(zip(scans, poses)):
        scan_color = cv2.cvtColor(scan_gray, cv2.COLOR_GRAY2BGR)
        cv2.circle(scan_color, (scan_w // 2, scan_h // 2), 1, (255, 0, 0), -1)

        warp_matrix = pose[:2, :]
        warped = cv2.warpAffine(scan_color, warp_matrix, canvas_size, flags=cv2.INTER_LINEAR)

        mask = warped.any(axis=2)
        canvas[mask] = np.maximum(canvas[mask], warped[mask])
        gif_frames.append(canvas.copy())

    cv2.imwrite(output_map_path, canvas)
    imageio.mimsave(gif_output_path, gif_frames, fps=3, loop=0)
