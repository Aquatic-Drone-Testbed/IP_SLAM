import os
import cv2
import numpy as np
import imageio
from glob import glob

def stitch_map_pix_thresh(aligned_folder, output_map_path, grayscale_output_path, match_output_folder, gif_output_path, center_log_path, canvas_size=(500, 500), min_inliers=20, pixel_threshold=100):
    os.makedirs(match_output_folder, exist_ok=True)
    os.makedirs(f"scans/5_stitched_map_pt/images", exist_ok=True)
    os.makedirs(f"scans/5_stitched_map_pt/counter_npy", exist_ok=True)

    feature_threshhold = 20
    matches_threshhold = 10

    scan_paths = sorted(glob(os.path.join(aligned_folder, "*.png")))
    scans = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in scan_paths]
    scan_h, scan_w = scans[0].shape

    akaze = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    poses = [np.eye(3)]
    gif_frames = []
    center_positions = []

    heatmap_counter = np.zeros((canvas_size[0], canvas_size[1]), dtype=np.uint16)

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
        mask = (scan_gray >= pixel_threshold).astype(np.uint8) * 255
        warp_matrix = pose[:2, :]
        warped_mask = cv2.warpAffine(mask, warp_matrix, canvas_size, flags=cv2.INTER_NEAREST)

        heatmap_counter[warped_mask > 0] += 1

        center = np.array([[scan_w / 2, scan_h / 2]], dtype=np.float32)
        transformed_center = cv2.transform(np.array([center]), warp_matrix)[0][0]
        center_positions.append((int(transformed_center[0]), int(transformed_center[1])))

        heatmap_vis = heatmap_counter.astype(np.uint16)
        heatmap_vis_8bit = np.clip(heatmap_vis, 0, 255).astype(np.uint8)
        color_mapped = cv2.applyColorMap(heatmap_vis_8bit, cv2.COLORMAP_JET)
        color_mapped[heatmap_vis_8bit == 0] = 0

        gif_frames.append(color_mapped.copy())
        cv2.imwrite(f"scans/5_stitched_map_pt/images/stitched_{format(i, '03')}.png", color_mapped)
        np.save(f"scans/5_stitched_map_pt/counter_npy/heatmap_counter_{format(i, '03')}.npy", heatmap_counter)

    cv2.imwrite(output_map_path, color_mapped)
    cv2.imwrite(grayscale_output_path, heatmap_counter.astype(np.uint16))
    imageio.mimsave(gif_output_path, gif_frames, fps=3, loop=0)

    with open(center_log_path, "w") as f:
        for x, y in center_positions:
            f.write(f"{x} {y}\n")