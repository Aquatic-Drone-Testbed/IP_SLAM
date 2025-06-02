import os
import cv2
import numpy as np
import imageio
from glob import glob
import cuda_akaze_py as akaze

def test_stitch2(
    aligned_folder, mask_folder, output_map_path, output_frames, match_output_folder, gif_output_path, overlay_gif_output_path, center_log_path, canvas_size=(500, 500), min_inliers=20, pixel_threshold=100, visible_radius=60, prob_threshold=0.3, gif_output=False):
    os.makedirs(match_output_folder, exist_ok=True)
    os.makedirs(output_frames, exist_ok=True)

    feature_threshhold = 20
    matches_threshhold = 10
    frames_handled = 0

    scan_paths = sorted(glob(os.path.join(aligned_folder, "*.png")))
    mask_paths = sorted(glob(os.path.join(mask_folder, "*.png")))
    scans = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in scan_paths]
    masks = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in mask_paths]

    # # Sort paths by modification time (newest first)
    # scan_paths_sorted = sorted(scan_paths, key=os.path.getmtime, reverse=True)
    # mask_paths_sorted = sorted(mask_paths, key=os.path.getmtime, reverse=True)

    # # Define the number of files to read (at most 20, or fewer if there are less than 20)
    # read_count = min(2, len(scan_paths_sorted))
    # read_mask_count = min(2, len(mask_paths_sorted))

    # # Select the most recent 20 paths (or fewer if there are less than 20)
    # scan_paths_to_read = scan_paths_sorted[:read_count]
    # mask_paths_to_read = mask_paths_sorted[:read_mask_count]

    # # Read the images from the selected files
    # scans = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in scan_paths_to_read]
    # masks = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in mask_paths_to_read]


    if scans:
        scan_h, scan_w = scans[0].shape

    # akaze = cv2.AKAZE_create()
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Set AKAZE options
    options = akaze.AKAZEOptions()
    # Set AKAZE matcher
    matcher = akaze.Matcher()

    poses = [np.eye(3)]
    gif_frames = []
    overlay_gif_frames = []
    
    center_positions = []

    times_viewed_path = output_map_path.replace(".png", "_times_viewed.npy")
    times_counted_path = output_map_path.replace(".png", "_times_counted.npy")
    if os.path.exists(times_viewed_path) and os.path.exists(times_counted_path):
        times_viewed = np.load(times_viewed_path)
        times_counted = np.load(times_counted_path)
    else:
        times_viewed = np.zeros((canvas_size[0], canvas_size[1]), dtype=np.uint16)
        times_counted = np.zeros((canvas_size[0], canvas_size[1]), dtype=np.uint16)

    for i in range(1, len(scans)):
        ref_scan = scans[i - 1]
        curr_scan = scans[i]

        img1_32 = np.float32(ref_scan) / 255.0
        img2_32 = np.float32(curr_scan) / 255.0

        # kp1, des1 = akaze.detectAndCompute(ref_scan, None)
        # kp2, des2 = akaze.detectAndCompute(curr_scan, None)

        # First image
        options.setWidth(ref_scan.shape[1])
        options.setHeight(ref_scan.shape[0])
        evo1 = akaze.AKAZE(options)
        evo1.Create_Nonlinear_Scale_Space(img1_32)
        des1, kp1 = evo1.Compute_Descriptors()

        # Second image
        options.setWidth(curr_scan.shape[1])
        options.setHeight(curr_scan.shape[0])
        evo2 = akaze.AKAZE(options)
        evo2.Create_Nonlinear_Scale_Space(img2_32)
        des2, kp2 = evo2.Compute_Descriptors()

        kp1 = [
            cv2.KeyPoint(
                float(pt[0]), #x
                float(pt[1]), #y
                float(pt[2]), #size
                float(pt[3]), #angle
                float(pt[4]), #response
                int(pt[5]), #octave
                int(pt[6]) #class_id
            )
            for pt in kp1
        ]

        kp2 = [
            cv2.KeyPoint(
                float(pt[0]), #x
                float(pt[1]), #y
                float(pt[2]), #size
                float(pt[3]), #angle
                float(pt[4]), #response
                int(pt[5]), #octave
                int(pt[6]) #class_id
            )
            for pt in kp2
        ]

        if des1 is None or des2 is None or len(kp1) < feature_threshhold or len(kp2) < feature_threshhold:
            poses.append(poses[-1])
            continue

        #matches = bf.knnMatch(des1, des2, k=2)
        matches = matcher.BFMatch(des1, des2)
        dmatch_list = []
        for row in matches:  # matches = numpy array returned from bfmatch_
            # First match
            dmatch1 = cv2.DMatch(
                _queryIdx=int(row[0]),
                _trainIdx=int(row[1]),
                _imgIdx=int(row[2]),
                _distance=float(row[3])
            )
            # Second match
            dmatch2 = cv2.DMatch(
                _queryIdx=int(row[4]),
                _trainIdx=int(row[5]),
                _imgIdx=int(row[6]),
                _distance=float(row[7])
            )
            dmatch_list.append((dmatch1, dmatch2))
        matches = dmatch_list
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
        output_path = os.path.join(output_frames, f"stitched_{i:05d}.png")
        overlay_path = os.path.join(output_frames, f"stitched_overlay_{i:05d}.png")

        if os.path.exists(output_path) and os.path.exists(overlay_path):
            if(gif_output):
                bw_frame = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
                gif_frames.append(bw_frame.copy())

                overlay_frame = cv2.imread(overlay_path)
                overlay_gif_frames.append(overlay_frame.copy())
            continue

        frames_handled += 1

        mask = (scan_gray >= pixel_threshold).astype(np.uint8) * 255
        visible_mask = masks[i]

        warp_matrix = pose[:2, :]
        warped_visible_mask = cv2.warpAffine(visible_mask, warp_matrix, canvas_size, flags=cv2.INTER_NEAREST)
        warped_mask = cv2.warpAffine(mask, warp_matrix, canvas_size, flags=cv2.INTER_NEAREST)

        visible_pixels = warped_visible_mask > 0
        times_viewed[visible_pixels] += 1

        active_pixels = (warped_mask > 0) & visible_pixels
        times_counted[active_pixels] += 1

        probability_map = np.zeros_like(times_viewed, dtype=np.float32)
        np.divide(times_counted, times_viewed, out=probability_map, where=times_viewed > 0)

        bw_frame = (255 - (probability_map * 255)).astype(np.uint8)
        bw_frame[times_viewed == 0] = 255
        if gif_output: gif_frames.append(bw_frame.copy())
        cv2.imwrite(output_path, bw_frame)

        overlay_map = cv2.cvtColor(bw_frame, cv2.COLOR_GRAY2BGR)
        empty_region_mask = (times_viewed > 0) & (probability_map < 0.1)
        overlay_map[empty_region_mask] = [240, 220, 220]
        cv2.imwrite(overlay_path, overlay_map)
        if gif_output: overlay_gif_frames.append(overlay_map.copy())

        center = np.array([[scan_w / 2, scan_h / 2]], dtype=np.float32)
        transformed_center = cv2.transform(np.array([center]), warp_matrix)[0][0]
        center_positions.append((int(transformed_center[0]), int(transformed_center[1])))

    if frames_handled > 0:
        final_bw_prob_map = (255 - (probability_map * 255)).astype(np.uint8)
        final_bw_prob_map[times_viewed == 0] = 255
        # cv2.imwrite(output_map_path, final_bw_prob_map)

        overlay_map = cv2.cvtColor(final_bw_prob_map, cv2.COLOR_GRAY2BGR)
        empty_region_mask = (times_viewed > 0) & (probability_map < 0.1)
        overlay_map[empty_region_mask] = [240, 220, 220]
        # cv2.imwrite(output_map_path.replace(".png", "_overlayed.png"), overlay_map)

        if gif_output:
            imageio.mimsave(gif_output_path, gif_frames, fps=3, loop=0)
            imageio.mimsave(overlay_gif_output_path, overlay_gif_frames, fps=3, loop=0)

        with open(center_log_path, "a") as f:
            for x, y in center_positions:
                f.write(f"{x} {y}\n")

        np.save(output_map_path.replace(".png", "_probability_map.npy"), probability_map)
        np.save(times_viewed_path, times_viewed)
        np.save(times_counted_path, times_counted)

        frequency_heatmap = (times_viewed / np.max(times_viewed) * 255).astype(np.uint8)
        frequency_color_mapped = cv2.applyColorMap(frequency_heatmap, cv2.COLORMAP_JET)
        frequency_color_mapped[frequency_heatmap == 0] = 0
        # cv2.imwrite(output_map_path.replace(".png", "_frequency_map.png"), frequency_color_mapped)

        thresholded_map = (probability_map >= prob_threshold).astype(np.uint8) * 255
        thresholded_color_mapped = cv2.applyColorMap(thresholded_map, cv2.COLORMAP_JET)
        thresholded_color_mapped[thresholded_map == 0] = 0
        # cv2.imwrite(output_map_path.replace(".png", f"_thresholded_prob_map_{int(prob_threshold*100)}.png"), thresholded_color_mapped)
