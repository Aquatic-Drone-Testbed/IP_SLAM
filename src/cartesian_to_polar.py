import os
import numpy as np
import cv2

def simulate_lidar_from_cartesian(cartesian_image, center, max_radius, num_spokes,  intensity_threshold=20, depth=10):
    polar_image = cv2.warpPolar(cartesian_image, dsize=(num_spokes, max_radius), center=center,
                                maxRadius=max_radius, flags=cv2.WARP_POLAR_LINEAR)
    
    mask_polar = np.zeros_like(polar_image, dtype=np.uint8)
    lidar_polar = np.zeros_like(polar_image)

    for angle in range(polar_image.shape[0]):
        for r in range(polar_image.shape[1]):
            if polar_image[angle, r] >= intensity_threshold and r > 10 and r < 50:
                end = min(r + depth, polar_image.shape[1])
                lidar_polar[angle, r:end] = polar_image[angle, r:end]
                mask_polar[angle, 0:end] = 255
                break
    return lidar_polar, mask_polar

def cartesian_to_polar(input_folder, polar_output_folder, lidar_polar_output_folder, lidar_cartesian_output_folder, max_radius=256, num_spokes=250):
    os.makedirs(polar_output_folder, exist_ok=True)
    os.makedirs(lidar_polar_output_folder, exist_ok=True)
    os.makedirs(lidar_cartesian_output_folder, exist_ok=True)
    os.makedirs(lidar_cartesian_output_folder + "_mask", exist_ok=True)


    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')])

    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        center = (img.shape[1] // 2, img.shape[0] // 2)

        polar_img = cv2.warpPolar(img, dsize=(num_spokes, max_radius), center=center,
                                  maxRadius=max_radius, flags=cv2.WARP_POLAR_LINEAR)

        polar_out_path = os.path.join(polar_output_folder, f"polar_{img_file}")
        cv2.imwrite(polar_out_path, polar_img)

        lidar_polar, polar_mask = simulate_lidar_from_cartesian(img, center=center, max_radius=max_radius, num_spokes=num_spokes)
        lidar_out_path = os.path.join(lidar_polar_output_folder, f"lidar_polar_{img_file}")
        cv2.imwrite(lidar_out_path, lidar_polar)

        mask_out_path = os.path.join(lidar_cartesian_output_folder + "_mask", f"lidar_mask_{img_file}")
        cv2.imwrite(mask_out_path, polar_mask)

        lidar_cartesian = cv2.warpPolar(lidar_polar, dsize=(img.shape[1], img.shape[0]), center=center,
                                        maxRadius=max_radius, flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)
        lidar_cartesian_mask = cv2.warpPolar(polar_mask, dsize=(img.shape[1], img.shape[0]), center=center,
                                             maxRadius=max_radius, flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)

        circular_mask = np.zeros_like(lidar_cartesian, dtype=np.uint8)
        cv2.circle(circular_mask, (max_radius, max_radius), max_radius - 10, (255), thickness=-1)

        radar_image = cv2.bitwise_and(lidar_cartesian, lidar_cartesian, mask=circular_mask)
        mask_image = cv2.bitwise_and(lidar_cartesian_mask, lidar_cartesian_mask, mask=circular_mask)

        lidar_cartesian_out_path = os.path.join(lidar_cartesian_output_folder, f"lidar_cartesian_{img_file}")
        cv2.imwrite(lidar_cartesian_out_path, radar_image)

        lidar_cartesian_mask_out_path = os.path.join(lidar_cartesian_output_folder + "_mask", f"lidar_cartesian_mask_{img_file}")
        cv2.imwrite(lidar_cartesian_mask_out_path, mask_image)