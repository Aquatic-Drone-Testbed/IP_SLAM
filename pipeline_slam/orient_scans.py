import cv2
import numpy as np
import os
from PIL import Image

def orient_scans(processed_folder, original_folder, aligned_folder, transformations_file, gif_processed_path, gif_original_path, gif_unoriented_path):

    os.makedirs(aligned_folder + "_proc", exist_ok=True)
    os.makedirs(aligned_folder + "_orig", exist_ok=True)
    os.makedirs(aligned_folder + "_unoriented", exist_ok=True)

    transformations = np.load(transformations_file, allow_pickle=True)

    processed_images = sorted([f for f in os.listdir(processed_folder) if f.endswith(".png") or f.endswith(".jpg")])
    original_images = sorted([f for f in os.listdir(original_folder) if f.endswith(".png") or f.endswith(".jpg")])

    reference_angle = 0
    aligned_processed = []
    aligned_original = []
    unoriented_images = []

    for i in range(len(processed_images) - 1):

        processed_img_path = os.path.join(processed_folder, processed_images[i])
        original_img_path = os.path.join(original_folder, original_images[i])

        processed_img = cv2.imread(processed_img_path, cv2.IMREAD_GRAYSCALE)
        original_img = cv2.imread(original_img_path, cv2.IMREAD_COLOR)

        # Save the unoriented image
        unoriented_path = os.path.join(aligned_folder + "_unoriented", f"unoriented_{i:03d}.png")
        cv2.imwrite(unoriented_path, original_img)
        unoriented_images.append(Image.open(unoriented_path))

        if i == 0:
            aligned_path_proc = os.path.join(aligned_folder + "_proc", f"aligned_proc_{i:03d}.png")
            aligned_path_orig = os.path.join(aligned_folder + "_orig", f"aligned_orig_{i:03d}.png")
            cv2.imwrite(aligned_path_proc, processed_img)
            cv2.imwrite(aligned_path_orig, original_img)

            aligned_processed.append(Image.open(aligned_path_proc))
            aligned_original.append(Image.open(aligned_path_orig))
            continue

        M = transformations[i - 1]
        angle = np.degrees(np.arctan2(M[0, 1], M[0, 0]))

        reference_angle -= angle

        h, w = processed_img.shape
        center = (w // 2, h // 2)
        R = cv2.getRotationMatrix2D(center, reference_angle, 1.0)

        aligned_proc_img = cv2.warpAffine(processed_img, R, (w, h))
        aligned_orig_img = cv2.warpAffine(original_img, R, (w, h))

        aligned_path_proc = os.path.join(aligned_folder + "_proc", f"aligned_proc_{i:03d}.png")
        aligned_path_orig = os.path.join(aligned_folder + "_orig", f"aligned_orig_{i:03d}.png")
        cv2.imwrite(aligned_path_proc, aligned_proc_img)
        cv2.imwrite(aligned_path_orig, aligned_orig_img)

        aligned_processed.append(Image.open(aligned_path_proc))
        aligned_original.append(Image.open(aligned_path_orig))
 
    if aligned_processed:
        aligned_processed[0].convert("RGB").save(
            gif_processed_path,
            save_all=True,
            append_images=[img.convert("RGB") for img in aligned_processed[1:]],
            duration=100,
            loop=0,
            transparency=None,
            disposal=2
        )

    if aligned_original:
        aligned_original[0].convert("RGB").save(
            gif_original_path,
            save_all=True,
            append_images=[img.convert("RGB") for img in aligned_original[1:]],
            duration=100,
            loop=0,
            transparency=None,
            disposal=2
        )

    if unoriented_images:
        unoriented_images[0].save(
            gif_unoriented_path,
            save_all=True,
            append_images=unoriented_images[1:],
            duration=100,
            loop=0
        )
