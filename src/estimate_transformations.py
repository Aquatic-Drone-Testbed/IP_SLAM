import cv2
import numpy as np
import os
import math
import cuda_akaze_py as akaze



def estimate_transformations(input_folder, output_transformations_file, match_output_folder, overlay_output_folder, angle_log_file, feature_detector_type="AKAZE", num_matches=55, nfeatures=5000):
    os.makedirs(match_output_folder, exist_ok=True)
    os.makedirs(overlay_output_folder, exist_ok=True)

    images = sorted([f for f in os.listdir(input_folder) if f.endswith(".png") or f.endswith(".jpg")])

    # def create_feature_detector():
    #     if feature_detector_type == "SIFT":
    #         return cv2.SIFT_create()
    #     elif feature_detector_type == "ORB":
    #         return cv2.ORB_create(nfeatures)
    #     elif feature_detector_type == "AKAZE":
    #         return cv2.AKAZE_create()

    # feature_detector = create_feature_detector()
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) if feature_detector_type != "ORB" else cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Set AKAZE options
    options = akaze.AKAZEOptions()
    # Set AKAZE matcher
    matcher = akaze.Matcher()


    transformations = []
    curr_angle = 0
    angle_list = []


    if os.path.exists(output_transformations_file):
        transformations = list(np.load(output_transformations_file, allow_pickle=True))

    #if os.path.exists(angle_log_file):
    #    with open(angle_log_file, "r") as f:
    #        angle_list = [float(line.split()[1]) for line in f.readlines()]


    for i in range(len(images) - 1):
        angle_deg = 0
        match_path = os.path.join(match_output_folder, f"match_{i:05d}.png")
        # print(match_path)
        if os.path.exists(match_path):
            continue

        img1_path = os.path.join(input_folder, images[i])
        img2_path = os.path.join(input_folder, images[i + 1])

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        img1_32 = np.float32(img1) / 255.0
        img2_32 = np.float32(img2) / 255.0

#        kp1, des1 = feature_detector.detectAndCompute(img1, None)
#        kp2, des2 = feature_detector.detectAndCompute(img2, None)

        # First image
        options.setWidth(img1.shape[1])
        options.setHeight(img1.shape[0])
        evo1 = akaze.AKAZE(options)
        evo1.Create_Nonlinear_Scale_Space(img1_32)
        des1, kp1 = evo1.Compute_Descriptors()

        # Second image
        options.setWidth(img2.shape[1])
        options.setHeight(img2.shape[0])
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

        good_matches = []

        try:
            # matches = bf.match(des1, des2)
            # Match descriptors
            matches = matcher.BFMatch(des1, des2)
            #print(matches)
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
            matches = [dm for pair in matches for dm in pair]
            matches = sorted(matches, key=lambda x: x.distance)
            # if(len(good_matches) < num_matches):
            #     return
            
            good_matches = matches[:num_matches]
        except cv2.error:
            pass

        # print(good_matches)
        if len(good_matches) == 0:
            print(len(good_matches) ) 
            angle_list.append("n")
            continue

        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # print(pts1.shape, pts2.shape)

        # M, mask = cv2.estimateAffine2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=0.1)
        M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=0.1)

        if M is not None:
            a, b = M[0, 0], M[0, 1]
            angle_rad = math.atan2(b, a)
            angle_deg = math.degrees(angle_rad)

        curr_angle += angle_deg - 90.0 #maybe
        angle_list.append(curr_angle)

        transformations.append(M)

        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imwrite(match_path, match_img)

        # h, w = img1.shape
        # img2_transformed = cv2.warpAffine(img2, M, (w, h))

        # overlay = cv2.addWeighted(img1, 0.5, img2_transformed, 0.5, 0)

        # overlay_path = os.path.join(overlay_output_folder, f"overlay_{i:05d}.png")
        # cv2.imwrite(overlay_path, overlay)

    np.save(output_transformations_file, transformations)

    with open(angle_log_file, "a") as f:
        for i, angle in enumerate(angle_list):
            if(angle == "n"):
                f.write(f"{i:05d} {angle}\n")
            else:
                f.write(f"{i:05d} {angle:.4f}\n")

##########

