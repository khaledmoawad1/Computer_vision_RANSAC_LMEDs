# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:53:31 2022

@author: Dell
"""

import cv2
import numpy as np
from task_1 import kp1,kp2,matches,img1,img2,h1,h2,w1,w2
from itertools import combinations
import random

#%%
# Storing the points of matched features
common_points = []
for match in matches:
    # Get the matching keypoints for each of the images
    img1_idx  = match.queryIdx
    img2_idx  = match.trainIdx
    # Get the coordinates
    x1y1 = kp1[img1_idx].pt   
    x2y2 = kp2[img2_idx].pt
    feature = list(map(int, list(x1y1) + list(x2y2) + [match.distance]))
    common_points.append(feature)
common_points = np.array(common_points)

#%%


def DLT(common_points):
    A = []
    for x, y, u, v, d in common_points:
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.asarray(A)

    # Taking SVD
    U, S, V = np.linalg.svd(A)

    L = V[-1, :] / V[-1, -1]
    H = L.reshape(3, 3)
    return H

#%%
def RANSAC(common_points):
    n = 50  # number of best features to apply ransac
    threshold = 2  #thresholding error for calculating inliers
    max_iterations = 50  # number of times ransac to be performed
    
    
    inlier_num = 0          # To store number of inliers
    best_inliers = None     # To store inliers
    best_points = common_points[:n]      # Separating best points
    # To get list of 4 points out of best_points
    matched_pairs = list(combinations(best_points, 4))
    random.shuffle(matched_pairs)
   
    
    # Performing Ransac
    for matches in matched_pairs[:max_iterations]:

        H = DLT(matches)
        
        inliers = []
        count = 0

        # Caclulating number of inliers
        for feature in best_points:
            source = np.ones((3, 1))
            target = np.ones((3, 1))
            source[:2, 0] = feature[:2]
            target[:2, 0] = feature[2:4]
            
            # Transforming other features based on the current homography
            target_hat = H@source
            
            # Normalize to the last element
            target_hat = target_hat/target_hat[-1, 0]
            target = target/target[-1, 0]

            # Checking if inlier
            err = np.linalg.norm(target_hat-target)

            if err < threshold:
                count += 1
                inliers.append(feature)
                    
        # Maintaining best inliers
        if count > inlier_num:
            inlier_num = count
            best_inliers = inliers
    # Caclulating Homography based on best inliers
    best_H = DLT(best_inliers)
    
    ratio = inlier_num / n
    #threshold = (math.log(ratio))*n
    #print(ratio)
    if ratio > 0.8:
        print("The ratio is:" , ratio, "ACCEPTED")
    else:
        print("The ratio is:",ratio,"Try again")


    return best_H



#%%


H = RANSAC(common_points)
print("The final Homography is:")

print(H)
im_out = cv2.warpPerspective(img1, H, (w1,h1))
cv2.imshow("warped Image", im_out)



h_new = max (h1,h2)
w_new = w1 + w2

im_out_stitch = cv2.warpPerspective(img1, H, (w_new,h_new))
im_out_stitch[0:h2,0:w2] = img2

cv2.imshow("stitched Image", im_out_stitch)
cv2.waitKey(0)
cv2.destroyAllWindows()











