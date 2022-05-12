# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 12:55:43 2022

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
def LMedS(common_points):
    n = 20  # number of best features to apply ransac
    
    
    best_points = common_points[:n]      # Separating best points
    # To get list of 4 points out of best_points
    matched_pairs = list(combinations(best_points, 4))
    # shuffling them
    random.shuffle(matched_pairs)
    # Applying the algorithm LMedS
    
    medians_total = []    # array of medians of all possible matches to get the best of them
    for i in range (len(matched_pairs)):
        matches = matched_pairs[i]
        H = DLT(matches)
        
       
        err = []
        
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
    
            # calculate error (x) for each feature in best point
            x = np.linalg.norm(target_hat-target)
            err.append(x)
            
        err = np.array(err) #array of errors
        med = np.median(err)   # median of errors in matched_pairs[i]
        medians_total.append(med)  # create a total median matrix to store all medians all over matched_pairs
        
        lms = min(medians_total)   # finding least meadian 
        lms_ind = medians_total.index(lms)  #find the index of least median
    # best matches are whwer the lms locate    
    best_matches = matched_pairs[lms_ind]  
    best_H = DLT(best_matches) # recalculate H using DLT
    #print(best_H)
    return best_H
    
    

        
    

    
    #%%
    
H = LMedS(common_points)
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
