# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 15:14:00 2022

@author: Dell
"""

import numpy as np
import cv2
#%%
#Read and Resize images
img1 = cv2.imread('fishbowl-00.png')
img2 = cv2.imread('fishbowl-01.png')

h1 = img1.shape[0]
w1 = img1.shape[1]

# #new size
# new_h1 = int(h1/1.5)
# new_w1 = int(w1/1.5)

#resize
# img1 = cv2.resize(img1,(new_w1,new_h1))

h2 = img2.shape[0]
w2 = img2.shape[1]

# #new size
# new_h2 = int(h2/1.5)
# new_w2 = int(w2/1.5)

#resize
# img2 = cv2.resize(img2,(new_w2,new_h2))

#%%

# Using ORB detector

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)


#draw only keypoints
f1 = cv2.drawKeypoints(img1, kp1, None, color=(0,0,255), flags=0)
f2 = cv2.drawKeypoints(img2, kp2, None, color=(0,0,255), flags=0)



#%%
# Matching using Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = True)
matches = bf.match(des1, des2)
#print(len(matches))     

# for m in matches:
#     print(m.distance)


matches = sorted(matches, key = lambda x:x.distance)  # sort it by distance
matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2) 


cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('detected_features1', f1)
cv2.imshow('detected_features2', f2)
cv2.imshow( 'matching_result', matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows() 





