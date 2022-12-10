# import the necessary packages
import argparse
import cv2
import numpy as np
from sklearn import preprocessing as p

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
# load the input image
image = cv2.imread(args["image"])
print('Data Type:',image.dtype)

# initialize OpenCV's static saliency spectral residual detector and
# compute the saliency map
'''
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
(success, saliencyMap) = saliency.computeSaliency(image)
saliencyMap = (saliencyMap * 255).astype("uint8")
cv2.imshow("Image", image)
cv2.imshow("Output", saliencyMap)
cv2.waitKey(0)
'''

# initialize OpenCV's static fine grained saliency detector and
# compute the saliency map
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(image)

# show the images
cv2.imshow("Original Image", image)
cv2.imshow("Saliency map", saliencyMap)

(B, G, R) = cv2.split(image)
    
salBlue = B * saliencyMap.astype(saliencyMap.dtype)
salGreen = G * saliencyMap.astype(saliencyMap.dtype)
salRed= R * saliencyMap.astype(saliencyMap.dtype)

salBlue = salBlue.astype("uint8")
salGreen = salGreen.astype("uint8")
salRed = salRed.astype("uint8")

reduction = np.ones((256,256))
inverse = reduction - saliencyMap

inverseBlue = B * inverse.astype(inverse.dtype)
inverseGreen = G * inverse.astype(inverse.dtype)
inverseRed = R * inverse.astype(inverse.dtype)

inverseBlue = inverseBlue.astype("uint8")
inverseGreen = inverseGreen.astype("uint8")
inverseRed = inverseRed.astype("uint8")

main = cv2.merge((salBlue, salGreen, salRed))
inverse = cv2.merge((inverseBlue, inverseGreen, inverseRed))

cv2.imshow("Image 1", main)
cv2.imshow("Image 2", inverse)
cv2.waitKey(0)

cv2.imwrite("imposed_map.jpg", inverse)

# min_max_scaler = p.MinMaxScaler()
# normalized_mask2 = min_max_scaler.fit_transform(saliencyMap)
# print(normalized_mask2)

# #normalized_mask2 = normalized_mask2 * 255
# cv2.imshow("mask", normalized_mask2)
# # normalized_mask2 = (saliencyMap-np.min(saliencyMap))/(np.max(saliencyMap)-np.min(saliencyMap))
# # print(normalized_mask2.dtype)
# fin_mask2 = cv2.merge((normalized_mask2, normalized_mask2, normalized_mask2))
# # print(fin_mask.shape)
# # print(image.shape)
# fin_image2 = cv2.multiply(image.astype(np.uint8), fin_mask2.astype(np.uint8))
# cv2.imshow("final image2", fin_image2)

# # if we would like a *binary* map that we could process for contours,
# # compute convex hull's, extract bounding boxes, etc., we can
# # additionally threshold the saliency map
# saliencyMap = np.clip(saliencyMap * 255, 0, 255)
# threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
# 	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# saliencyMap = saliencyMap.astype(np.uint8)
# cv2.imwrite("test_sm.jpg", saliencyMap)
# cv2.imwrite("thresh_map.jpg", threshMap)

# normalized_mask = (threshMap-np.min(threshMap))/(np.max(threshMap)-np.min(threshMap))
# print(normalized_mask.dtype)
# fin_mask = cv2.merge((normalized_mask, normalized_mask, normalized_mask))
# # print(fin_mask.shape)
# # print(image.shape)
# fin_image = cv2.multiply(image.astype(np.uint8), fin_mask.astype(np.uint8))
# cv2.imshow("final image", fin_image)

# # fin_image2 = cv2.bitwise_and(image.astype(np.uint8), fin_mask.astype(np.uint8), mask=None)
# # cv2.imshow("final image bitwise", fin_image2)
# cv2.waitKey(0)