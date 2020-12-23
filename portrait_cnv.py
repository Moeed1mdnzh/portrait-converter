from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import numpy as np
import cv2
img_name = input("Please enter the path of your image --> ")
img = cv2.imread(img_name)
clone = img.copy()
img_f = img_as_float(img)
sgm = slic(img_f,n_segments=200,max_iter=100,enforce_connectivity=False,start_label=1)
for sv in np.unique(sgm):
    mask = np.zeros(img.shape[:2],np.uint8)
    mask[sgm == sv] = 255
    applied = cv2.bitwise_and(img,img,mask=mask)
    avg = np.average(img[sgm == sv],axis=0)
    img[sgm == sv] = tuple(list(map(int,avg)))
    cv2.imshow("mask",applied)
    cv2.waitKey(1)

blur = cv2.erode(img,np.ones((3,3),np.uint8))
cv2.imshow("clone",np.hstack([blur,clone]))
cv2.waitKey(0)


