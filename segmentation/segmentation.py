import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

original_image = cv2.imread(os.path.join(os.path.dirname(__file__), "test3.jpg"))
image = original_image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dilate = cv2.dilate(thresh, kernel , iterations=12)

#cv2.imshow("thresh", thresh)
#cv2.imshow("dilate", dilate)

cnts = cv2.findContours(dilate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]


threshold_min_area = 0
threshold_max_area = 100000

area = []
box_coordinates=[]

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    area.append(cv2.contourArea(c))
    box_coordinates.append((x,y,w,h))
    

mean_area = sum(area)/len(area)

idx=0
for bcor in box_coordinates:
	if(area[idx]>mean_area/2):
		cv2.rectangle(original_image, (bcor[0],bcor[1]), (bcor[0]+bcor[2], bcor[1]+bcor[3]), (0,255,0),10)
	
	idx+=1

        
        

plt.imshow(original_image,'gray') 
plt.show()
cv2.waitKey(0)