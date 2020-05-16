import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

def thresh_callback(val):
    threshold = val

    canny_img = cv.Canny(img_gray, threshold, threshold*2)
    contours, _ = cv.findContours(canny_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None]*len(contours)
    bound_rect = [None]*len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        bound_rect[i] = cv.boundingRect(contours_poly[i])
    
    contours_poly_new, bound_rect_new, contours_poly_rep = rect_filter(contours_poly, bound_rect)
    drawing = np.zeros((canny_img.shape[0], canny_img.shape[1], 3), dtype = np.uint8)

    for i in range(len(contours_poly_new)):
        color = (rnd.randint(0, 256), rnd.randint(0, 256), rnd.randint(0, 256))
        cv.drawContours(drawing, contours_poly_new, i, (255, 255, 255))
        cv.rectangle(drawing, (bound_rect_new[i][0], bound_rect_new[i][1]), ((bound_rect_new[i][0] + bound_rect_new[i][2]), (bound_rect_new[i][1] + bound_rect_new[i][3])), (0, 0, 255), 2)

    for j in range(len(contours_poly_rep)):
        cv.drawContours(drawing, contours_poly_rep, j, (255, 255, 255))

    cv.imshow('Contours', drawing)

def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return 0
    return w*h

def union(a,b):
    area1 = a[2]*a[3]
    area2 = b[2]*b[3]
    area_union = area1 + area2 - intersection(a,b)
    return area_union  

def iou(a,b):
    union_area = union(a,b)
    inter_area = intersection(a,b)
    iou = inter_area/union_area
    return iou

def rect_filter(contours_poly, bound_rect):
    contours_poly_new = []
    bound_rect_new = []
    contours_poly_rep = []
    bound_rect_rep = []
    
    blank = np.zeros((650, 650, 3))

    for i in range(len(contours_poly)):
        cont1 = contours_poly[i]
        rect1 = bound_rect[i]
        for j in range(len(contours_poly)):
            if i==j:
                continue
            
            cont2 = contours_poly[j]
            rect2 = bound_rect[j]

            img1 = cv.drawContours(blank.copy(), contours_poly, i, 1)
            img2 = cv.drawContours(blank.copy(), contours_poly, j, 1)

            img = np.logical_and(img1, img2)

            if img.any() != 0:
                x1, y1, w1, h1 = rect1[0], rect1[1], rect1[2], rect1[3]
                x2, y2, w2, h2 = rect2[0], rect2[1], rect2[2], rect2[3]

                if w1*h1 > w2*h2:
                    contours_poly_rep.append(contours_poly[j])
                    bound_rect_rep.append(bound_rect[j])
    
    for k in range(len(contours_poly)):
        repeat = False
        for l in range(len(contours_poly_rep)):
            if bound_rect[k] == bound_rect_rep[l]:
                repeat = True

        if not repeat:
            contours_poly_new.append(contours_poly[k])
            bound_rect_new.append(bound_rect[k])

    return contours_poly_new, bound_rect_new, contours_poly_rep



#IMG_PATH = os.path.join(os.path.dirname(__file__), "int_img6.jpg")
IMG_PATH = '/Users/thomasmacbookair/Desktop/ITSP/int_img6.jpg'
kernel = np.ones((3,3), np.uint8)

img = cv.imread(IMG_PATH) 
img = cv.resize(img, (650, 650), interpolation = cv.INTER_CUBIC)
_, img_gray = cv.threshold(img, 50, 255, cv.THRESH_BINARY_INV)
img_gray = cv.dilate(img_gray, kernel, iterations = 10)
img_gray = cv.blur(img, (3,3))
cv.namedWindow('Source')
cv.imshow('Source', img)
cv.imshow('GRAY', img_gray)

def_thresh = 100
max_thresh = 255
cv.createTrackbar('Threshold', 'Source', def_thresh, max_thresh, thresh_callback)
thresh_callback(def_thresh)

cv.waitKey()
