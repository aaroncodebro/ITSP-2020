import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

rnd.seed(1)

def thresh_callback(val):
    global fac
    threshold = val

    canny_img = cv.Canny(img_out, threshold, threshold*2)
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


IMG_PATH = '/Users/thomasmacbookair/Desktop/ITSP/test2.jpg'
kernel = np.ones((3,3), np.uint8)

img = cv.imread(IMG_PATH) 
img = cv.resize(img, (650, 650), interpolation = cv.INTER_CUBIC)

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, img_gray = cv.threshold(img_gray, 90, 255, cv.THRESH_BINARY_INV)
img_gray = cv.dilate(img_gray, kernel, iterations = 2)
#img_gray = cv.blur(img_gray, (3, 3))

h, w = img_gray.shape[:2]
mask = np.zeros((h+2, w+2), dtype = np.uint8)
img_ffill = img_gray.copy()
cv.floodFill(img_ffill, mask, (0, 0), (255, 255, 255));

img_ffill_inv = cv.bitwise_not(img_ffill)
img_out = img_ffill_inv | img_gray
img_out = cv.erode(img_out, kernel)
img_out = cv.dilate(img_out, kernel)

cv.imshow('Floodfill', img_out)
cv.imshow('GRAY', img_gray)

def_thresh = 100
thresh_callback(def_thresh)

cv.waitKey()
