import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

rnd.seed(1)
#fac = 100

def thresh_callback(val):
    global fac
    threshold = val

    canny_img = cv.Canny(img_gray, threshold, threshold*2)
    contours, _ = cv.findContours(canny_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None]*len(contours)
    bound_rect = [None]*len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        bound_rect[i] = cv.boundingRect(contours_poly[i])
    
    #h, w = canny_img.shape[:2]
    #bound_rect_new, contours_poly_new, contours_poly_rep = find_valid_rect(contours_poly, bound_rect, h, w, fac)
    contours_poly_new, bound_rect_new, contours_poly_rep = rect_filter(contours_poly, bound_rect)
    drawing = np.zeros((canny_img.shape[0], canny_img.shape[1], 3), dtype = np.uint8)

    for i in range(len(contours_poly_new)):
        color = (rnd.randint(0, 256), rnd.randint(0, 256), rnd.randint(0, 256))
        cv.drawContours(drawing, contours_poly_new, i, (255, 255, 255))
        cv.rectangle(drawing, (bound_rect_new[i][0], bound_rect_new[i][1]), ((bound_rect_new[i][0] + bound_rect_new[i][2]), (bound_rect_new[i][1] + bound_rect_new[i][3])), (0, 0, 255), 2)

    for j in range(len(contours_poly_rep)):
        cv.drawContours(drawing, contours_poly_rep, j, (255, 255, 255))

    cv.imshow('Contours', drawing)

'''
def fac_callback(pos):
    global fac
    fac = pos
    val = cv.getTrackbarPos('Threshold', 'Source')
    thresh_callback(val)
'''

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

'''
def find_valid_rect(poly_contours, bound_rect, h, w, fac1):
    contours_rep = []
    rect_rep = []
    contours_new = []
    rect_new = []
    for i in range(len(poly_contours)):
        #rect = bound_rect[i]
        for j in range(len(bound_rect)):
            if i == j:
                continue
            
            x1, y1, w1, h1 = bound_rect[i][0], bound_rect[i][1], bound_rect[i][2], bound_rect[i][3]
            x2, y2, w2, h2 = bound_rect[j][0], bound_rect[j][1], bound_rect[j][2], bound_rect[j][3]

            if intersection((x1,y1,w1,h1), (x2,y2,w2,h2)) == w2 * h2 and w2 * h2 < (w * h)/fac1:
                contours_rep.append(poly_contours[j])
                rect_rep.append(bound_rect[j])
    
    for k in range(len(poly_contours)):
        repeat = False
        for l in range(len(contours_rep)):
            if bound_rect[k] == rect_rep[l]:
                repeat = True

        if not repeat:
            contours_new.append(poly_contours[k])
            rect_new.append(bound_rect[k])

    return rect_new, contours_new, contours_rep                 
'''

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



IMG_PATH = '/Users/thomasmacbookair/Desktop/ITSP/int_img6.jpg'
kernel = np.ones((3,3), np.uint8)

img = cv.imread(IMG_PATH) 
#img = cv.resize(img, None, fx = 0.25, fy = 0.25)
img = cv.resize(img, (650, 650), interpolation = cv.INTER_CUBIC)
_, img_gray = cv.threshold(img, 50, 255, cv.THRESH_BINARY_INV)
img_gray = cv.dilate(img_gray, kernel, iterations = 10)
img_gray = cv.blur(img, (3,3))
cv.namedWindow('Source')
cv.imshow('Source', img)
cv.imshow('GRAY', img_gray)

def_thresh = 100
max_thresh = 255
#def_fac = 100
cv.createTrackbar('Threshold', 'Source', def_thresh, max_thresh, thresh_callback)
#cv.createTrackbar('Factor', 'Source', def_fac, 5000, fac_callback)
#fac_callback(def_fac)
thresh_callback(def_thresh)

cv.waitKey()
