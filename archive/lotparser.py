"""
Takes a JPG image from my Kodak camera, gets value from 7 seg display

Uses Python Tesserect
Works great with a high key (white) background and flash.

"""
import argparse
import pprint
import os
import sys

import cv2
import numpy as np
import pytesseract

DEFAULT_BORDER=20 
DEFAULT_MIN_RADIUS=100
DEFAULT_MAX_RADIUS=1000
#DEFAULT_PARAM1=1.2     # from sample code
DEFAULT_PARAM1=200      
#DEFAULT_PARAM2=100     # from sample code
DEFAULT_PARAM2=75       # from sample code
DEFAULT_CONTRAST=1.25   # Alpha,  Sample suggests 1.0. Value determined experiementally with high key BG
DEFAULT_BRIGHTNESS=100.0  # Beta, Sample suggests 0. value determined experimentally with kigh key BG 

def parse_args():
    parser = argparse.ArgumentParser(
        prog = 'circcropper',
        description = 'Finds a circle in the center of an image and crops it with a border');
    parser.add_argument('filename',
                        nargs='+',
                        help='File to process')
    parser.add_argument('-outprefix', '-o',
                        nargs=1,
                        help='Set the prefix for the output filename')
    parser.add_argument('-border', '-b',
                        default=DEFAULT_BORDER,
                        nargs=1,
                        help='Number of pixels to use for border, overriding default')
    parser.add_argument('-minradius',
                        default=DEFAULT_MIN_RADIUS,
                        help='Smallest radius of circule to find in pixels')
    parser.add_argument('-maxradius',
                        default=DEFAULT_MAX_RADIUS,
                        help='Largest radius of circule to find in pixels')    
    parser.add_argument('-param1', 
                        default=DEFAULT_PARAM1,
                        help='Value for param1, overriding default')
    parser.add_argument('-param2', 
                        default=DEFAULT_PARAM2,
                        help='value for param2, use 100 for perfect circles, less to allow more variance')    
    parser.add_argument('-debug', '-d', action='store_true',
                        help='Enable debugging')
    parser.add_argument('-brightness',
                        default=float(DEFAULT_BRIGHTNESS),
                        help='Set brightness to floating point value 0 to 100')
    parser.add_argument('-contrast',
                        default=float(DEFAULT_CONTRAST),
                        help='Set contrast to floating point value 0 to 5.0')
    
    args = parser.parse_args()
    if (args.debug):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(args)
        print(flush=True)
        
    return args

def resize(scale_percent, img):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def preprocess(image):
    gray = get_grayscale(image)
    thresh = thresholding(gray)
    cv2.imshow("thresh", thresh);
    cv2.waitKey(0);
    
    opening_img = opening(gray)
    if (opening_img is not None):
        cv2.imshow("opening", opening_img);
        cv2.waitKey(0);

    canny_img = canny(gray)
    if (canny_img is not None):
        cv2.imshow("canny", canny_img);
        cv2.waitKey(0);

    return canny
    

args = parse_args()

for filename in args.filename:
    img = cv2.imread(filename)
    img = preprocess(img)

    h, w, c = img.shape
    print ("Image is h=",h," w=",w, " h=", h)

    boxes = pytesseract.image_to_boxes(img) 
    for b in boxes.splitlines():
        print ("Bounding Box is ", b)
        b = b.split(' ')

        # ignore empty bounding box
        if (int(b[3]) == 0 and int(b[4]) == 0):
            print ("Skipping empty")
            continue

        y1 = int(b[1])
        y2 = h - int(b[2])
        x1 = int(b[3])
        x2 = w - int(b[4])
                 
        box_img = cv2.rectangle(img, (y1,y2,x1,x2), (0, 255, 0), 2)

        resize_img = resize(25, box_img)
        cv2.imshow('Resized image', resize_img)
        cv2.waitKey(0)
