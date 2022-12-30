"""
Takes a JPG image from my Kodak camera, gets value from 7 seg display

Uses Python Tesserect.

What I discovered is that I couldn't get this to work very reliably with
the default training files, even when tweaking the Tesserect config values.
I believe the next step is to train it on an LCD font. I downloaded a TTF
font that looks like my LCD (fonts-DSEG_v046) but got bogged down trying 
to do the training.   -EZA
"""
import argparse
import pprint
import os
import sys
import pdb

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

DEFAULT_CONTRAST=1.25   # Alpha,  Sample suggests 1.0. Value determined experiementally with high key BG 
DEFAULT_BRIGHTNESS=100.0  # Beta, Sample suggests 0. value determined experimentally with kigh key BG 

def parse_args():
    parser = argparse.ArgumentParser(
        prog = 'ledparser',
        description = 'Finds an image from an LED and tries to OCR it.');
    parser.add_argument('filename',
                        nargs='+',
                        help='File to process')
    parser.add_argument('-outprefix', '-o',
                        nargs=1,
                        help='Set the prefix for the output filename')
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

# Changes to b&w image with red mapped to white and everything else to black
def red_filter(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    out_img = np.zeros((h, w, 3), np.uint8)
    
    RED_THRESHOLD = 1.2    
    for y in range(0,h):
        for x in range(0,w):
            rgb = img[y,x]
            r = rgb[2]
            g = rgb[1]
            b = rgb[0]
            #print ("x=",x,"y=",y,"rgb=",rgb)
            if (r > 180 and (g == 0 or b == 0 or (float(r)/float(g) > RED_THRESHOLD
                                                  and float(r)/float(b) > RED_THRESHOLD))):
                #pdb.set_trace()
                out_img[y,x] = (0,0,0)
            else:
                out_img[y,x] = (255,255,255)

    return out_img

def preprocess(image):
    #image = resize(50, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #cv2.imshow("resized", image)
    #cv2.waitKey(0)
    image = red_filter(image)
    image=resize(5,image)
    cv2.imshow("red threshold", image)
    cv2.waitKey(0)
    return image

    #gray = get_grayscale(image)
    
    #thresh = thresholding(gray)
    #cv2.imshow("thresh", thresh);
    #cv2.waitKey(0);
    
    #opening_img = opening(gray)
    #if (opening_img is not None):
    #    cv2.imshow("opening", opening_img);
    #    cv2.waitKey(0);

    #canny_img = canny(gray)
    #if (canny_img is not None):
    #    cv2.imshow("canny", canny_img);
    #    cv2.waitKey(0);
    #return canny
    
def bounding_boxes(img):
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
    

args = parse_args()

for filename in args.filename:
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = preprocess(img)
    output_filename = os.path.splitext(filename)[0] + "-redfilter.png"
    cv2.imwrite(output_filename, img) 

    h, w, c = img.shape
    print ("Image is h=",h," w=",w, " h=", h)
    #d = pytesseract.image_to_data(img, Output.DICT)
    #print(d.keys())
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(d)
    result = pytesseract.image_to_string(img, lang="letsgodigital", config="--oem 3 --psm 8")
    print("Result is: ", result)
    print(flush=True)

