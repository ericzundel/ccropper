#!env python
"""
Takes a JPG image from my SLR camera, looks for a circle in the middle and crops w/ a border.

Uses OpenCV to look for a circle in the middle of the frame and crops it.

This is image processing. Depends on some "magic" nubmers. I used several
online tutorials to get started. Some vestiges of that code remain, but I'm
sorry I don't remember where I got them all.
"""

import argparse
import pprint
import os
import sys

import cv2
import numpy as np

MIN_SIZE_PIXELS=500
DEFAULT_BLUR_RADIUS=50
DEFAULT_BORDER=20
DEFAULT_MIN_RADIUS=500
DEFAULT_MAX_RADIUS=2000
DEFAULT_INVERT_IMAGE=True # This code expects a black background. You might want to change if you are using a white background.
DEFAULT_PARAM1=200        # works well with high contrast circles
DEFAULT_PARAM2=.95        # Works well with nearly perfect circles
DEFAULT_CONTRAST=1.25     # Alpha,  Sample suggests 1.0. Value determined experiementally with high key BG
DEFAULT_BRIGHTNESS=100.0  # Beta, Sample suggests 0. value determined experimentally with kigh key BG

def debug(*arguments, **kwargs):
    """Conditionally print an debugging message to stderr and flush it."""

    global args

    if (args.debug):
        print(*arguments, **kwargs, file=sys.stderr, flush=True)

def warn(*arguments, **kwargs):
    """Unconditionally print a warning message to stderr and flush it."""

    print(*arguments, **kwargs, file=sys.stderr, flush=True)

def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(
        prog = 'ccropper',
        description = 'Finds a circle in the center of an image and crops it with a border');
    parser.add_argument('filename',
                        nargs='+',
                        help='File to process')
    parser.add_argument('-outprefix', '-o',
                        nargs=1,
                        help='Set the prefix for the output filename')
    parser.add_argument('-blur',
                        default=DEFAULT_BLUR_RADIUS,
                        help='Blur radius in pixels, overriding default')
    parser.add_argument('-border',
                        default=DEFAULT_BORDER,
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
                        help='value for param2, use 1.0 for perfect circles, less to allow more variance')
    parser.add_argument('-debug', '-d', action='store_true',
                        help='Enable debugging')
    parser.add_argument('-brightness',
                        default=float(DEFAULT_BRIGHTNESS),
                        help='Set brightness to floating point value 0 to 100')
    parser.add_argument('-contrast',
                        default=float(DEFAULT_CONTRAST),
                        help='Set contrast to floating point value 0 to 5.0')
    parser.add_argument('-noinvert',
                        default=False,
                        action='store_true',
                        help="Do not invert image. Defaults to invert for black background")

    args = parser.parse_args()
    if (args.debug):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(args)
        print(flush=True)

    return args

def crop_image(img, pt):
    """ Crop a circle out of an image given a tuple for the center

    img : Image
    The raw input image to crop

    pt : tuple(int, int, int)
    (center y coordinate, center x coordinate, radius)

    :rtype: Image
    processed image on success, None on failure
    """

    global args

    border_width = int(args.border)

    a, b, r = pt[0], pt[1], pt[2]
    detected_circle_width = max(2*r + 2*border_width, MIN_SIZE_PIXELS)
    image_width = min(min(img.shape[0], img.shape[1]), detected_circle_width)
    debug("width=%d height=%d detected_width=%d chose=%d" % (img.shape[0],
                                                             img.shape[1],
                                                             detected_circle_width,
                                                             image_width))

    x1 = a - int(image_width/2)
    y1 = b - int(image_width/2)
    x2 = x1 + image_width
    y2 = y1 + image_width

    if (args.debug):
        print("Center is " + str(a) + "," + str(b) + " radius "+str(r))
        print("Image width = ", str(image_width))
        print("Border width = ", str(border_width))
        print("x1=", str(x1), " y1=", str(y1),
              "x2=", str(x2), " y2=", str(y2))

    if (r > a or r > b):
        warn("Skipping circle that goes off the page with r=", r,
             " a=", a, " b=", b)
        return None

    if (x2 > img.shape[1] or y2 > img.shape[0]):
        warn("I think we detected the wrong circle. Keep trying.")
        return None

    circle_img = np.zeros((image_width, image_width, 3),
                              np.uint8)
    circle_img[0:image_width, 0:image_width] = \
            img[y1:y2, x1:x2]

    if (args.debug):
        #cv2.circle(img, (a, b), r, (0, 255, 0), 2)
        #cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
        #cv2.imshow("Detected circle region", img)
        #cv2.imshow("Copied region", circle_img)
        pass

    return circle_img

def adjust_brightness(img):
    """Perform a brightness/contrast conversion.  The quality of the image suffers."""
    global args
    return cv2.convertScaleAbs(img, alpha=float(args.contrast),
                               beta=float(args.brightness))

def try_detect_circles(img):
    """Try to find a circle in the passed image.

    Retries with some fallback values if the default one does not suceed.
    """
    global args

    #param1_vals = [int(args.param1), 100, 500, 50]
    param1_vals = [int(args.param1)]
    #param2_vals = [float(args.param2), 0.8]
    param2_vals = [float(args.param2)]

    tries = 1
    for param1_value in param1_vals:
        for param2_value in param2_vals:
            debug("Trying to detect cirlces with param1=",
                  param1_value, "param2=", param2_value)

            # Apply Hough transform on the blurred image.
            # The HOUGH_GRADIENT_ALT I found to sometimes be more reliable in my images.
            detected_circles = cv2.HoughCircles(img,
                                                cv2.HOUGH_GRADIENT,
                                                dp=1,
                                                minDist=500,
                                                param1 = param1_value,
                                                param2 =  param2_value,
                                                minRadius = int(args.minradius),
                                                maxRadius = int(args.maxradius))
            if (detected_circles is not None):
                print("Found circle with param1=", param1_value, " param2=",
                      param2_value, " after ", tries, "tries.")
                return detected_circles
            # Try another combination
            tries = tries + 1

def process_file(filename):
    """Handle a single file.

    Writes out the processed file as a .png using the basename of the input and
    a suffix on success.
    """

    global args

    # Read image.
    orig_img = cv2.imread(filename, cv2.IMREAD_COLOR)
    if (orig_img is None):
        warn("Couldn't read %s" % (filename))
        return

    orig_height = orig_img.shape[0]
    orig_width = orig_img.shape[1]
    if (args.debug):
        print("orig_width=", orig_width, " orig_height=", orig_height,
              flush=True)

    x1 = x2 = y1 = y2 = img = 0
    if False:
        # Crop out the middle of the image, most of the screen is wasted
        new_width = int(orig_width/2)
        new_height = int(orig_height/2)

        img = np.zeros((new_height, new_width, 3), np.uint8)

        #x1 = int(new_width/2)
        #x2 = x1 + new_width
        #y1 = int(new_height/2)
        #y2 = y1 + new_height
        #img[0:new_height,0:new_width] = orig_img[y1:y2,x1:x2]
    else:
        x2 = orig_width
        y2 = orig_height
        img = orig_img

    # Convert to grayscale.
    adjusted_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #if (args.debug):
    #    debug("After grayscale:")
    #    cv2.imshow("After grayscale", adjusted_img)
    #    cv2.waitKey(0)

    if(args.noinvert is False):
        adjusted_img = cv2.bitwise_not(adjusted_img)
        if (args.debug):
            debug("After invert:")
            cv2.imshow("After invert", cv2.resize(adjusted_img, (int(orig_width / 6), int(orig_height / 6))))
            cv2.waitKey(0)

    # Blur using 3 * 3 kernel.
    blur_radius = int(args.blur)
    adjusted_img = cv2.blur(adjusted_img, (blur_radius, blur_radius))
    if (args.debug):
        debug("After blur:")
        cv2.imshow("After blur", cv2.resize(adjusted_img, (int(orig_width / 6), int(orig_height / 6))))
        cv2.waitKey(0)

    #adjusted_img = adjust_brightness(adjusted_img)
    #if (args.debug):
    #    debug("After brightness:")
    #    cv2.imshow("After brightness", adjusted_img)
    #    cv2.waitKey(0)

    detected_circles = try_detect_circles(adjusted_img)
    if detected_circles is None:
        print("No circles found in ", filename)
        return

    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))

    print("found " + str(len(detected_circles[0])) + " circles")
    if (args.debug):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(detected_circles)

    for pt in detected_circles[0, :]:
        circle_img = crop_image(orig_img, pt)
        if (circle_img is None):
            continue
        output_filename = os.path.splitext(filename)[0] + "-cropped.png"
        print("Writing ", output_filename, flush=True)
        cv2.imwrite(output_filename, circle_img)
        # only print the first circle
        break

# Main line
args = parse_args()
print("Args are: ",args.filename, args.border, args.outprefix)

for filename in args.filename:
    print("Processing ", filename)
    process_file(filename)
    #try:
    #    process_file(filename)
    #except Exception as e:
    #    print("ERROR: Failed to process ", filename, " Exception:\n", e,
    #          file=sys.stderr, flush=True)

if args.debug:
    cv2.waitKey(0)

