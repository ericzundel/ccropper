#!env python
"""
Takes a JPG image from SLR camera, looks for circles and crops to a square.

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

DEFAULT_NUM_CIRCLES = 2
MIN_SIZE_PIXELS = 500
DEFAULT_BLUR_RADIUS = 50
DEFAULT_BORDER = 20
DEFAULT_MIN_RADIUS = 500
DEFAULT_MAX_RADIUS = 2000
# This code expects a black background. You might
# want to change if you are using a white background.
DEFAULT_INVERT_IMAGE = True
DEFAULT_PARAM1 = 200  # works well with high contrast circles
DEFAULT_PARAM2 = 0.95  # Works well with nearly perfect circles
# Alpha,  Sample suggests 1.0.
# Value determined experiementally with high key BG
DEFAULT_CONTRAST = 1.25
# Beta, Sample suggests 0. value determined experimentally with kigh key BG
DEFAULT_BRIGHTNESS = 100.0


def debug(*arguments, **kwargs):
    """Conditionally print an debugging message to stderr and flush it."""

    global args

    if args.debug:
        print(*arguments, **kwargs, file=sys.stderr, flush=True)


def warn(*arguments, **kwargs):
    """Unconditionally print a warning message to stderr and flush it."""

    print(*arguments, **kwargs, file=sys.stderr, flush=True)


def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(
        prog="ccropper",
        description="Finds circles in the center of image and crops a square",
    )
    parser.add_argument("filename", nargs="+", help="File to process")
    parser.add_argument(
        "-numcircles",
        "-n",
        default=DEFAULT_NUM_CIRCLES,
        help="Number of non-overlapping circles to detect",
    )
    parser.add_argument(
        "-outprefix", "-o", nargs=1, help="Set the prefix for the output filename"
    )
    parser.add_argument(
        "-blur",
        default=DEFAULT_BLUR_RADIUS,
        help="Blur radius in pixels, overriding default",
    )
    parser.add_argument(
        "-border",
        default=DEFAULT_BORDER,
        help="Number of pixels to use for border, overriding default",
    )
    parser.add_argument(
        "-minradius",
        default=DEFAULT_MIN_RADIUS,
        help="Smallest radius of circule to find in pixels",
    )
    parser.add_argument(
        "-maxradius",
        default=DEFAULT_MAX_RADIUS,
        help="Largest radius of circule to find in pixels",
    )
    parser.add_argument(
        "-param1", default=DEFAULT_PARAM1, help="Value for param1, overriding default"
    )
    parser.add_argument(
        "-param2",
        default=DEFAULT_PARAM2,
        help="use 1.0 for perfect circles, less to allow more variance",
    )
    parser.add_argument("-debug", "-d", action="store_true", help="Enable debugging")
    parser.add_argument(
        "-brightness",
        default=float(DEFAULT_BRIGHTNESS),
        help="Set brightness to floating point value 0 to 100",
    )
    parser.add_argument(
        "-contrast",
        default=float(DEFAULT_CONTRAST),
        help="Set contrast to floating point value 0 to 5.0",
    )
    parser.add_argument(
        "-noinvert",
        default=False,
        action="store_true",
        help="Do not invert image. Defaults to invert for black bg",
    )

    args = parser.parse_args()
    if args.debug:
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(args)
        print(flush=True)

    return args


def get_circle_bounds(pt):
    """Returns upper left and lower right coords from circle tuple x, y, radius

    pt: tuple (x, y, radius)

    returns ((x1, y1), (x2, y2))
    """

    return ((pt[0] - pt[2], pt[1] - pt[2]), (pt[0] + pt[2], pt[1] + pt[2]))


def bounds_intersect(b1, b2):
    """
    Checks if two bounding boxes intersect.

    Args:
      box1: A tuple of tuples representing the 1st bounding box: ((x1, y1), (x2, y2)).
      box2: A tuple of tuples representing the 2nd bounding box: ((x3, y3), (x4, y4)).

    Returns:
      True if the boxes intersect, False otherwise.
    """

    ((x1, y1), (x2, y2)) = b1
    rect1 = np.array(
        [[int(x1), int(y1)], [int(x1), int(y2)], [int(x2), int(y1)], [int(x2), int(y2)]]
    ).astype(np.float64)
    print(rect1)
    ((x3, y3), (x4, y4)) = b2
    rect2 = np.array(
        [[int(x3), int(y3)], [int(x3), int(y4)], [int(x4), int(y3)], [int(x4), int(y4)]]
    ).astype(np.float64)
    print(rect2)

    for point in rect2:
        if (point[0] > x3 and point[0] < x4) and (point[1] > y3 and point[1] < y4):
            return True


def add_bounds(b1, b2):
    """
    Combines two bounding boxes into a single bounding box. Courtesy ChatGPT

    Args:
        box1: A tuple (x1, y1, x2, y2) representing the first bounding box.
        box2: A tuple (x3, y3, x4, y4) representing the second bounding box.

    Returns:
        A tuple (x, y, w, h) representing the combined bounding box.
    """

    ((x1, y1), (x2, y2)) = b1
    ((x3, y3), (x4, y4)) = b2

    # Find the minimum x and y coordinates
    x_min = min(x1, x3)
    y_min = min(y1, y3)

    # Find the maximum x and y coordinates
    x_max = max(x2, x4)
    y_max = max(y2, y4)

    return ((x_min, y_min), (x_max, y_max))


def crop_image(img, bounds):
    """Crop a circle out of an image given a tuple for the center

    img : Image
    The raw input image to crop

    bounds : tuple(tuple(int, int), tuple(int, int_)
    ((x1, y1), (x2, y2))

    :rtype: Image
    processed image on success, None on failure
    """

    global args

    ((x1, y1), (x2, y2)) = bounds
    print(
        "Bounds to crop: x1=", str(x1), " y1=", str(y1), "x2=", str(x2), " y2=", str(y2)
    )

    border_width = int(args.border)
    width = x2 - x1
    height = y2 - y1

    # Turn area to crop into a square
    if width > height:
        extra = (width - height) / 2
        y1 = y1 - extra
        y2 = y2 + extra
        height = width
    else:
        extra = (height - width) / 2
        x1 = x1 - extra
        x2 = x2 + extra
        width = height

    bounds_width = max(width + (2 * border_width), MIN_SIZE_PIXELS)
    image_width = min(min(img.shape[0], img.shape[1]), bounds_width)
    debug(
        "width=%d height=%d detected_width=%d chose=%d"
        % (img.shape[0], img.shape[1], bounds_width, image_width)
    )

    x1 = int(x1 - border_width)
    y1 = int(y1 - border_width)
    x2 = int(x2 + border_width)
    y2 = int(y2 + border_width)

    if args.debug:
        print("Image width = ", str(image_width))
        print("Border width = ", str(border_width))
        print("x1=", str(x1), " y1=", str(y1), "x2=", str(x2), " y2=", str(y2))

    if x1 < 0 or y1 < 0:
        warn("Skipping box that goes off the page in a negative way")
        return None

    if x2 > img.shape[1] or y2 > img.shape[0]:
        warn("I think we detected the wrong circle. Keep trying.")
        return None

    circle_img = np.zeros((image_width, image_width, 3), np.uint8)
    circle_img[0:image_width, 0:image_width] = img[y1:y2, x1:x2]

    if args.debug:
        # cv2.circle(img, (a, b), r, (0, 255, 0), 2)
        # cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
        # cv2.imshow("Detected circle region", img)
        # cv2.imshow("Copied region", circle_img)
        pass

    return circle_img


def adjust_brightness(img):
    """Perform a brightness/contrast conversion.  The quality of the image suffers."""
    global args
    return cv2.convertScaleAbs(
        img, alpha=float(args.contrast), beta=float(args.brightness)
    )


def try_detect_circles(img):
    """Try to find a circle in the passed image.

    Retries with some fallback values if the default one does not suceed.
    """
    global args

    # param1_vals = [int(args.param1), 100, 500, 50]
    param1_vals = [int(args.param1)]
    # param2_vals = [float(args.param2), 0.8]
    param2_vals = [float(args.param2)]

    tries = 1
    for param1_value in param1_vals:
        for param2_value in param2_vals:
            debug(
                "Trying to detect circles with param1=",
                param1_value,
                "param2=",
                param2_value,
            )

            # Apply Hough transform on the blurred image.
            # The HOUGH_GRADIENT_ALT I found to sometimes be more reliable in my images.
            detected_circles = cv2.HoughCircles(
                img,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=500,
                param1=param1_value,
                param2=param2_value,
                minRadius=int(args.minradius),
                maxRadius=int(args.maxradius),
            )
            if detected_circles is not None:
                print(
                    "Found circles with param1=",
                    param1_value,
                    " param2=",
                    param2_value,
                    " after ",
                    tries,
                    "tries.",
                )
                return detected_circles
            # Try another combination
            tries = tries + 1


def process_file(filename):
    """Handle a single file.

    Writes out the processed file as a .png using the basename of the input and
    a suffix on success.
    """

    global args

    numcircles = int(args.numcircles)

    # Read image.
    orig_img = cv2.imread(filename, cv2.IMREAD_COLOR)
    if orig_img is None:
        warn("Couldn't read %s" % (filename))
        return

    orig_height = orig_img.shape[0]
    orig_width = orig_img.shape[1]

    # for debugging
    debug_dst = np.zeros((orig_width, orig_width), np.uint8)
    debug_color = (255, 0, 0)

    if args.debug:
        print("orig_width=", orig_width, " orig_height=", orig_height, flush=True)

    # x1 = y1 = 0
    # x2 = orig_width
    # y2 = orig_height

    # Convert to grayscale.
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    if args.noinvert is False:
        img = cv2.bitwise_not(img)
        if args.debug:
            debug("After invert:")
            cv2.imshow(
                "After invert",
                cv2.resize(img, (int(orig_width / 6), int(orig_height / 6))),
            )
            cv2.waitKey(0)

    # Blur using 3 * 3 kernel.
    blur_radius = int(args.blur)
    img = cv2.blur(img, (blur_radius, blur_radius))
    # if args.debug:
    #    debug("After blur:")
    #    cv2.imshow("After blur",
    #               cv2.resize(img, (int(orig_width/6), int(orig_height/6))))
    # cv2.waitKey(0)

    detected_circles = try_detect_circles(img)
    if detected_circles is None:
        print("No circles found in ", filename)
        return

    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))

    print("found " + str(len(detected_circles[0])) + " circles")
    if args.debug:
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(detected_circles)
    found_circles = []
    bounds = None
    for pt in detected_circles[0, :]:
        circle_bounds = get_circle_bounds(pt)

        if bounds is None:
            bounds = circle_bounds
            found_circles.append(bounds)
            if args.debug:
                print(
                    "Adding 1st (%d,%d),(%d,%d)"
                    % (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1])
                )
                cv2.circle(debug_dst, (pt[0], pt[1]), pt[2], debug_color)
        elif not (bounds_intersect(circle_bounds, bounds)):
            found_circles.append(circle_bounds)
            bounds = add_bounds(bounds, circle_bounds)
            if args.debug:
                print(
                    "Adding additional (%d,%d),(%d,%d)"
                    % (
                        circle_bounds[0][0],
                        circle_bounds[0][1],
                        circle_bounds[1][0],
                        circle_bounds[1][1],
                    )
                )
                cv2.circle(debug_dst, (pt[0], pt[1]), pt[2], debug_color)
        if len(found_circles) == numcircles:
            if args.debug:
                cv2.imshow(
                    "Found Circles",
                    cv2.resize(debug_dst, (int(orig_width / 6), int(orig_height / 6))),
                )
                cv2.waitKey(0)
            break
    if args.debug:
        print(
            "All bounds: (%d,%d),(%d,%d)"
            % (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1])
        )
    if len(found_circles) != numcircles:
        print(
            "Only found %d/%d of circles in file %s"
            % (len(found_circles), numcircles, filename)
        )
        return
    # Crop the images in found_circles
    circle_img = crop_image(orig_img, bounds)
    if circle_img is None:
        # Error is already printed
        return
    output_filename = os.path.splitext(filename)[0] + "-cropped.png"
    print("Writing ", output_filename, flush=True)
    cv2.imwrite(output_filename, circle_img)


# Main line
args = parse_args()
print("Args are: ", args.filename, args.border, args.outprefix)

for filename in args.filename:
    print("Processing ", filename)
    process_file(filename)

if args.debug:
    cv2.waitKey(0)
