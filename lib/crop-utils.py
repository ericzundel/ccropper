
import numpy as np


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
    ((x3, y3), (x4, y4)) = b2
    rect1 = [[x1, y1], [x1, y2], [x2, y1], [x2, y2]]
    rect2 = [[x3, y3], [x3, y4], [x4, y3], [x4, y4]]

    for point in rect2:
        if (point[0] > x3 and point[0] < x4) and (point[1] > y3 and point[1] < y4):
            return True


def circles_interect(circle1, circle2):
    """
    Checks if two circles intersect (or one totally encloses the other).

    Args:
      circle1: A tuple representing the 1st circle: (center_x, center_y, radius).
      circle2: A tuple representing the 2nd bounding box: (center_x, center_y, radius)

    Returns:
      True if the circles intersect, False otherwise.
    """
   
    (x1, y1, r1) = circle1
    (x2, y2, r2) = circle2
    a = abs(x2 - x1)
    b = abs(y2 - y1)
    c = r1 + r2

    if (a*a) + (b*b) < (c*c):
        return True

    return False

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

