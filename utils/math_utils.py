import math

def euclidean_distance(p1, p2):

    x1, y1 = p1
    x2, y2 = p2

    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)