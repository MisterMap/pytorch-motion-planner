import json
import os
from bitarray import bitarray
import numpy as np
import math as math
from matplotlib.patches import Polygon, Circle

def checker(map_cell, centr, max_dist, x1, y1):
    valid = True
    points = []
    for i in range(map_cell.shape[1]):
        for j in range(map_cell.shape[0]):
            if (i-centr[1])**2 + (j-centr[0])**2 <= (max_dist)**2 and map_cell[i,j] == 0:
                valid = False
                points.append([i,j])
    return valid, points

def max_dist(poly, start_point):
    coord = poly.get_xy()
    distance = []
    for i in coord: distance.append(math.dist(i, start_point))    
    return np.max(distance)

def nearest_point(suit_points, centr):
    distance = []
    for i in suit_points: distance.append(math.dist(i, centr))
    return suit_points[np.argmin(distance)]

def read_data(filename):
    file = open(filename, "r")
    data = json.load(file)
    file.close()
    shape = data['settings']['env']['collision']['robot_shape']
    run = data['runs'][0]
    points = np.array(shape)
    env = run['environment']
    w = env["width"]
    h = env["height"]
    map_data = np.array(list(bitarray(env["map"]))).reshape((h, w))
    map_data = 1. - map_data
    start_point = data['runs'][0]['environment']['start'][:2]
    return map_data, points, start_point