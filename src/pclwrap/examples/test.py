from pcl import *
import numpy as np
import pcl
import random

cloud = pcl.PointCloud()

points = np.zeros((150, 4), dtype=np.float32)
RAND_MAX = 1.0
# Generate the data
for i in range(0, 75):
    # set Point Plane
    points[i][0] = 1024 * random.random () / (RAND_MAX + 1.0)
    points[i][1] = 1024 * random.random () / (RAND_MAX + 1.0)
    points[i][2] =  0.1 * random.random () / (RAND_MAX + 1.0)
    points[i][3] = 255 << 16 | 255 << 8 | 255

for i in range(75, 150):
    # set Point Randomize
    points[i][0] = 1024 * random.random () / (RAND_MAX + 1.0)
    points[i][1] = 1024 * random.random () / (RAND_MAX + 1.0)
    points[i][2] = 1024 * random.random () / (RAND_MAX + 1.0)
    points[i][3] = 255 << 16 | 255 << 8 | 255

# Set a few outliers
points[0][2] = 2.0;
points[3][2] = -2.0;
points[6][2] = 4.0;

cloud.from_array(points)
