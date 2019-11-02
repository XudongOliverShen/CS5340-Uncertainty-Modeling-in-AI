# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 20:58:25 2019

@author: olive
"""

# visualization, done in local machine not on server
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

res_file = 'outputs1'
npzfile = np.load(res_file+'.npz')
pixel_indices_ori = npzfile['arr_0']
pixel_assignments = npzfile['arr_1']
vp_trans = npzfile['arr_2']
# Convert vp to non-homogeneous coordinates [x, y]
vp_trans = vp_trans.transpose()
vp_final = np.array([[p[0]/p[2], p[0]/p[1]] for p in vp_trans])
pt_0 = []
pt_1 = []
pt_2 = []
pt_3 = []
for i in range(len(pixel_assignments)):
    pixel_coordinate = [pixel_indices_ori[i][1]*5+4, pixel_indices_ori[i][0]*5+4]
    if pixel_assignments[i] == 0:
        pt_0.append(pixel_coordinate)
    elif pixel_assignments[i] == 1:
        pt_1.append(pixel_coordinate)
    elif pixel_assignments[i] == 2:
        pt_2.append(pixel_coordinate)
    else:
        pt_3.append(pixel_coordinate)

# plot original image
im = mpimg.imread('P1030001.jpg')
implot = plt.imshow(im, zorder=0)

plt.scatter(x=vp_final[:,0], y=vp_final[:,1], marker='+', s=10, zorder=1)
# plot each group of points
px, py = map(list, zip(*pt_0))
plt.scatter(x=px, y=py, c='r', s=2, zorder=1)
px, py = map(list, zip(*pt_1))
plt.scatter(x=px, y=py, c='g', s=2, zorder=1)
px, py = map(list, zip(*pt_2))
plt.scatter(x=px, y=py, c='b', s=2, zorder=1)
px, py = map(list, zip(*pt_3))
plt.scatter(x=px, y=py, c='y', s=2, zorder=1)

plt.savefig("output1.jpg",dpi=400)