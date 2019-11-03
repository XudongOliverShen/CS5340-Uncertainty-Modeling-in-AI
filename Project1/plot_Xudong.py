# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 20:58:25 2019

@author: olive
"""

# visualization, done in local machine not on server
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from utils import EM_help_fucntions as emhelp

camparam = 'cameraParameters.mat'
vp_dir = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)    # (4,3)
K = emhelp.cam_intrinsics(camparam)


res_file = 'outputs2'
npzfile = np.load(res_file+'.npz')
pixel_indices_ori = npzfile['arr_0']
pixel_assignments = npzfile['arr_1']
vp_trans = npzfile['arr_2']
vp_trans_init = npzfile['arr_3']
# Convert vp to non-homogeneous coordinates [x, y]
vp_trans = vp_trans.transpose()
vp_final = np.array([[p[0]/p[2], p[1]/p[2]] for p in vp_trans])
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
        
vp_trans_init = vp_trans_init.transpose()
vp_final_init = np.array([[p[0]/p[2], p[1]/p[2]] for p in vp_trans_init])


# plot original image
im = mpimg.imread('P1030001.jpg')
implot = plt.imshow(im, zorder=0)

plt.scatter(x=vp_final_init[(0,2),0], y=vp_final_init[(0,2),1], marker='o', s=10, zorder=1)
plt.scatter(x=vp_final[(0,2),0], y=vp_final[(0,2),1], marker='+', s=10, zorder=1)
# plot each group of points
px, py = map(list, zip(*pt_0))
plt.scatter(x=px, y=py, c='r', s=0.1, zorder=1)
px, py = map(list, zip(*pt_1))
plt.scatter(x=px, y=py, c='g', s=0.1, zorder=1)
px, py = map(list, zip(*pt_2))
plt.scatter(x=px, y=py, c='b', s=0.1, zorder=1)
px, py = map(list, zip(*pt_3))
plt.scatter(x=px, y=py, c='y', s=0.1, zorder=1)

plt.savefig("output2.jpg",dpi=800)