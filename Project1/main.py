"""
Vanishing Point Detection
Usage: python3 main.py -imgpath P1030001.jpg
Author: Ruixi Lin
Date: 09-14-2019
Version: v0.1

References:
[1] J. M. Coughlan , A. L. Yuille. Manhattan world: orientation and outlier detection by Bayesian inference. Neural Computation. 2003
"""

import argparse
import numpy as np
import cv2
from utils import EM_help_fucntions as emhelp

	
# Parameters
vp_dir = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)	# (4,3)
print('vp_dir shape ', vp_dir.shape)

# sig and mu are for error probability computation P(phi_p|m_vp1, V) = N(theta_norm_vp1 - theta_grad | sig, mu)
sig = 0.5
mu = 0.0

# Define model assignment prior
# use [1]: assume that 40% of all edges are outliers/others and that x, y and z edges occur in roughly equal proportions. 
# P_m_prior = [0.2, 0.2, 0.2, 0.4]
# use the provided model assignment prior
P_m_prior = [0.13, 0.24, 0.13, 0.5]





# TODO: with scaling returns 1970 edge pixels not 2000 pixels...but tuning the size will give even fewer pixels...
# TODO: without scaling returns 1999 pixels not 2000...
def downsample_image(img_path): 
	# Load a jpeg figure and convert it to grayscale
	image = cv2.imread(img_path)
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#cv2.imwrite('gray.jpg', image_gray)
	sobelx = cv2.Sobel(image_gray,cv2.CV_64F,1,0,ksize=3)
	sobely = cv2.Sobel(image_gray,cv2.CV_64F,0,1,ksize=3)
	# Calculate gradient magnitude TODO: remove scaling and try...
	gradmag = np.sqrt(sobelx**2+sobely**2)
	#scale_factor = np.max(gradmag)/255
	#gradmag = (gradmag/scale_factor).astype(np.uint8)
	# Calculate gradient direction
	graddir = np.arctan2(sobely, sobelx)
	abs_sobelx = np.absolute(sobelx)
	abs_sobely = np.absolute(sobely)
	graddir = np.arctan2(abs_sobely, abs_sobelx)
	# Downsample the grayscale image
	Gdir, idx = emhelp.down_sample(gradmag, graddir)
	return Gdir, idx

'''
def find_vp(K, R, pixels): 
	# Initialized VPs TODO: THIS RESULTS IN ONLY A SINGLE POINT!!! WRONG
	v_init = K*R*vp_dir
	convergence = 10e-4
	while err > convergence:
		# E-step: Assign each pixel to one of the VPs find argmax of phi_old
		max_score = 0
		phi_old = np.zeros([0.0,0.0,0.0])
		for i in range(4):	# 4 model settings according to P_m_prior
			for u in pixels: # compute log likelihood over all pixels, to avoid underflow
				likelihood += log(vp2dir(K, R[i], u)) # theta_edges, i.e, is the arctan# edge orientation
			score = likelihood*P_m_prior[i]
			if score > max_score:
				max_score = score
				phi_old = R[i] # TODO: ???

		# M-step
		# cgr representation 
		S = matrix2vector(R)
		
		# TODO: the following is for whole image, what about for a single pixel?
		abs_sobelx = np.absolute(sobelx)
		abs_sobely = np.absolute(sobely)
		graddir = np.arctan2(abs_sobely, abs_sobelx)
		
		err = theta_edges − graddir
		err = remove_polarity(err)

		# Convert the rotation matrix to the Cayley-Gibbs-Rodrigu representation 
		# to satisfy the orthogonal constraint
		R_vector = matrix2vector(R)

		# Use scipy.optimize.least_squares for optimization of Eq 3. 
		scipy.optimize.least_squares(func, R_vector)

	# Covert the Cayley-Gibbs-Rodrigu representation back into a rotation matrix. 
	# (convert the optimal S to R so as to generate results.)
	R_opt = vector2matrix(S_opt)
'''


def calculate_pixel_evidence(a, b, g, pixel, P_m_prior, vp_dir):
	# return a list of evidence scores over all 4 models for a single pixel
	# return the angle differences between the predicted normal direction and the gradient direction of a pixel.
	# x is in shape [3,] which represent the normal direction with respect to the three edge models.
	
	R = emhelp.angle2matrix(a,b,g)
	theta_norm = emhelp.vp2dir(K, R, pixel) # [3,]
	theta_grad = TODO
	err = theta_norm - theta_grad # [3,]
	err = emhelp.remove_polarity(err)
	# normal distribution 关于 x, 把err带入x，可算那个probability TODO
	prob = normal(err, sig, mu)
	return prob

	



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-imgpath", help="Path of the original image.", required=True)
	parser.add_argument("-camparam", help="Path of the camera parameters.", default="/home/ruixi/workspace/Project1/cameraParameters.mat")
	args = parser.parse_args()
	
	# Load and downsample image to use only the 1999 (TODO should be 2000) edge pixels (which are of the highest gradient magnitudes)
	Gdir_pixels, pixel_indices_ori = downsample_image(args.imgpath)
	# Convert indices to homogeneous coordinates
	sh = pixel_indices_ori.shape
	pixel_indices = np.zeros((sh[0],sh[1]+1))
	pixel_indices[:,:-1] = pixel_indices_ori
	print('Gdir_pixels ', Gdir_pixels.shape, ' ', Gdir_pixels) # TODO: what is it used for?	# (96, 128)
	print('pixel_indices ', pixel_indices.shape, ' ', pixel_indices) 						# (1999, 3)
	
	# Initialize K
	K = emhelp.cam_intrinsics(args.camparam)
	print('Initialized K: ', K.shape, ' ', K )		# (3, 3)

	# Initialize R, by adopting [1] a coarse-to-fine search over combinations of a, b, and g that maximizes towards a MAP objective
	# STEP 1: find optimal b around the y-axis, fixing a, g to 0
	max_score = 0
	score = 0
	b_c = -np.infty
	for b in np.arange(-np.pi/4, np.pi/4, np.pi/45):
		for u in pixel_indices:
			score += np.log(np.dot(calculate_pixel_evidence(b,0,0,u,P_m_prior), P_m_prior)) 
			# Note that log P(camera_params) is ignored as we do not assume any priors, so this term is omitted to be added to the score
		if score > max_score:
			max_score = score
			b_c = b
		score = 0
	
	# STEP 2: Do a medium-scale search. 
	b_m = -np.infty
	a_m = -np.infty
	g_m = -np.infty
	max_score = 0
	score = 0
	for b in [b_c-np.pi/90, 0, b_c+np.pi/90]:
		for a in [-np.pi/36, 0, np.pi/36]:
			for g in [-np.pi/36, 0, np.pi/36]:
				for u in pixel_indices:
					score += np.log(np.dot(calculate_pixel_evidence(b,a,g,u,P_m_prior), P_m_prior))
				if score > max_score:
					max_score = score
					b_m = b
					a_m = a
					g_m = g
				score = 0
	
	# STEP 3: b_m is fixed. Do a fine-scale search
	a_f = -np.infty
	g_f = -np.infty
	max_score = 0
	score = 0
	for a in [a_m-np.pi/36, a_m-np.pi/72, 0, a_m+np.pi/72, a_m+np.pi/36]:
		for g in [g_m-np.pi/36, g_m-np.pi/72, 0, g_m+np.pi/72, g_m+np.pi/36]:
			for u in pixel_indices:
				score += np.log(np.dot(calculate_pixel_evidence(b_m,a,g,u,P_m_prior), P_m_prior))
			if score > max_score:
				max_score = score
				a_f = a
				g_f = g
			score = 0

	# Compute R given the optimal MAP a_f, b_m, g_f
	R = emhelp.angle2matrix(a_f, b_m, g_f)

	# Initialize the VPs
	v_init = K.dot(R).dot(vp_dir)



	# Iteratively find the VPs and optimal assignments
	#find_vp()































