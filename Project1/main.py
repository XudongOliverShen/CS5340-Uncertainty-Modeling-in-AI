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
import scipy.stats
from tqdm import tqdm

# TODO check if vp_trans needs to be converted to homogeneous coordinates

# Parameters
vp_dir = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)    # (4,3)
print('vp_dir shape ', vp_dir.shape)

# sig and mu are for error probability computation P(phi_p|m_vp1, V) = N(theta_norm_vp1 - theta_grad | sig, mu)
sig = 0.5
mu = 0.0 # meaning that (theta_norm-theta_grad) is better to be close to 0 for a pixel in a vp group

# Define model assignment prior
# use [1]: assume that 40% of all edges are outliers/others and that x, y and z edges occur in roughly equal proportions. 
# P_m_prior = [0.2, 0.2, 0.2, 0.4]
# use the provided model assignment prior
P_m_prior = np.array([0.13, 0.24, 0.13, 0.5])


# TODO: with scaling returns 1970 edge pixels not 2000 pixels...but tuning the size will give even fewer pixels...
# TODO: without scaling returns 1999 pixels not 2000...
def downsample_image(img_path): 
    # Load a jpeg figure and convert it to grayscale
    image = cv2.imread(img_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(image_gray,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(image_gray,cv2.CV_64F,0,1,ksize=3)
    # Calculate gradient magnitude
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


def calculate_pixel_evidence(a, b, g, pixel, theta_grad):
    # return a list of evidence scores over all 4 models for a single pixel
    R = emhelp.angle2matrix(a,b,g)
    theta_norm = []
    for i in range(3): # only calculate 3 evidence scores, the last one is give by uniform distribution
        vp_trans = K.dot(R).dot(vp_dir[i]) # computes vp location, loop through each vp_dir
        #vp_trans = vp_trans/vp_trans[-1]    # NO NEED?? represent it in homogeneous coordinates
        edges = np.cross(vp_trans, pixel)  # np.cross computes the vector perpendicular to both vp_trans and u, i.e., edges.dot(vp_trans)=0, edges.dot(u)=0
        theta = np.arctan2(edges[1], edges[0])
        theta_norm.append(theta)
    theta_norm = np.array(theta_norm) 
    err = theta_norm - theta_grad # [4,]
    err = emhelp.remove_polarity(err)
    prob = np.zeros(4)
    # normal distribution
    prob[:3] = scipy.stats.norm.pdf(err,mu,sig)      # TODO or can write your own normal
    prob[3] = 1/(2*np.pi) # the last prob is given by unifrom distribution
    return prob


def cost_func(R, K, pixels, Gdir_pixels, weights):
    # R is our variable
    sum_of_weighted_errors = 0
    for i in range(len(pixels)):
        theta_norm = []
        pixel = pixels[i]
        theta_grad = Gdir_pixels[int(pixel[0])][int(pixel[1])]
        for j in range(3): # only calculate 3 evidence scores, the last one is give by uniform distribution
            vp_trans = K.dot(R).dot(vp_dir[j]) # computes vp location, loop through each vp_dir
            #vp_trans = vp_trans/vp_trans[-1]    # NO NEED?? represent it in homogeneous coordinates
            edges = np.cross(vp_trans, pixel) 
            theta = np.arctan2(edges[1], edges[0])
            theta_norm.append(theta)
        theta_norm = np.array(theta_norm) 
        err = theta_norm - theta_grad # [4,]
        err = emhelp.remove_polarity(err)
        prob = np.zeros(4)
        # normal distribution
        prob[:3] = scipy.stats.norm.pdf(err,mu,sig)
        prob[3] = 1/(2*np.pi) # the last prob is given by unifrom distribution
        log_likelihood = np.log(prob)
        sum_of_weighted_errors += weights[i].dot(log_likelihood)
    return sum_of_weighted_errors



def find_vp(K, initialized_R, pixels, Gdir_pixels): 
    '''
    TODO
    '''    
    convergence = 10e-4
    R = initialized_R
    err_diff = np.infty
    ct = 0
    while err_diff > convergence:
        pixel_assignments = []
        scores = []
        print('step ', ct)
        # E-step: Assign each pixel to one of the VPs by finding argmax of log-posterior
        for pixel in pixel_indices: # compute log likelihood over all pixels, to avoid underflow
            theta_norm = []
            theta_grad = Gdir_pixels[int(pixel[0])][int(pixel[1])]
            for i in range(3): # only calculate 3 evidence scores, the last one is give by uniform distribution
                vp_trans = K.dot(R).dot(vp_dir[i]) # computes vp location, loop through each vp_dir
                #vp_trans = vp_trans/vp_trans[-1]    # NO NEED?? represent it in homogeneous coordinates
                edges = np.cross(vp_trans, pixel)  # np.cross computes the vector perpendicular to both vp_trans and u, i.e., edges.dot(vp_trans)=0, edges.dot(u)=0
                theta = np.arctan2(edges[1], edges[0])
                theta_norm.append(theta)
            theta_norm = np.array(theta_norm) 
            err = theta_norm - theta_grad # [4,]
            err = emhelp.remove_polarity(err)
            prob = np.zeros(4)
            # normal distribution
            prob[:3] = scipy.stats.norm.pdf(err,mu,sig)      # TODO or can write your own normal
            prob[3] = 1/(2*np.pi) # the last prob is given by unifrom distribution
            score = prob*P_m_prior
            print('pixel score', pixel, ' ', score)
            scores.append(score) #appends the probability that a pixel u belongs to each of the 4 cases (vp models)
            pixel_assignments.append(np.argmax(score, axis=0))
        # normalize scores
        scores = np.array(scores)
        weights=scores/np.sum(scores,axis=1)[:, np.newaxis]


        # M-step
        # Error is defined as a sum of weighted error, error is a function of R matrix
        error = cost_func(R, K, pixel_indices, Gdir_pixels, weights)

        # Convert the rotation matrix to the Cayley-Gibbs-Rodrigu representation 
        # to satisfy the orthogonal constraint
        S = matrix2vector(R)

        # Use scipy.optimize.least_squares for optimization of Eq 3. 
        res = scipy.optimize.least_squares(cost_func, S, args=(K, pixel_indices, Gdir_pixels, weights)) # todo:TODO error is defined as a sum of weighted error, error is a function of R matrix
        S_opt = res.x
        # Convert R vector back to R matrix
        R = vector2matrix(S_opt)
        cur_error = res.cost
        err_diff = np.abs(error - cur_error)
        print('S_opt ', S_opt)
        print('err_diff ', err_diff)

        ct+=1


    # Covert the Cayley-Gibbs-Rodrigu representation back into a rotation matrix. 
    # (convert the optimal S to R so as to generate results.)
    R_opt = vector2matrix(S_opt)
    print('sum_of_weighted_errors ')

    return R_opt, pixel_assignments




    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-imgpath", help="Path of the original image.", default='P1030001.jpg')
    parser.add_argument("-camparam", help="Path of the camera parameters.", default="cameraParameters.mat")
    args = parser.parse_args()
    
    # Load and downsample image to use only the 1999 (TODO should be 2000) edge pixels (which are of the highest gradient magnitudes)
    Gdir_pixels, pixel_indices_ori = downsample_image(args.imgpath)
    # Convert indices to homogeneous coordinates
    sh = pixel_indices_ori.shape
    pixel_indices = np.ones((sh[0],sh[1]+1))
    pixel_indices[:,:-1] = pixel_indices_ori
    print('Gdir_pixels ', Gdir_pixels.shape, ' ', Gdir_pixels)                                 # (96, 128)
    print('pixel_indices ', pixel_indices.shape, ' ', pixel_indices)                         # (1999, 3)
    
    # Initialize K
    K = emhelp.cam_intrinsics(args.camparam)
    print('Initialized K: ', K.shape, ' ', K )        # (3, 3)

    # Initialize R, by adopting [1] a coarse-to-fine search over combinations of a, b, and g that maximizes towards a MAP objective
    # STEP 1: find optimal b around the y-axis, fixing a, g to 0
    print('========== Start step 1 ==============')
    #search_step =0
    max_score = -np.infty
    score = 0
    b_c = 0
    #a_optimal = 0
    #g_optimal = 0
    for b in tqdm(np.arange(-np.pi/3, np.pi/3, np.pi/45)):
        for u in pixel_indices:
            gdir_u = Gdir_pixels[int(u[0])][int(u[1])]
            evidence = calculate_pixel_evidence(b,0,0,u,gdir_u)
            #print('evidence ', evidence)
            score += np.log(np.dot(evidence, P_m_prior)) 
            #score += np.log(evidence).sum() + np.log(P_m_prior).sum() # TODO double check
        if score > max_score:
            max_score = score
            b_c = b
            score = 0

    
    #STEP 2: Do a medium-scale search. 
    print('========== Start step 2 ==============')
    #search_step =0
    max_score = -np.infty
    score = 0
    b_m = b_c
    a_m = 0
    g_m = 0
    for b in tqdm([b_c-np.pi/90, 0, b_c+np.pi/90]):
        for a in [-np.pi/36, 0, np.pi/36]:
            for g in [-np.pi/36, 0, np.pi/36]:
                for u in pixel_indices:
                    #print('Search step ', search_step)
                    gdir_u = Gdir_pixels[int(u[0])][int(u[1])]
                    evidence = calculate_pixel_evidence(b,a,g,u,gdir_u)
                    score += np.log(np.dot(evidence, P_m_prior)) 
                    #score += np.log(evidence).sum() + np.log(P_m_prior).sum()
                    #print('score ', score)
                if score > max_score:
                    #print('in b ', b, 'score ', score)
                    max_score = score
                    b_m = b
                    a_m = a
                    g_m = g
                score = 0
                #search_step+=1
    
    # STEP 3: b_m is fixed. Do a fine-scale search
    print('========== Start step 3 ==============')
    #search_step =0
    a_f = 0
    b_f = b_m
    g_f = 0
    # b_old = b_optimal
    # a_old = a_optimal
    # g_old = g_optimal
    max_score = -np.infty
    score = 0
    for a in tqdm([a_m-np.pi/36, a_m-np.pi/72, 0, a_m+np.pi/72, a_m+np.pi/36]):
        for g in [g_m-np.pi/36, g_m-np.pi/72, 0, g_m+np.pi/72, g_m+np.pi/36]:
            for u in pixel_indices:
                #print('Search step ', search_step)
                gdir_u = Gdir_pixels[int(u[0])][int(u[1])]
                evidence = calculate_pixel_evidence(b_f,a,g,u,gdir_u)
                score += np.log(np.dot(evidence, P_m_prior)) 
                #score += np.log(evidence).sum() + np.log(P_m_prior).sum()
            if score > max_score:
                #print('in b ', b, 'score ', score)
                max_score = score
                a_f = a
                g_f = g
            score = 0
            #search_step+=1

    # Compute R given the optimal MAP a_f, b_m, g_f
    R = emhelp.angle2matrix(a_f, b_f, g_f)
    print('Initialized R ', R.shape, R)

    # Initialize the VPs (homogenenous? is this why it's 3 by 3?)
    # v_init = K.dot(R).dot(vp_dir)
    # print('v_init ', v_init.shape, v_init)


    #Iteratively find the VPs and optimal assignments
    print('Start EM...')
    R_opt, pixel_assignments = find_vp(K, R, pixel_indices, Gdir_pixels)




























