"""
Vanishing Point Detection
Usage: python3 main.py -imgpath P1030001.jpg
Author: Ruixi Lin and Xudong Shen
Date: 09-14-2019
Version: v0.1

References:
[1] J. M. Coughlan , A. L. Yuille. Manhattan world: orientation and outlier detection by Bayesian inference. Neural Computation. 2003
"""

#import argparse
import numpy as np
import cv2
from utils import EM_help_fucntions as emhelp
import scipy.stats
from tqdm import tqdm
#from time import time


# Parameters
vp_dir = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)    # (4,3)
print('vp_dir shape ', vp_dir.shape)
convergence = 1e-2

# sig and mu are for error probability computation P(phi_p|m_vp1, V) = N(theta_norm_vp1 - theta_grad | sig, mu)
sig = 0.5
mu = 0.0 # meaning that (theta_norm-theta_grad) is better to be close to 0 for a pixel in a vp group

# Define model assignment prior
# use [1]: assume that 40% of all edges are outliers/others and that x, y and z edges occur in roughly equal proportions. 
# P_m_prior = [0.2, 0.2, 0.2, 0.4]
# use the provided model assignment prior
P_m_prior = np.array([0.13, 0.24, 0.13, 0.5])


def downsample_image(img_path): 
    # Load a jpeg figure and convert it to grayscale
    image = cv2.imread(img_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(image_gray,cv2.CV_64F,1,0)#,ksize=3)
    sobely = cv2.Sobel(image_gray,cv2.CV_64F,0,1)#,ksize=3)
    # Calculate gradient magnitude
    gradmag = np.sqrt(sobelx**2+sobely**2)
    # Calculate gradient direction
    graddir = np.arctan2(sobely, sobelx)
    # Downsample the grayscale image
    Gdir, idx = emhelp.down_sample(gradmag, graddir)
    return Gdir, idx


def calculate_pixel_evidence(a, b, g, pixel, theta_grad):
    # return a list of evidence scores over all 4 models for a single pixel
    R = emhelp.angle2matrix(a,b,g)
    theta_norm = emhelp.vp2dir(K, R, pixel) # only calculate 3 evidence scores, the last one is give by uniform distribution
    err = theta_norm - theta_grad # [4,]
    err = emhelp.remove_polarity(err)
    prob = np.zeros(4)
    # normal distribution
    prob[:3] = scipy.stats.norm.pdf(err,mu,sig)
    prob[3] = 1/(2*np.pi) # the last prob is given by unifrom distribution
    # TODO I normalized prob to make it sums to 1
    prob = prob/prob.sum()
    return prob


def E_step(S):
    '''
    :param S : the Cayley-Gibbs-Rodrigu representation of camera rotation parameters
    :return: weights, pixel_assignments
    '''
    R = emhelp.vector2matrix(S)  # Note that the 'S' is just for optimization, it has to be converted to R during computation
    pixel_assignments = []
    scores = []
    # E-step: Assign each pixel to one of the VPs by finding argmax of log-posterior
    for pixel in pixel_indices: # compute log likelihood over all pixels, to avoid underflow
        theta_grad = Gdir_pixels[int(pixel[0])][int(pixel[1])]
        ori_pixel = np.array([pixel[1]*5+4, pixel[0]*5+4, 1])
        theta_norm = emhelp.vp2dir(K, R, ori_pixel)
        err = theta_norm - theta_grad # [4,]
        err = emhelp.remove_polarity(err)
        prob = np.zeros(4)
        # normal distribution
        prob[:3] = scipy.stats.norm.pdf(err,mu,sig)      # TODO or can write your own normal
        prob[3] = 1/(2*np.pi) # the last prob is given by unifrom distribution
        score = prob*P_m_prior
        #print('pixel score', pixel, ' ', score)
        scores.append(score) #appends the probability that a pixel u belongs to each of the 4 cases (vp models)
        pixel_assignments.append(np.argmax(score, axis=0))
    # normalize scores
    scores = np.array(scores)
    weights=scores/np.sum(scores,axis=1)[:, np.newaxis]
    # weights=scores/np.sum(scores,axis=1)[:, np.newaxis]
    return weights, pixel_assignments


def M_step(S0, K, pixel_indices, Gdir_pixels, w_pm):
    '''
    :param S0 : the camera rotation parameters from the previous step
    :param w_pm : weights from E-step
    :return: R_m : the optimized camera rotation matrix
    '''
    S_m = scipy.optimize.least_squares(error_fun, S0, args= (K, pixel_indices, Gdir_pixels, w_pm))
    return S_m

def error_fun(S, K, pixels, Gdir_pixels, weights):
    '''
    :param S : the variable we are going to optimize over
    :param w_pm : weights from E-step
    :return: error : the error we are going to minimize
    '''
    R = emhelp.vector2matrix(S) # Note that the 'S' is just for optimization, it has to be converted to R during computation
    sum_of_weighted_errors = 0.0
    for i in range(len(pixels)):
        pixel = pixels[i]
        theta_grad = Gdir_pixels[int(pixel[0])][int(pixel[1])]
        ori_pixel = np.array([pixel[1]*5+4, pixel[0]*5+4, 1])
        theta_norm = emhelp.vp2dir(K, R, ori_pixel)
        err = theta_norm - theta_grad # [3,]
        err = emhelp.remove_polarity(err)
        sum_of_weighted_errors += weights[i,:3].dot(err**2)
    return sum_of_weighted_errors


    



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-imgpath", help="Path of the original image.", default='P1030001.jpg')
    # parser.add_argument("-camparam", help="Path of the camera parameters.", default="cameraParameters.mat")
    # args = parser.parse_args()
    
    imgpath = 'P1080055.jpg'
    camparam = 'cameraParameters.mat'
    # Load and downsample image to use only the 1999 (TODO should be 2000) edge pixels (which are of the highest gradient magnitudes)
    #Gdir_pixels, pixel_indices_ori = downsample_image(args.imgpath)
    Gdir_pixels, pixel_indices_ori = downsample_image(imgpath)
    # Convert indices to homogeneous coordinates
    sh = pixel_indices_ori.shape
    pixel_indices = np.ones((sh[0],sh[1]+1))
    pixel_indices[:,:-1] = pixel_indices_ori
    # print('Gdir_pixels ', Gdir_pixels.shape)                                 # (96, 128)
    # print('pixel_indices ', pixel_indices.shape)                         # (1999, 3)
    
    # Initialize K
    #K = emhelp.cam_intrinsics(args.camparam)
    K = emhelp.cam_intrinsics(camparam)
    print('Initialized K: ', K.shape, ' ', K )        # (3, 3)

    # Initialize R, by adopting [1] a coarse-to-fine search over combinations of a, b, and g that maximizes towards a MAP objective
    # STEP 1: find optimal b around the y-axis, fixing a, g to 0
    print('========== Start step 1 ==============')
    #search_step =0
    max_score = -np.infty
    score = 0
    a_c = 0
    #a_optimal = 0
    #g_optimal = 0
    for a in tqdm(np.arange(-np.pi/3, np.pi/3, np.pi/45)):
        for u in pixel_indices:
            gdir_u = Gdir_pixels[int(u[0])][int(u[1])]
            # Represent downsampled pixel arrays in original image coordinates
            ori_u = np.array([u[1]*5 + 4, u[0]*5 + 4, 1]) # TODO: was a bug, fixed now
            evidence = calculate_pixel_evidence(a,0,0,ori_u,gdir_u)
            #print('evidence ', evidence)
            score += np.log(np.dot(evidence, P_m_prior)) 
            #score += np.log(evidence).sum() + np.log(P_m_prior).sum()
        if score > max_score:
            max_score = score
            a_c = a
        score = 0 # TODO: previously it was a bug, wrong intentation

    
    #STEP 2: Do a medium-scale search. 
    print('========== Start step 2 ==============')
    #search_step =0
    max_score = -np.infty
    score = 0
    a_m = a_c
    b_m = 0
    g_m = 0
    for a in tqdm([a_c-np.pi/90, a_c, a_c+np.pi/90]):
        for b in [-np.pi/36, 0, np.pi/36]:
            for g in [-np.pi/36, 0, np.pi/36]:
                for u in pixel_indices:
                    #print('Search step ', search_step)
                    gdir_u = Gdir_pixels[int(u[0])][int(u[1])]
                    ori_u = np.array([u[1]*5+4, u[0]*5+4, 1]) # TODO: was a bug, fixed now
                    evidence = calculate_pixel_evidence(a, b, g, ori_u, gdir_u)
                    score += np.log(np.dot(evidence, P_m_prior)) 
                    #score += np.log(evidence).sum() + np.log(P_m_prior).sum()
                    #print('score ', score)
                if score > max_score:
                    #print('in b ', b, 'score ', score)
                    max_score = score
                    a_m = a
                    b_m = b
                    g_m = g
                score = 0
                #search_step+=1
    
    # STEP 3: b_m is fixed. Do a fine-scale search
    print('========== Start step 3 ==============')
    #search_step =0
    a_f = a_m
    b_f = b_m
    g_f = g_m
    # b_old = b_optimal
    # a_old = a_optimal
    # g_old = g_optimal
    max_score = -np.infty
    score = 0
    for b in tqdm([b_m-np.pi/36, b_m-np.pi/72, b_m, b_m+np.pi/72, b_m+np.pi/36]):
        for g in [g_m-np.pi/36, g_m-np.pi/72, g_m, g_m+np.pi/72, g_m+np.pi/36]:
            for u in pixel_indices:
                #print('Search step ', search_step)
                gdir_u = Gdir_pixels[int(u[0])][int(u[1])]
                ori_u = np.array([u[1]*5+4, u[0]*5+4, 1])
                evidence = calculate_pixel_evidence(a_f, b, g, ori_u, gdir_u)
                score += np.log(np.dot(evidence, P_m_prior)) 
                #score += np.log(evidence).sum() + np.log(P_m_prior).sum()
            if score > max_score:
                #print('in b ', b, 'score ', score)
                max_score = score
                b_f = b
                g_f = g
            score = 0
            #search_step+=1

    # Compute R given the optimal MAP a_f, b_m, g_f
    R = emhelp.angle2matrix(b_f, a_f, g_f)
    R_init = R.copy()
    print('Initialized R ', R.shape, R)
    
#    #load result from coarse-to-fine search for debug
#    res_file = 'outputs1'
#    npzfile = np.load(res_file+'.npz')
#    R = npzfile['arr_4']
#    R_init = R.copy()

    #Iteratively find the VPs and optimal assignments
    print('Start EM...')
    num_iter = 20
    S = emhelp.matrix2vector(R)
    error = 1e7
    for i in range(num_iter):
        #t = time()
        w_pm, pixel_assignments = E_step(S)
        opt = M_step(S, K, pixel_indices, Gdir_pixels, w_pm)
        S = opt.x
        cur_error = opt.cost
        error_diff = np.absolute(error - cur_error)
        print('iter %d error %f and error difference %f: '%(i, cur_error, error_diff))
        if error_diff < convergence:
            print('Reached convergence at iter %d, error difference is %f: '%(i, error_diff))
            break
        if cur_error > error:
            break
        error = cur_error
        #print('iter {}: {}'.format(i, time()-t))
    
    print('final pixel_assignments ', len(pixel_assignments), pixel_assignments)
    R = emhelp.vector2matrix(S)
    vp_trans = K.dot(R).dot(vp_dir)
    vp_trans_init = K.dot(R_init).dot(vp_dir)
    print('Final R ', R)
    print('vp points ', vp_trans)

    # save to files
    res_file = 'outputs2'
    # if os.path.exists(res_file):
    #     os.remove(res_file)

    np.savez(res_file, pixel_indices_ori, pixel_assignments, vp_trans, vp_trans_init, R_init)

    '''
    # visualization, done in local machine not on server
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    res_file = 'outputs1'
    npzfile = np.load(res_file+'.npz')
    pixel_indices_ori = npzfile['arr_0']
    pixel_assignments = npzfile['arr_1']
    vp_trans = npzfile['arr_2']
    # Convert vp to non-homogeneous coordinates [x, y]
    vp_final = np.array([[p[0]/p[2], p[0]/p[1]] for p in vp_trans])
    pt_0 = []
    pt_1 = []
    pt_2 = []
    pt_3 = []
    for i in range(len(pixel_assignments)):
        pixel_coordinate = [pixel_indices_ori[i][1]*5, pixel_indices_ori[i][0]*5]
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


    # plot each group of points
    px, py = map(list, zip(*pt_0))
    plt.scatter(x=px, y=py, c='r', s=10, zorder=1)
    px, py = map(list, zip(*pt_1))
    plt.scatter(x=px, y=py, c='g', s=10, zorder=1)
    px, py = map(list, zip(*pt_2))
    plt.scatter(x=px, y=py, c='b', s=10, zorder=1)
    px, py = map(list, zip(*pt_3))
    plt.scatter(x=px, y=py, c='y', s=10, zorder=1)

    plt.show()
    '''

    




