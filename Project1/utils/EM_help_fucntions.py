import numpy as np
import cv2
import scipy.io
import matplotlib.pyplot as plt


vp_dir = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
P_m_prior = [0.13, 0.24, 0.13, 0.5]
sig = 0.5
mu = 0.0

def cam_intrinsics(path):
    '''
    :param path: path where you store the camera parameters
    :return: camera intrinsic matrix K
    '''
    cam_data = scipy.io.loadmat(path)
    f = cam_data['focal']
    pixelSize = cam_data['pixelSize']
    pp = cam_data['pp']
    K = np.array([[f[0][0] / pixelSize[0][0], 0, pp[0][0]], [0, f[0][0] / pixelSize[0][0], pp[0][1]], [0, 0, 1]],
                 dtype=np.float32)
    return K


def remove_polarity(x):
    '''
    :param x:  the angle differences between the predicted normal direction and the gradient direction of a pixel.
               x is in shape [3,] which represent the normal direction with respect to the three edge models.
    :return: the minimal value after add pi and -pi
    '''
    x = np.expand_dims(x, axis=0)
    new = np.abs(np.concatenate([x, x + np.pi, x - np.pi], axis= 0))
    diff = np.min(new, axis=0)
    return diff


def angle2matrix(a, b, g):
    '''

    :param a: the rotation angle around z axis
    :param b: the rotation angle around y axis
    :param g: the rotation angle around x axis
    :return: rotation matrix
    '''

    R = np.array([[np.cos(a)*np.cos(b), -np.sin(a)*np.cos(g)+np.cos(a)*np.sin(b)*np.sin(g),  np.sin(a)*np.sin(g)+np.cos(a)*np.sin(b)*np.cos(g), 0],
                  [np.sin(a)*np.cos(b),  np.cos(a)*np.cos(g)+np.sin(a)*np.sin(b)*np.sin(g), -np.cos(a)*np.sin(g)+np.sin(a)*np.sin(b)*np.cos(g), 0],
                  [-np.sin(b) ,         -np.cos(b)*np.sin(g),                                np.cos(b)*np.cos(g),                               0]], dtype=np.float32)

    return R

def vector2matrix(S):
    '''

    :param S: the Cayley-Gibbs-Rodrigu representation
    :return: rotation matrix R
    '''
    S = np.expand_dims(S, axis=1)
    den = 1 + np.dot(S.T, S)
    num = (1 - np.dot(S.T, S))*(np.eye(3)) + 2 * skew(S) + 2 * np.dot(S, S.T)
    R = num/den
    homo = np.zeros([3,1], dtype=np.float32)
    R = np.hstack([R, homo])
    return R

def skew(a):
    s = np.array([[0, -a[2, 0], a[1, 0]], [a[2, 0], 0, -a[0, 0]], [-a[1, 0], a[0, 0], 0]])
    return s

def matrix2quaternion(T):

    R = T[:3, :3]

    rotdiff = R - R.T

    r = np.zeros(3)
    r[0] = -rotdiff[1, 2]
    r[1] = rotdiff[0, 2]
    r[2] = -rotdiff[0, 1]
    sintheta = np.linalg.norm(r) / 2
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps)

    costheta = (np.trace(R) - 1) / 2

    theta = np.arctan2(sintheta, costheta)

    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = r0 * np.sin(theta / 2)
    return q


def matrix2vector(R):
    '''
    :param R: the camera rotation marix
    :return:  the Cayley-Gibbs-Rodrigu representation
    '''
    Q = matrix2quaternion(R)
    S = Q[1:]/Q[0]
    return S


def vp2dir(K, R, u):
    '''
    :param K: camera intrinsic matrix
    :param R: camera rotation matrix
    :param u: pixel location represented in homogeneous coordinate
    :return: the estimated normal direction for the edge that passes through pixel u
    '''
    vp_trans = K.dot(R).dot(vp_dir)
    edges = np.cross(vp_trans, u)  # np.cross computes the vector perpendicular to both vp_trans and u, i.e., edges.dot(vp_trans)=0, edges.dot(u)=0
    thetas_es = np.arctan2(edges[1], edges[0])
    return thetas_es

def down_sample(Gmag_, Gdir_):
    '''
    :param Gmag_: gradient magtitude of the original image
    :param Gir_: gradient direction of the original image
    :return: pixels we will use in the EM algorithm and the corresponding gradient direction
    '''
    Gmag = Gmag_[4::5, 4::5]
    Gdir = Gdir_[4::5, 4::5]
    threshold = np.sort(np.reshape(Gmag, [Gmag.shape[0]*Gmag.shape[1]]))
    idx = np.argwhere(Gmag > threshold[-2001])
    return Gdir, idx








