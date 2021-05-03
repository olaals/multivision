# Some of these functions are taken from modern robotics with MIT license (see below). Additional functions are added, but assume every function is from modern robotics.

"""
MIT License

Copyright (c) 2019 NxRLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import cv2

def testtest():
    print("test successful")

def rotx(rad):
    R = np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]])
    #R  = np.round(R, 10)
    return R

def roty(rad):
    R = np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]])
    #R = np.round(R, 10)
    return R

def rotz(rad):
    R = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0,0,1]])
    #R = np.round(R,10)
    return R

def make_transf(rot, transl):
    transl = np.squeeze(transl)
    transf = np.zeros((4,4), dtype=np.float32)
    transf[0:3,0:3] = rot
    transf[3,3] = 1.0
    transf[0:3,3] = transl
    return transf




def invert_transf(T):
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    R_inv = R.transpose()
    t_inv = -R_inv@t
    T_inv = np.zeros((4,4), dtype=np.float32)
    T_inv[0:3,0:3] = R_inv
    T_inv[0:3, 3] = t_inv
    T_inv[3,3] = 1.0
    return T_inv

def decompose_transf_mat(transf):
    assert(transf.shape == (4,4) and transf[3,3] == 1.0)
    transl = np.expand_dims(transf[0:3, 3], 1)
    rot = transf[0:3, 0:3]
    return rot, transl

def decompose_homg_coord(homg_coord):
    return homg_coord[0:3], homg_coord[3]

def normalize_homg_coord(homg_coord):
    assert(len(homg_coord) == 4)
    norm_homg_coord = homg_coord/homg_coord[3]
    return norm_homg_coord

def homg_to_point(P):
    return (P/P[3])[0:3]

def point_to_nic(p):
    return p/p[2]

def plane_to_normal_distance(U):
    U = U/U[3]
    distance = np.linalg.norm(U[0:3])
    normal = -U[0:3]/distance
    return normal, distance



def homg_line_from_2_points(p1, p2):
    l = p1[3]*p2[0:3] - p2[3]*p1[0:3]
    l_dash = np.cross(p1[0:3], p2[0:3])
    homg_line = np.concatenate((l,l_dash))
    return homg_line

def point_to_homg(p):
    return np.vstack((p,1.0))


    
def homgPlaneFrom3Points(p1, p2, p3):
    n = np.cross((p1[0:3]-p3[0:3]), (p2[0:3] - p3[0:3]))
    d = -np.dot(p3[0:3], np.cross(p1[0:3], p2[0:3]))
    d = np.array([d])
    homg_plane = np.concatenate((n,d))
    return homg_plane

def plucker_plane_from_transf_mat(transf, plane):
    plane = ''.join(sorted(plane))
    transl = transf[0:3, 3]
    assert(plane == 'xy' or plane =='xz' or plane == 'yz')
    if plane == 'xy':
        normal = transf[0:3,2]
        dist = -np.dot(normal, transl)
        pass
    elif plane == 'xz':
        normal = transf[0:3,1]
        dist = -np.dot(normal, transl)
        pass
    elif plane == 'yz':
        normal = transf[0:3,0]
        dist = -np.dot(normal, transl)
    u = np.zeros((4,1))
    u[0:3] = np.expand_dims(normal/dist,1)
    u[3] = 1.0
    return u


def skew(v):
    if len(v) == 4: v = v[:3]/v[3]
    v1 = v[0]
    v2 = v[1]
    v3 = v[2]
    mat = np.array([[0, -v3, v2],[v3, 0, -v1],[-v2, v1, 0]])
    return mat

"""

did not get this to work
def getRotMatFrom2Vec(a, b):
    print("hei")
    v = np.cross(a,b)
    s = np.linalg.norm(v)
    c = np.dot(a,b)

    R = np.eye(3) + skew(v) + skew(v)@skew(v) /(1+c)

    return R
"""

# this one works
def cross(a,b):
    c = np.cross(a.T,b.T).T
    return c

def decompose_plucker_line(plucker_line):
    l = plucker_line[0:3]
    l_dash = plucker_line[3:6]
    return l, l_dash


def triangulate_point_known_plane(s, u):
    """
    10.3 from O.Egeland vision note p.119
    """
    u, u4 = decompose_homg_coord(u)
    x = - s / (u4*s.T@u)
    x = np.expand_dims(x,1)
    return x

def intersection_line_plane(plucker_line, plane):
    """
    9.18 from O.Egeland vision note p.105
    """
    u,u4 = decompose_homg_coord(plane)
    l, l_dash = decompose_plucker_line(plucker_line)
    x = -u4*l + cross(u, l_dash)
    x4 = u.T@l
    return x/x4


def getRotMatFrom2Vec(a, b):
    x = np.cross(a, b) 
    x = x / np.linalg.norm(x)

    theta = np.arccos(np.dot(a,b)/np.dot(np.linalg.norm(a), np.linalg.norm(b)))

    A = skew(x)

    R = np.eye(3) + np.sin(theta) * A + (1 - np.cos(theta)) * A@A

    return R

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def exponentialRotToRot(axis, theta):
    w_skew = skew(axis)
    R = np.eye(3) + np.sin(theta)*w_skew + (1-np.cos(theta))*w_skew@w_skew
    return R

def exponentialRotToRot2(u):
    theta = np.linalg.norm(u)
    axis = u/theta
    return exponentialRotToRot(axis, theta)

def MatrixLog3(R):
    """Computes the matrix logarithm of a rotation matrix
    :param R: A 3x3 rotation matrix
    :return: The matrix logarithm of R
    Example Input:
        R = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
    Output:
        np.array([[          0, -1.20919958,  1.20919958],
                  [ 1.20919958,           0, -1.20919958],
                  [-1.20919958,  1.20919958,           0]])
    """
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not NearZero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                  * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not NearZero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                  * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                  * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return VecToso3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)

def so3ToVec(so3mat):
    """
    LICENSE: Modern robotics

    Converts an so(3) representation to a 3-vector
    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The 3-vector corresponding to so3mat
    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([1, 2, 3])
    """
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

def vec_to_so3(vec):
    """
    LICENSE: Modern Robotics
    
    Converts a 3-vector to an so(3) representation
    :param omg: A 3-vector
    :return: The skew symmetric representation of omg
    Example Input:
        vec = np.array([1, 2, 3])
    Output:
        np.array([[ 0, -3,  2],
                  [ 3,  0, -1],
                  [-2,  1,  0]])
    """
    return np.array([[0,      -vec[2],  vec[1]],
                     [vec[2],       0, -vec[0]],
                     [-vec[1], vec[0],       0]])


def get_homography(u_C1, T_C2_C1, K1, K2):
    K1_inv = np.linalg.inv(K1)
    points_C1 = np.array([[900,100, 1], [600,900,1], [600,100,1],[900,900,1]]) # choose some pixel coords from cam1
    points_C2 = []

    for point in points_C1:
        norm_C1 = K1_inv@point
        x_C1 = triangulate_point_known_plane(norm_C1, u_C1)
        X_C1 = point_to_homg(x_C1)
        X_C2 = T_C2_C1@X_C1
        x_C2 = homg_to_point(X_C2)
        norm_C2 = point_to_nic(x_C2) # to normalized image coord
        pix_C2 = K2@norm_C2
        points_C2.append(pix_C2)

    points_C1 = points_C1
    points_C1 = points_C1[:, 0:2]
    points_C2 =  np.array(points_C2).squeeze()
    points_C2 = points_C2[:,0:2]
    homography = cv2.getPerspectiveTransform(points_C1.astype(np.float32), points_C2.astype(np.float32))
    return homography


def matrixLog3AngAx(R):
    R_log = MatrixLog3(R)
    angAx = so3ToVec(R_log)
    return angAx

if __name__ == '__main__':
    pass

