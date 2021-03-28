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

def makeTrans(R,t):
    T = np.zeros((4,4))
    T[3,3] = 1.0
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T

def invertTrans(T):
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    R_inv = R.transpose()
    t_inv = -R_inv@t
    T_inv = np.zeros((4,4))
    T_inv[0:3,0:3] = R_inv
    T_inv[0:3, 3] = t_inv
    T_inv[3,3] = 1.0
    return T_inv

def homgLineFrom2Points(p1, p2):

    l = p1[3]*p2[0:3] - p2[3]*p1[0:3]
    l_dash = np.cross(p1[0:3], p2[0:3])

    homg_line = np.concatenate((l,l_dash))
    print(homg_line)

    return homg_line
    
def homgPlaneFrom3Points(p1, p2, p3):
    n = np.cross((p1[0:3]-p3[0:3]), (p2[0:3] - p3[0:3]))
    d = -np.dot(p3[0:3], np.cross(p1[0:3], p2[0:3]))
    d = np.array([d])
    homg_plane = np.concatenate((n,d))
    return homg_plane

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

def matrixLog3AngAx(R):
    R_log = MatrixLog3(R)
    angAx = so3ToVec(R_log)
    return angAx

