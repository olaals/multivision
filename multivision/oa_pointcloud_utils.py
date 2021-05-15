import numpy as np
from oa_robotics import *
import open3d as o3d
import cv2



def get_image_coordinates_treshold(img, threshold):
    coords = np.flip(np.row_stack(np.where(img > threshold)))
    pad = np.pad(coords, [(0,1),(0,0)], mode='constant', constant_values=[(0,1.0),(0,0)])
    return pad

def scan_image_to_pointcloud(scan_img, transf_cam_laser, cam_mat, threshold):
    if len(scan_img.shape) == 3:
        scan_img_gray = cv2.cvtColor(scan_img, cv2.COLOR_RGB2GRAY)
    else:
        scan_img_gray = scan_img

    inv_cam_mat = np.linalg.inv(cam_mat)
    img_coords= get_image_coordinates_treshold(scan_img_gray, threshold)
    u = plucker_plane_from_transf_mat(transf_cam_laser, 'yz')
    normal, _ = decompose_homg_coord(u)
    norm_coords = np.einsum('ij, jk -> ik', inv_cam_mat, img_coords)
    p4 = -np.einsum('ij, jk -> ik', normal.T, norm_coords)
    points = norm_coords / p4
    return points

def change_frame_of_pointcloud(points_frame2, transf_frame1_frame2):
    assert(points_frame2.shape[0] == 3)
    assert(transf_frame1_frame2.shape == (4,4))
    points = points_frame2
    transf = transf_frame1_frame2
    points = np.pad(points, [(0,1),(0,0)], mode='constant', constant_values=[(0,1.0),(0,0)])
    points_frame1 = np.einsum('ij,jk->ik', transf, points)[:3, :]
    return points_frame1

def pointcloud_to_norm_coords(points):
    assert(points.shape[0] == 3)
    norm_coords = points/points[2,:]
    return norm_coords

def norm_to_pixel(norm_coords, cam_mat):
    pix_coords = np.einsum('ij,jk->ik', cam_mat, norm_coords)
    return pix_coords









def pointcloud_to_image(points, cam_mat):
    assert(points.shape[0] == 3)
    assert(cam_mat.shape == (3,3))
    img_width = int(cam_mat[0, 2]*2)
    img_height = int(cam_mat[1,2]*2)
    img = np.zeros((img_height, img_width), dtype='uint8')
    norm_coords = points/points[2,:]
    pix_coords = np.round(np.einsum('ij,jk->ik', cam_mat, norm_coords)).astype(np.uint32)
    xs = pix_coords[0,:]
    xs[xs>=img_width] = 0
    xs[xs<0] = 0
    ys = pix_coords[1,:]
    ys[ys>=img_height] = 0
    ys[ys<0] = 0
    img[(ys,xs)] = 255
    return img

def pointcloud_to_image2(points, cam_mat, img_width, img_height):
    assert(points.shape[0] == 3)
    assert(cam_mat.shape == (3,3))
    img = np.zeros((img_height, img_width), dtype='uint8')
    norm_coords = points/points[2,:]
    pix_coords = np.round(np.einsum('ij,jk->ik', cam_mat, norm_coords)).astype(np.uint32)
    xs = pix_coords[0,:]
    xs[xs>=img_width] = 0
    xs[xs<0] = 0
    ys = pix_coords[1,:]
    ys[ys>=img_height] = 0
    ys[ys<0] = 0
    img[(ys,xs)] = 255
    return img

def pointcloud_to_image3(points, cam_mat, img_width, img_height):
    assert(points.shape[0] == 3)
    assert(cam_mat.shape == (3,3))
    img = np.zeros((img_height*2, img_width*2), dtype='uint8')
    new_cam_mat = cam_mat
    new_cam_mat[0,2] = (img_width*2-1.0)/2.0
    new_cam_mat[1,2] = (img_height*2-1.0)/2.0
    new_cam_mat[0,0] *= 2.0
    new_cam_mat[1,1] *= 2.0
    print("new cam mat")
    print(new_cam_mat)
    
    norm_coords = points/points[2,:]
    pix_coords = np.round(np.einsum('ij,jk->ik', new_cam_mat, norm_coords)).astype(np.uint32)
    xs = pix_coords[0,:]
    xs[xs>=img_width*2] = 0
    xs[xs<0] = 0
    ys = pix_coords[1,:]
    ys[ys>=img_height*2] = 0
    ys[ys<0] = 0
    img[(ys,xs)] = 255
    img = cv2.resize(img, (img_width, img_height))
    return img

def pointcloud_to_image4(points, cam_mat):
    assert(points.shape[0] == 3)
    assert(cam_mat.shape == (3,3))
    img_width = int(cam_mat[0, 2]*2)
    img_height = int(cam_mat[1,2]*2)
    img = np.zeros((img_height, img_width), dtype='uint8')
    norm_coords = points/points[2,:]
    pix_coords = np.einsum('ij,jk->ik', cam_mat, norm_coords)

    xs = pix_coords[0,:]
    xs[xs>=img_width] = 0
    xs[xs<0] = 0
    ys = pix_coords[1,:]
    ys[ys>=img_height] = 0
    ys[ys<0] = 0
    img[(ys,xs)] = 255
    return img

def save_pointcloud(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.T)
    o3d.io.write_point_cloud(filename,pcd)

def load_pointcloud(filename):
    pcd_load = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd_load.points)
    return points.T

