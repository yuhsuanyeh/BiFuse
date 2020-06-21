from __future__ import print_function

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/..')
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from .Equirec2Cube import Equirec2Cube
from .Equirec2Cube import EquirecRotate2 as EquirecRotate

def MyRodrigues_varify(axis):
    theta = np.linalg.norm(axis)
    r = (axis / theta).reshape(-1, 1)
     
    g = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]], np.float32)
    R = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * r.dot(r.T) + np.sin(theta) * g
    
    print(R)

def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix in the order of R, t -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    else:
        raise ValueError("Unknown rotation mode!!")

    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]

    return transform_mat

def RodriguesTensor(axis):
    #
    # axis is [batch x K x 3]
    # K dimension is for SfM-Net
    # output is [batch x K x 9]

    theta = torch.unsqueeze(torch.norm(axis, p = 2, dim = 2), 2)
    r = axis / theta
    a = torch.cos(theta)
    b = 1 - a
    c = torch.sin(theta)
    rx = torch.unsqueeze(r[:, :, 0], 2)
    ry = torch.unsqueeze(r[:, :, 1], 2)
    rz = torch.unsqueeze(r[:, :, 2], 2)
    
    zero = (theta == 0).detach()
    R1 = a + b * rx * rx
    R1[zero] = 1

    R2 = b * rx * ry - c * rz
    R2[zero] = 0

    R3 = b * rx * rz + c * ry
    R3[zero] = 0

    R4 = b * rx * ry + c * rz
    R4[zero] = 0

    R5 = a + b * ry * ry
    R5[zero] = 1

    R6 = b * ry * rz - c * rx
    R6[zero] = 0

    R7 = b * rx * rz - c * ry
    R7[zero] = 0

    R8 = b * ry * rz + c * rx
    R8[zero] = 0

    R9 = a + b * rz * rz
    R9[zero] = 1
    
    out = torch.cat([R1, R2, R3, R4, R5, R6, R7, R8, R9], dim = 2)
    return out
    #print theta
    #print axis / theta 


def RodriguesTensor_unitest():
    s = 6
    k = 4
    batch = np.zeros([s, k, 3])

    for i in range(s):
        for j in range(k):
            batch[i, j, :] = np.random.rand(3) - 0.5
    batch[s/2, k/2, :] = 0

    batch_tensor = Variable(torch.FloatTensor(batch), requires_grad = True).cuda()
    
    R = RodriguesTensor(batch_tensor)
    
    for i in range(s):
        for j in range(k):
            print('=============')
            axis = batch[i, j, :]
            R_GT = cv2.Rodrigues(axis)[0]
            R_my = R[i, j, :].view(3, 3).data.cpu().numpy()
            
            print(R_GT)
            print(R_my)
            try:
                a = input()
            except:
                pass


# Convert euler angle to rotation matrix
def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).view(B, 3, 3)

    rotMat = xmat.bmm(ymat).bmm(zmat)
    return rotMat


# Convert rotation matrix to euler angles
def mat2euler(mat):
    """ Convert rotation matrix to euler angles.
    
    https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L283

    Args:
        mat: rotation matrix in zyx format -- size = [B, 3, 3]
    Returns:
        angle: rotation angle along 3 axis (in radians, it's not unique) -- size = [B, 3]

    """
    cy_thresh = 1e-10
    cy = torch.sqrt(mat[:, 2, 2]*mat[:, 2, 2] + mat[:, 1, 2]*mat[:, 1, 2])
    
    if (cy > cy_thresh).any(): # cos(y) not close to zero, standard form
        z = torch.atan2(-mat[:, 0, 1],  mat[:, 0, 0]) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = torch.atan2(mat[:, 0, 2],  cy) # atan2(sin(y), cy)
        x = torch.atan2(-mat[:, 1, 2], mat[:, 2, 2]) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = torch.atan2(mat[:, 1, 0],  mat[:, 1, 1])
        y = torch.atan2(mat[:, 0, 2],  cy) # atan2(sin(y), cy)
        x = torch.zeros_like(mat[:, 0, 0])

    return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], -1).view(-1, 3)


# Convert quaternion to rotation matrix
def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

class Depth2Points(nn.Module):
    def __init__(self, xyz_grid, CUDA=True):
        #
        # xyz_grid is [6 x h x w x 3]
        # grid order is ['back', 'down', 'front', 'left', 'right', 'up']

        super(Depth2Points, self).__init__()
        self.xyz_grid = xyz_grid
        self.order = ['back', 'down', 'front', 'left', 'right', 'up']
        self.CUDA = CUDA

    def forward(self, x):
        #
        # x is [6*bs x 1 x h x w]
        #
        #


        [bs, c, h, w] = x.size()

        if bs % 6 != 0 or c != 1:
            print("Batch size mismatch in Depth2Points")
            exit()

        bs = bs // 6
        grid = self.xyz_grid
        grid = grid.cuda() if self.CUDA else grid
        all_pts = [] 
        for i in range(bs):
            cubemap = x[i*6:(i+1)*6, 0, :, :] # 6 x h x w
            for j, face in enumerate(self.order):
                if face == 'back' or face == 'front':
                    # depth is z axis
                    # cubemap[j, :, :] is h x w
                    # grid[j, :, :, 0] is h x w
                    #
                    scale = cubemap[j, :, :] / torch.abs(grid[j, :, :, 2])
                elif face == 'down' or face == 'up':
                    # depth is y axis
                    scale = cubemap[j, :, :] / torch.abs(grid[j, :, :, 1])
                elif face == 'left' or face == 'right':
                    # depth is x axis
                    scale = cubemap[j, :, :] / torch.abs(grid[j, :, :, 0])
                else:
                    print('Order error in Depth2Points')
                    exit()

                pt_x = (scale * grid[j, :, :, 0]).view(1, h, w, 1)
                pt_y = (scale * grid[j, :, :, 1]).view(1, h, w, 1)
                pt_z = (scale * grid[j, :, :, 2]).view(1, h, w, 1)
                pt = torch.cat([pt_x, pt_y, pt_z], dim=3) 
                all_pts.append(pt)
        point_cloud = torch.cat(all_pts, dim=0)
        #print point_cloud
        return point_cloud

class EquirecDepth2Points(nn.Module):
    def __init__(self, xyz_grid, CUDA=True):
        #
        # xyz_grid is [1 x n x 2n x 3]
        # grid order is ['back', 'down', 'front', 'left', 'right', 'up']

        super(EquirecDepth2Points, self).__init__()
        self.grid = xyz_grid
        self.CUDA = CUDA

    def forward(self, depth):
        #
        #  depth is [batch x 1 x 2n x n]
        #
        #print self.grid
        norm = torch.norm(self.grid, p=2, dim=3).unsqueeze(3) # 1 x n x 2n x 1
        pts = []
        grid = self.grid.cuda() if self.CUDA else self.grid
        for i in range(depth.size()[0]):
            tmp = (grid / norm) * depth[i:i+1, 0, :, :].unsqueeze(3) # 1 x n x 2n x 3
            pts.append(tmp)

        result = torch.cat(pts, dim=0)
        return result

# Convert 6DoF parameter to transformation matrix


if __name__ == '__main__':
    '''
    #RodriguesTensor_unitest()
    bs = 2
    out_dim = 240
    e2c = Equirec2Cube(bs, 300, 600, out_dim, 90)
    batch = Variable(torch.FloatTensor(np.ones([bs * 6, 1, out_dim, out_dim])), requires_grad=False).cuda()

    d2p = Depth2Points(e2c.GetGrid())

    print d2p(batch)
    #print d2p.state_dict()
    '''
    bs = 24
    h = 500
    w = 1000
    grid = EquirecRotate(h, w).GetGrid()
    depth = torch.FloatTensor(np.random.rand(bs, 1, h, w) * 0.5).cuda()
    t = EquirecDepth2Points(grid)
    print(t(Variable(depth, requires_grad=True)))
    #import ipdb
    #ipdb.set_trace()
