from __future__ import print_function

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../..')
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import Utils
from . import Equirec2Cube as E2C

def resize(img, scale):
    [h, w, _] = img.shape
    tmp = cv2.resize(img, (int(round(scale*w)), int(round(scale*h))))
    return tmp

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

    zeros = z.clone()*0
    ones = zeros.clone()+1
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

def Rodrigues(axis):
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
    
    zero = (theta == 0)
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

class EquirecRotate:
    def __init__(self, equ_h, equ_w, RADIUS=128, CUDA=True, VAR=False):
        cen_x = (equ_w - 1) / 2.0
        cen_y = (equ_h - 1) / 2.0

        theta = (2 * (np.arange(equ_w) - cen_x) / equ_w) * np.pi
        phi = (2 * (np.arange(equ_h) - cen_y) / equ_h) * (np.pi / 2)
        self.RADIUS = RADIUS        
        if CUDA:
            theta = torch.FloatTensor(theta).cuda()
            phi = torch.FloatTensor(phi).cuda()
        else:
            theta = torch.FloatTensor(theta)
            phi = torch.FloatTensor(phi)
        
        if VAR:
            theta = Variable(theta)
            phi = Variable(phi)
        theta = theta.repeat(equ_h, 1)
        phi = phi.repeat(equ_w, 1).transpose(0, 1)
        
        x = (RADIUS * torch.cos(phi) * torch.sin(theta)).view(equ_h, equ_w, 1)
        y = (RADIUS * torch.sin(phi)).view(equ_h, equ_w, 1)
        z = (RADIUS * torch.cos(phi) * torch.cos(theta)).view(equ_h, equ_w, 1)

        self.xyz = torch.cat([x, y, z], dim=2).view(1, equ_h, equ_w, 3)
    
    def GetGrid(self):
        return Variable(self.xyz)

    def Rotate(self, batch, rotation):
        #
        # batch is [batch_size x 3 x h x w]
        # rotation is [batch x 2], theta and phi
        #
        theta = rotation[:, 0].contiguous().view(-1, 1)
        phi = -rotation[:, 1].contiguous().view(-1, 1)

        #x_angle = phi
        #y_angle = theta
        #z_angle = phi.clone() * 0
        
        zero = theta.clone() * 0
        one = theta.clone() * 0 + 1
        
        x_axis = torch.cat([one, zero, zero], dim=1)
        y_axis = torch.cat([zero, one, zero], dim=1)

        R1 = Rodrigues((theta * y_axis).view(-1, 1, 3)).view(-1, 3, 3)
        new_axis = torch.matmul(R1, torch.unsqueeze(x_axis, 2)).view(-1, 3)

        R2 = Rodrigues((phi * new_axis).view(-1, 1, 3)).view(-1, 3, 3)

        [batch_size, _, _, _] = batch.size()
        tmp = []
        for i in range(batch_size):
            this_img = batch[i:i+1, :, :, :]
            
            new_xyz = torch.matmul(self.xyz, R1[i:i+1, :, :].transpose(1, 2))
            new_xyz = torch.matmul(new_xyz, R2[i:i+1, :, :].transpose(1, 2))
            
            x = torch.unsqueeze(new_xyz[:, :, :, 0], 3)
            y = torch.unsqueeze(new_xyz[:, :, :, 1], 3)
            z = torch.unsqueeze(new_xyz[:, :, :, 2], 3)
            
            lon = torch.atan2(x, z) / np.pi
            lat = torch.asin(y / self.RADIUS) / (0.5 * np.pi)
            loc = torch.cat([lon, lat], dim=3)
            
            #print this_img.size()
            #print torch.max(loc)
            #print torch.min(loc)
            #exit()
            new_img = F.grid_sample(this_img, loc)
            tmp.append(new_img)
        out = torch.cat(tmp, dim=0)
        return out

if __name__ == '__main__':

    img = cv2.imread('src/image.jpg', cv2.IMREAD_COLOR)

    img = resize(img, 0.22)
    [h, w, _] = img.shape
    #cv2.namedWindow('GG')
    #cv2.imshow('GG', img)
    #cv2.waitKey(0)
    equirec = EquirecRotate(h, w)

    BS = 3

    e2c = E2C.Equirec2Cube(BS, h, w, 256, 90)
    print(e2c.intrisic)
    print(img.shape)
    #exit()
    batch = img.reshape([1, h, w, 3])
    batch = np.swapaxes(batch, 1, 3)
    batch = np.swapaxes(batch, 2, 3)
    tmp = [batch for x in range(BS)]
    batch = np.concatenate(tmp, axis=0)

    angle = np.zeros([BS, 2])
    angle[0, 0] = 180.0 / 180 * np.pi
    angle[0, 1] = 90.0 / 180 * np.pi
    angle[1, 0] = 20.0 / 180 * np.pi
    angle[1, 1] = 30.0 / 180 * np.pi
    angle[2, 0] = -60.0 / 180 * np.pi
    angle[2, 1] = 23.0 / 180 * np.pi

    batch = torch.FloatTensor(batch.astype(np.float32) / 255).cuda()
    angle = torch.FloatTensor(angle).cuda()
    result = equirec.Rotate(batch, angle)
    cubes = e2c.ToCubeTensor(result)
    cv2.namedWindow('GG')
    for i in range(BS):
        origin = batch[i, :, :, :].transpose(0, 2).transpose(0, 1).cpu().numpy()
        new = result[i, :, :, :].transpose(0, 2).transpose(0, 1).data.cpu().numpy()
        big = resize(np.concatenate([origin, new], axis=0), 0.5)

        cv2.imshow('GG', big)
        cv2.waitKey(0)
        
        for j in range(6):
            cube = cubes[i*6 + j, :, :, :].transpose(0, 2).transpose(0, 1).data.cpu().numpy()
            cv2.imshow('GG', cube)
            cv2.waitKey(0)













