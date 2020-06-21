from __future__ import print_function

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../..')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable


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

class EquirecRotate2:
    def __init__(self, equ_h, equ_w, RADIUS=128, CUDA=True, VAR=False):
        cen_x = (equ_w - 1) / 2.0
        cen_y = (equ_h - 1) / 2.0
        equ_h = int(equ_h)
        equ_w = int(equ_w)
        theta = (2 * (np.arange(equ_w) - cen_x) / equ_w) * np.pi
        phi = (2 * (np.arange(equ_h) - cen_y) / equ_h) * (np.pi / 2)
        self.RADIUS = RADIUS        
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
        self.CUDA = CUDA 
    def GetGrid(self):
        return Variable(self.xyz)

    def Rotate(self, batch, rotation, mode='bilinear'):
        #
        # batch is [batch_size x 3 x h x w]
        # rotation is [batch x 3], x, y, z
        #
        assert mode in ['bilinear', 'nearest']
        R = euler2mat(rotation).transpose(1, 2)
        [batch_size, _, _, _] = batch.size()
        tmp = []
        xyz = self.xyz.cuda() if self.CUDA else self.xyz
        for i in range(batch_size):
            this_img = batch[i:i+1, :, :, :]
            
            new_xyz = torch.matmul(xyz, R[i:i+1, :, :].transpose(1, 2))
            
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
            new_img = F.grid_sample(this_img, loc, mode=mode)
            tmp.append(new_img)
        out = torch.cat(tmp, dim=0)
        return out

if __name__ == '__main__':
    img = cv2.imread('/media/external/Fu-En.Wang/Data/360/final/rotated/117a5a3b1cd3298e31aeaae786c6bf02/0.txt/14_color.png', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    origin = img.astype(float).copy() / 255
    img = img.reshape([1, 512, 1024, 3]).swapaxes(1, 3).swapaxes(2, 3)
    
    img = img.astype(float) / 255
    
    
    batch = torch.FloatTensor(img).cuda()

    ER = EquirecRotate(512, 1024)

    angle = torch.FloatTensor(np.array([0, 90, 0]).reshape([1, 3])).cuda()
    angle = angle / 180 * np.pi
    
    import time
    a = time.time()
    c = 1
    for i in range(c):
        print(i)
        batch = ER.Rotate(batch, angle)
        after = batch.view(3, 512, 1024).transpose(0, 2).transpose(0, 1).data.cpu().numpy()
    b = time.time()
    print('FPS: %lf'%(c / (b - a)))
    big = np.concatenate([origin, after], axis=0)
    plt.imshow(big) 
    plt.show()
