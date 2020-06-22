import torch
import torch.nn as nn
import math
import pdb
import numpy as np

import matplotlib.pyplot as plt
import torch.utils.model_zoo as model_zoo

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from .CETransform import CETransform
from .Equirec2Cube import Equirec2Cube
from .SpherePad import SpherePad 

CE = CETransform()

class CustomPad(nn.Module):
    def __init__(self, pad_func):
        super(CustomPad, self).__init__()
        self.pad_func = pad_func

    def forward(self, x):
        return self.pad_func(x)

class NoOp(nn.Module):
    def __init__(self):
        super(NoOp, self).__init__()

    def forward(self, x):
        return x


class ZeroPad(nn.Module):
    def __init__(self, pad_s):
        super(ZeroPad, self).__init__()
        self.pad_s = pad_s
    
    def forward(self, x):
        x = F.pad(x, (self.pad_s, self.pad_s, self.pad_s, self.pad_s)) 
        return x

class CubePad(nn.Module):
    def __init__(self, pad_size, pad_corner=True, CUDA=True):
        super(CubePad, self).__init__()
        self.CUDA = CUDA
        self.pad_corner = pad_corner 
        if type(pad_size) == int:
            self.up_pad = pad_size
            self.down_pad = pad_size
            self.left_pad = pad_size
            self.right_pad = pad_size
        elif type(pad_size) == list:
            [self.up_pad, self.down_pad, self.left_pad, self.right_pad] = pad_size

        # pad order: up, down, left, right sides
        # use yes/no flag to choose flip/transpose or not
        # notation: #face-#side_#flip-hor_#flip_ver_#transpose
        # transpose is applied first

        self.relation = {
            'back': ['top-up_yes_yes_no', 'down-down_yes_yes_no', 'right-right_no_no_no', 'left-left_no_no_no'],
            'down': ['front-down_no_no_no', 'back-down_yes_yes_no', 'left-down_yes_no_yes', 'right-down_no_yes_yes'],
            'front': ['top-down_no_no_no', 'down-up_no_no_no', 'left-right_no_no_no', 'right-left_no_no_no'],

            'left': ['top-left_yes_no_yes', 'down-left_no_yes_yes', 'back-right_no_no_no', 'front-left_no_no_no'],
            'right': ['top-right_no_yes_yes', 'down-right_yes_no_yes', 'front-right_no_no_no', 'back-left_no_no_no'],
            'top': ['back-up_yes_yes_no', 'front-up_no_no_no', 'left-up_no_yes_yes', 'right-up_yes_no_yes']
        }

    def forward(self, x):
        # [back, down, front, left, right, top]
        [bs, c, h, w] = x.size()
        assert (bs % 6 == 0) and (h == w)
        [up_pad, down_pad, left_pad, right_pad] = [self.up_pad, self.down_pad, self.left_pad, self.right_pad]
        mx_pad = max([up_pad, down_pad, left_pad, right_pad])
        if mx_pad <= 0:
            return x
        faces = {
            'back': None,
            'down': None,
            'front': None,
            'left': None,
            'right': None,
            'top': None
        }
        sides = {
            'back-up': None, 'back-down': None, 'back-left': None, 'back-right': None,
            'down-up': None, 'down-down': None, 'down-left': None, 'down-right': None,
            'front-up': None, 'front-down': None, 'front-left': None, 'front-right': None,
            'left-up': None, 'left-down': None, 'left-left': None, 'left-right': None,
            'right-up': None, 'right-down': None, 'right-left': None, 'right-right': None,
            'top-up': None, 'top-down': None, 'top-left': None, 'top-right': None
        }
        for idx, face in enumerate(['back', 'down', 'front', 'left', 'right', 'top']):
            tmp = x[idx::6, :, :, :]
            faces[face] = tmp
            for side in ['up', 'down', 'left', 'right']:
                if side == 'up':
                    pad_array = tmp[:, :, 0:mx_pad, :]
                elif side == 'down':
                    pad_array = tmp[:, :, h-mx_pad:h, :]
                elif side == 'left':
                    pad_array = tmp[:, :, :, 0:mx_pad]
                elif side == 'right':
                    pad_array = tmp[:, :, :, w-mx_pad:w]
                key = '%s-%s' % (face, side)
                assert key in sides
                sides[key] = pad_array
        
        out = []
        for idx, f in enumerate(['back', 'down', 'front', 'left', 'right', 'top']):
            face  = faces[f]
            new_face = F.pad(face, (left_pad, right_pad, up_pad, down_pad), 'constant', 0)
            [bs, _, new_h, new_w] = new_face.size()
            assert new_h == new_w
            for pad_order, relation in zip(['up', 'down', 'left', 'right'], self.relation[f]):
                pad_side, flip_h, flip_w, transpose = relation.split('_')
                pad_array = sides[pad_side]
                #print pad_order, pad_array is None

                if transpose == 'yes':
                    pad_array = pad_array.transpose(2, 3)
                [_, _, hh, ww] = pad_array.size()
                if flip_h == 'yes':
                    index = Variable(torch.arange(hh-1, -1, -1).type(torch.LongTensor))
                    if self.CUDA: index = index.cuda()
                    pad_array = torch.index_select(pad_array, dim=2, index=index)
                if flip_w == 'yes':
                    index = Variable(torch.arange(ww-1, -1, -1).type(torch.LongTensor))
                    if self.CUDA: index = index.cuda()
                    pad_array = torch.index_select(pad_array, dim=3, index=index)

                if pad_order == 'up' and up_pad != 0:
                    new_face[:, :, 0:up_pad, left_pad:new_w-right_pad] = pad_array[:, :, 0:up_pad, :]
                    #print (new_face[:, :, 0:pad, pad:new_w-pad].size(), pad_array.size())
                elif pad_order == 'down' and down_pad != 0:
                    new_face[:, :, new_h-down_pad:new_h, left_pad:new_w-right_pad] = pad_array[:, :, 0:down_pad, :]
                    #print (new_face[:, :, new_h-pad:new_h, pad:new_w-pad].size(), pad_array.size())
                elif pad_order == 'left' and left_pad != 0:
                    new_face[:, :, up_pad:new_h-down_pad, 0:left_pad] = pad_array[:, :, :, 0:left_pad]
                    #print (new_face[:, :, pad:new_h-pad, 0:pad].size(), pad_array.size())
                elif pad_order == 'right' and right_pad != 0:
                    #print (left_pad, right_pad, up_pad, down_pad)
                    new_face[:, :, up_pad:new_h-down_pad, new_w-right_pad:new_w] = pad_array[:, :, :, 0:right_pad]
                    #print new_face[:, :, up_pad:new_h-down_pad, new_w-right_pad:new_w]
            out.append(new_face)
        out = torch.cat(out, dim=0)
        [bs, c, h, w] = out.size()
        out2 = out.view(-1, bs//6, c, h, w).transpose(0, 1).contiguous().view(bs, c, h, w)
        if self.pad_corner:
            for corner in ['left_up', 'right_up', 'left_down', 'right_down']:
                if corner == 'left_up' and (left_pad > 0 and up_pad > 0):
                    out2[:, :, 0:up_pad, 0:left_pad] = out[:, :, 0:up_pad, left_pad:left_pad+1].repeat(1, 1, 1, left_pad).clone()

                elif corner == 'right_up' and (right_pad > 0 and up_pad > 0):
                    out2[:, :, 0:up_pad, w-right_pad:w] = out[:, :, 0:up_pad, (w-right_pad-1):(w-right_pad)].repeat(1, 1, 1, right_pad).clone()

                elif corner == 'left_down' and (left_pad > 0 and down_pad > 0):
                    out2[:, :, h-down_pad:h, 0:left_pad] = out[:, :, h-down_pad:h, left_pad:left_pad+1].repeat(1, 1, 1, left_pad).clone()

                elif corner == 'right_down' and (right_pad > 0 and down_pad > 0):
                    out2[:, :, h-down_pad:h, w-right_pad:w] = out[:, :, h-down_pad:h, (w-right_pad-1):(w-right_pad)].repeat(1, 1, 1, right_pad).clone()

        return out2


