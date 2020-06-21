import torch
import torch.nn as nn
from . import Equirec2Cube, Cube2Equirec


class CETransform(nn.Module):
    def __init__(self):
        super(CETransform, self).__init__()
        equ_h = [512, 256, 128, 64, 32, 16]
        cube_h = [256, 128, 64, 32, 16, 8]

        self.c2e = dict()
        self.e2c = dict()

        for h in equ_h:
            a = Equirec2Cube(1, h, h*2, h//2, 90)
            self.e2c['(%d,%d)' % (h, h*2)] = a

        for h in cube_h:
            a = Cube2Equirec(1, h, h*2, h*4)
            self.c2e['(%d)' % (h)] = a

    def E2C(self, x, mode='bilinear'):
        [bs, c, h, w] = x.shape
        key = '(%d,%d)' % (h, w)
        assert key in self.e2c and mode in ['nearest', 'bilinear']
        return self.e2c[key].ToCubeTensor(x, mode=mode)

    def C2E(self, x, mode='bilinear'):
        [bs, c, h, w] = x.shape
        key = '(%d)' % (h)
        assert key in self.c2e and h == w and mode in ['nearest', 'bilinear']
        return self.c2e[key].ToEquirecTensor(x, mode=mode)

    def forward(self, equi, cube):
        return self.e2c(equi), self.c2e(cube)
