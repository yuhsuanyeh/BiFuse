import sys
import utils
import numpy as np
from vispy import app

if __name__ == '__main__':
    argv = sys.argv
    '''
    if len(argv) != 2:
        print ('Usage: python vis3D.py xxxx.npy')
        exit()
    data = np.load(argv[1], allow_pickle=True)
    '''

    data = np.load('../My_Test_Result/Data000.npy', allow_pickle=True).item()
    RGB = data['RGB']
    depth = data['depth']
    grid = utils.SphereGrid(*depth.shape)
    
    '''
    import matplotlib.pyplot as plt
    plt.imshow(depth)
    plt.show()
    '''

    aaa = 85
    bbb = 420

    pts = depth[..., None] * grid
    
    pts = pts[aaa:bbb, ...].reshape([-1, 3])
    RGB = RGB[aaa:bbb, ...].reshape([-1, 3])
    utils.CreateView(pts, RGB, pt_size=2)
    app.run()
