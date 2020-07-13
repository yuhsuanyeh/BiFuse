# Point Cloud Visualization

We provide simple script visualize the 3D point cloud from predicted depth map. After you finish running depth prediction by
```bash
python main.py
```
You can run the script **vis3D.py** and **specify the npy file** from main.py. For example,
```bash
python vis3D.py ../My_Test_Result/Data000.npy
```
Then the visualized result will start running.

**Please notice that this code is based on VisPy library. If you haven't install it, the simplest way to install is using anaconda**
```
conda install vispy
```

<img src="https://github.com/Yeh-yu-hsuan/BiFuse/blob/master/src/fig.png" href="https://github.com/Yeh-yu-hsuan/BiFuse/blob/master/src/fig.png" width="60%"></img>
