# BiFuse: Monocular 360 Depth Estimation via Bi-Projection Fusion

<p align='center'>
<img src='1690-teaser.gif'>
</p>

### [[Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_BiFuse_Monocular_360_Depth_Estimation_via_Bi-Projection_Fusion_CVPR_2020_paper.pdf)] [[Project Page](https://fuenwang.ml/project/bifuse/)]

## Getting Started
#### Requirements
- Linux (tested on Ubuntu 18.04.4 LTS)
- Python (tested on 3.7.4)
- PyTorch (tested on 1.4.0)

## Usage
#### Step1
Download our [pretrained Model](https://drive.google.com/file/d/1EOEfyVuaJC1k5xAtqG37yXHxN-LnxA2n/view?usp=sharing) and put the .pkl file into **save** folder.
#### Step2
Put your own rgb images into **My_Test_Data** folder and run ```main.py``` to test your data. <br>
After testing, you can see the results in **My_Test_Result** folder!

## License
This work is licensed under MIT License. See [LICENSE](LICENSE) for details. 

If you find our code/models useful, please consider citing our paper:
```
@InProceedings{Wang_2020_CVPR,
author = {Wang, Fu-En and Yeh, Yu-Hsuan and Sun, Min and Chiu, Wei-Chen and Tsai, Yi-Hsuan},
title = {BiFuse: Monocular 360 Depth Estimation via Bi-Projection Fusion},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

