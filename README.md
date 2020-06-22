# [CVPR2020]BiFuse: Monocular 360 Depth Estimation via Bi-Projection Fusion

<p align='center'>
<img src='1690-teaser.gif'>
</p>

### [[Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_BiFuse_Monocular_360_Depth_Estimation_via_Bi-Projection_Fusion_CVPR_2020_paper.pdf)] [[Project Page](https://fuenwang.ml/project/bifuse/)]

## Getting Started
#### Requirements
- Python (tested on 3.7.4)
- PyTorch (tested on 1.4.0)
- Other dependencies
```
pip install -r requirements.txt
```

## Usage
First clone our repo:
```
git clone https://github.com/Yeh-yu-hsuan/BiFuse.git
cd BiFuse
```
#### Step1
Download our [pretrained Model](https://drive.google.com/file/d/1EOEfyVuaJC1k5xAtqG37yXHxN-LnxA2n/view?usp=sharing) and create a **save** folder:
```
mkdir save
```
then put the ```BiFuse_Pretrained.pkl``` into **save** folder.
#### Step2
**My_Test_Data** folder has contained a ```Sample.jpg``` RGB image as an example. <br> 
If you want to test your own data, please put your own rgb images into **My_Test_Data** folder and run:
```
python main.py --path './My_Test_Data' --crop True
```
The default argument of **path** and **crop** is defined as above. <br>
**```--path```** is the folder path of your own testing images.  <br>
**```--crop```** is **True** or **False** which decides whether discard the invalid region or not. <br>

If you don't need to change anything, you can simply run:
```
python main.py
```
After testing, you can see the results in **My_Test_Result** folder!

## License
This work is licensed under MIT License. See [LICENSE](LICENSE) for details. 

If you find our code/models useful, please consider citing our paper:
```
@InProceedings{BiFuse20,
author = {Wang, Fu-En and Yeh, Yu-Hsuan and Sun, Min and Chiu, Wei-Chen and Tsai, Yi-Hsuan},
title = {BiFuse: Monocular 360 Depth Estimation via Bi-Projection Fusion},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

