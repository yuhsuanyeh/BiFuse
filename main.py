from __future__ import print_function
import os
import cv2
import tqdm
import json
import argparse
import numpy as np
from PIL import Image
from imageio import imwrite
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
import Utils

parser = argparse.ArgumentParser(description='BiFuse script for 360 depth prediction!',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', default='./My_Test_Data', type=str, help='Path of source images')
parser.add_argument('--nocrop', action='store_true', help='Disable cropping')
args = parser.parse_args()

class MyData(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = transforms.Compose([
            transforms.ToTensor()
            ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        rgb_img = Image.open(img_path).convert("RGB")
        rgb_img = np.array(rgb_img, np.float32) / 255
        rgb_img = cv2.resize(rgb_img, (1024, 512), interpolation=cv2.INTER_AREA)
        data = self.transforms(rgb_img)

        return data

    def __len__(self):
        return len(self.imgs)

def Run(loader, model, crop):
    model = model.eval()
    pbar = tqdm.tqdm(loader)
    pbar.set_description('Validation process')
    gpu_num = torch.cuda.device_count()
    os.system('mkdir -p My_Test_Result')

    CE = Utils.CETransform()
    count = 0

    with torch.no_grad():
        for it, data in enumerate(pbar):
            inputs = data.cuda()
            raw_pred_var, pred_cube_var, refine = model(inputs)
            ### Convert to Numpy and Normalize to 0~1 ###
            dep_np = torch.clamp(refine, 0, 10).data.cpu().numpy()
            d_tmp = dep_np.copy()
            dep_np = dep_np/10
            rgb_np = data.permute(0,2,3,1).data.cpu().numpy()

            for i in range(dep_np.shape[0]):
                cat_rgb = rgb_np[i]
                cat_dep = dep_np[i, 0][..., None]
                cat_dep = np.repeat(cat_dep, 3, axis=2)
                white = np.ones((5, 1024, 3))
                ### Crop area is 68 to up and down ### 
                area = 68 if crop else 0
                upper = area
                lower = 512 - area

                big = np.concatenate([cat_rgb[upper:lower], white, cat_dep[upper:lower]], axis=0)
                only_dep = cat_dep[upper:lower]
                imwrite('My_Test_Result/Combine%.3d.jpg'%count, (big*255).astype(np.uint8))
                imwrite('My_Test_Result/Depth%.3d.jpg'%count, (only_dep*255).astype(np.uint8))
                d = {
                            'RGB': cat_rgb,
                            'depth': d_tmp[i, 0, ...]
                        }
                np.save('My_Test_Result/Data%.3d.npy'%count, d)

                count += 1

def main():
    test_img = MyData(args.path)
    print('Test Data Num:', len(test_img))
    dataset_val = DataLoader(
            test_img,
            batch_size=1,
            num_workers=2,
            drop_last=False,
            pin_memory=True,
            shuffle=False
            )

    saver = Utils.ModelSaver('./save')
    from models.FCRN import MyModel as ResNet
    model = ResNet(
    		layers=50,
    		decoder="upproj",
    		output_size=None,
    		in_channels=3,
    		pretrained=True
    		).cuda()

    saver.LoadLatestModel(model, None)
    Run(dataset_val, model, not args.nocrop)

if __name__ == '__main__':
    main()
