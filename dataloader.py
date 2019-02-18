import os

import torch
import torch.utils.data as data

from PIL import Image

class ImageData(data.Dataset):
    
    def __init__(self, root_origin, root_pose, transform=None, loader=None):
        def convert_str2int(path):
            li_int = []
            for filename in os.listdir(path):
                if filename[0] == '0':
                    filename = filename[1:]

                if filename.endswith('.png'):
                    li_int.append(int(filename.split('.')[0]))

            return sorted(li_int)
        
        li_origin = convert_str2int(root_origin)
        li_pose = convert_str2int(root_pose)
        
        #assert len(li_origin) == len(li_pose)
        
        self.origin_set = [root_origin + '/' + str(file_origin) + '.png' for file_origin in li_origin]
        self.pose_set = [root_pose + '/' + str(file_pose) + '.png' for file_pose in li_pose]
        
        self.transform = transform
        
        return None
        
    def __getitem__(self, index):
        path_img_origin_pre = self.origin_set[index]
        path_img_pose_pre = self.pose_set[index]
        
        path_img_origin_cur = self.origin_set[index+1]
        path_img_pose_cur = self.pose_set[index+1]
        
        img_origin_pre = self.transform(Image.open(path_img_origin_pre).convert('RGB'))
        img_pose_pre = self.transform(Image.open(path_img_pose_pre).convert('RGB'))

        img_origin_cur = self.transform(Image.open(path_img_origin_cur).convert('RGB'))
        img_pose_cur = self.transform(Image.open(path_img_pose_cur).convert('RGB'))
        
        return torch.cat([img_origin_pre, img_pose_pre]),  torch.cat([img_origin_cur, img_pose_cur])
    
    def __len__(self):
        return len(self.origin_set)