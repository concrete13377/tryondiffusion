import pickle 

import torch
from torch.utils.data import DataLoader, Dataset

from pathlib import Path
from PIL import Image
import numpy as np
from clothing_segmentation import HumanParser
import pandas as pd

from torchvision import transforms as T


from controlnet_aux import OpenposeDetector
from controlnet_aux.open_pose import HWC3, resize_image
from diffusers.utils import load_image



class MyOpenPoseDetector(OpenposeDetector):
    
    def __call__(self, input_image, detect_resolution=512, include_hand=False, include_face=False):
       
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        poses = self.detect_poses(input_image, include_hand, include_face)
        xy=[]
        if len(poses)>0:
            for point in poses[0].body.keypoints:
                if point is not None:
                    xy.append([point.x, point.y])
                else:
                    xy.append([0, 0])
            return np.array(xy)
        else:
            return None
        

class SyntheticTryonDataset(Dataset):
    def __init__(self, image_size=(64,64), pose_size=(18, 2), apply_transform=True):

        self.human_parser = HumanParser()
        self.transform = T.Compose([
            # T.Resize(image_size),
            # T.CenterCrop(image_size),
            T.ToTensor(),
        ])
        self.apply_transform = apply_transform
        self.df = pd.read_csv('/home/roman/tryondiffusion_implementation/tryondiffusion_danny/all_imgs.csv')
        self.df = self.df[~self.df['pose_is_none']]
        
        self.items_reverse_index = {}
        self.items_reverse_index_poses = {}
        for group_idx, group in self.df.groupby(by='item_idx'):
            self.items_reverse_index[group_idx] = group['fullpath'].values        
            self.items_reverse_index_poses[group_idx] = group['pose_512'].values        
        
        self.image_size = image_size
        # self.pose_size = pose_size
        self.openpose = MyOpenPoseDetector.from_pretrained("lllyasviel/ControlNet")

        
    def __len__(self):
        return len(self.items_reverse_index)
    
    def prepare_clothing_agnostic(self, img, hp_mask)-> np.array: 
        # classes_to_rm=[4,6] 
        classes_to_rm=[4]      
        # def get_clothing_agnostic(image, hp_mask, classes_to_rm=[4,6]):
        bg_color = (255,255,255)
        assert img.shape[:-1] == hp_mask.shape
        # cloths_to_rm_mask = np.zeros(hp_mask.shape)
        # for i in np.unique(res):
        #     if i in classes_to_rm:
        #         cloths_to_rm_mask[res==i] = 255
        cloths_to_rm_mask = np.isin(hp_mask, classes_to_rm)
        img[cloths_to_rm_mask!=0] = bg_color
        return img
        
    def prepare_segmented_garment(self, img, hp_mask)-> np.array:
        # classes_to_rm=[4,6] 
        classes_to_rm=[4]        
        bg_color = (255,255,255)
        assert img.shape[:-1] == hp_mask.shape
        # cloths_to_rm_mask = np.zeros(hp_mask.shape)
        # for i in np.unique(res):
        #     if i in classes_to_rm:
        #         cloths_to_rm_mask[res==i] = 255
        cloths_to_rm_mask = np.isin(hp_mask, classes_to_rm)
        img[cloths_to_rm_mask==0] = bg_color
        return img

    def __getitem__(self, idx):
        if idx not in self.items_reverse_index:
            return self.__getitem__(idx-1)    
        items_images_list = self.items_reverse_index[idx]
        # items_images_list_poses = self.items_reverse_index_poses[idx]
        
        if len(items_images_list)<2:
            return self.__getitem__(idx-1)
        else:
            img_person = items_images_list[0]
            # person_pose = items_images_list_poses[0]
            # person_pose = pickle.loads(items_images_list_poses[0])
            
            img_garment = items_images_list[1]
            # garment_pose = items_images_list_poses[1]
            # garment_pose = pickle.loads(items_images_list_poses[1])
        
        # inputs from img1
        # noisy
        # clothing agnostic
        # person pose
        # img = img.resize((768, 768), Image.BICUBIC)
        person_image = Image.open(img_person).convert('RGB').resize((768, 768), Image.BICUBIC)
        op_img = load_image(img_person)
        person_pose = self.openpose(op_img)
        # print(person_image.size)
        # .resize(self.image_size, Image.BICUBIC)
        np_person_image = np.array(person_image)
        person_image_resized = person_image.resize(self.image_size, Image.BICUBIC)
        person_image_hp = self.human_parser.forward_img(person_image).squeeze(0)
        
        ca_image = self.prepare_clothing_agnostic(np_person_image, person_image_hp)
        
        # inputs from img2
        # garment pose
        # segmented garmend
        garment_image = Image.open(img_garment).convert('RGB').resize((768, 768), Image.BICUBIC)
        op_img = load_image(img_garment)
        garment_pose = self.openpose(op_img)
        # print(garment_image.size)
        # .resize(self.image_size, Image.BICUBIC)
        np_garment_image = np.array(garment_image)
        garment_image_hp = self.human_parser.forward_img(garment_image).squeeze(0)
        
        segmented_garment = self.prepare_segmented_garment(np_garment_image, garment_image_hp) 
        
        # person_image = torch.randn(3, *self.image_size)
        # ca_image = torch.randn(3, *self.image_size)
        # garment_image = torch.randn(3, *self.image_size)
        # person_pose = torch.randn(*self.pose_size)
        # garment_pose = torch.randn(*self.pose_size)

        # sample = {
        #     "person_images": person_image,
        #     "ca_images": ca_image,
        #     "garment_images": garment_image,
        #     "person_poses": person_pose,
        #     "garment_poses": garment_pose,
        # }

        sample = {
            "person_images": person_image_resized,
            "ca_images": Image.fromarray(ca_image.astype('uint8')).resize(self.image_size, Image.BICUBIC),
            "garment_images": Image.fromarray(segmented_garment.astype('uint8')).resize(self.image_size, Image.BICUBIC),
            "person_poses": person_pose,
            "garment_poses": garment_pose,
        }
        
        if self.apply_transform:
            sample = {
                "person_images": self.transform(sample['person_images']),
                "ca_images": self.transform(sample['ca_images']),
                "garment_images": self.transform(sample['garment_images']),
                "person_poses": sample['person_poses'],
                "garment_poses": sample['garment_poses']
            }
        
        return sample
    

class SyntheticTryonDatasetFromDisk(Dataset):
    def __init__(self, max_imgs, path='/mnt/datadrive/asos_dataset/prepared_64/tensors/'):
        self.path = Path(path)
        self.glob=sorted(self.path.rglob('*.pt'), key=lambda x: int(x.stem))
        self.glob = self.glob[:max_imgs]
        
    def __len__(self):
        return len(self.glob)
    
    def __getitem__(self, idx):
        tensor_path = self.glob[idx]
        record = torch.load(tensor_path)
        newrecord = {}
        for k,v in record.items():
            if 'pose' in k:
                newrecord[k] = torch.tensor(v, dtype=torch.float)
            else: 
                newrecord[k]=v
        # for k,v in newrecord.items():
        #     print(k, type(v), v.dtype)
        return newrecord
    

def tryondiffusion_collate_fn(batch):
    return {
        "person_images": torch.stack([item["person_images"] for item in batch]),
        "ca_images": torch.stack([item["ca_images"] for item in batch]),
        "garment_images": torch.stack([item["garment_images"] for item in batch]),
        "person_poses": torch.stack([item["person_poses"] for item in batch]),
        "garment_poses": torch.stack([item["garment_poses"] for item in batch]),
    }