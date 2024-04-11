import torch
from torch.utils.data import DataLoader, Dataset

from PIL import Image
import numpy as np
from clothing_segmentation import HumanParser
import pandas as pd

from torchvision import transforms as T



class SyntheticTryonDataset(Dataset):
    def __init__(self, num_samples, image_size=(64,64), pose_size=(18, 2)):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            image_size (tuple): The height and width of the images (height, width).
            pose_size (tuple): The size of the pose tensors (default: (18, 2)).
        """
        self.human_parser = HumanParser()
        self.transform = T.Compose([
            # T.Resize(image_size),
            # T.CenterCrop(image_size),
            T.ToTensor(),
        ])
                
        self.df = pd.read_csv('/home/roman/tryondiffusion_implementation/tryondiffusion_danny/all_imgs.csv')
        # self.item_ids = np.unique(self.df['item_idx'].values)
        self.items_reverse_index = {}
        for group_idx, group in self.df.groupby(by='item_idx'):
            self.items_reverse_index[group_idx] = group['fullpath'].values        
        
        self.num_samples = num_samples
        self.image_size = image_size
        self.pose_size = pose_size

    def __len__(self):
        return self.num_samples
    
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
        
    def prepare_pose(self, img):
        # pass
        pose = torch.randn(*self.pose_size)
        return pose


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
        
        if idx in self.items_reverse_index:
            items_images_list = self.items_reverse_index[idx]
        else:
            items_images_list = self.items_reverse_index[idx-1]
        img_person = items_images_list[0]
        img_garment = items_images_list[1]
        
        # inputs from img1
        # noisy
        # clothing agnostic
        # person pose
        # img = img.resize((768, 768), Image.BICUBIC)
        person_image = Image.open(img_person).convert('RGB').resize((768, 768), Image.BICUBIC)
        # print(person_image.size)
        # .resize(self.image_size, Image.BICUBIC)
        np_person_image = np.array(person_image)
        person_image_resized = person_image.resize(self.image_size, Image.BICUBIC)
        person_image_hp = self.human_parser.forward_img(person_image).squeeze(0)
        
        ca_image = self.prepare_clothing_agnostic(np_person_image, person_image_hp)
        person_pose = self.prepare_pose(person_image_resized)
        
        # inputs from img2
        # garment pose
        # segmented garmend
        garment_image = Image.open(img_garment).convert('RGB').resize((768, 768), Image.BICUBIC)
        # print(garment_image.size)
        # .resize(self.image_size, Image.BICUBIC)
        np_garment_image = np.array(garment_image)
        garment_image_hp = self.human_parser.forward_img(garment_image).squeeze(0)
        
        segmented_garment = self.prepare_segmented_garment(np_garment_image, garment_image_hp) 
        garment_pose = self.prepare_pose(garment_image.resize(self.image_size, Image.BICUBIC))
        
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

        # TODO transforms PIL Images to Tensors, Normalization
        sample = {
            "person_images": person_image_resized,
            "ca_images": Image.fromarray(ca_image.astype('uint8')).resize(self.image_size, Image.BICUBIC),
            "garment_images": Image.fromarray(segmented_garment.astype('uint8')).resize(self.image_size, Image.BICUBIC),
            "person_poses": person_pose,
            "garment_poses": garment_pose,
        }
        
        sample = {
            "person_images": self.transform(sample['person_images']),
            "ca_images": self.transform(sample['ca_images']),
            "garment_images": self.transform(sample['garment_images']),
            "person_poses": sample['person_poses'],
            "garment_poses": sample['garment_poses']
        }
        
        return sample

def tryondiffusion_collate_fn(batch):
    return {
        "person_images": torch.stack([item["person_images"] for item in batch]),
        "ca_images": torch.stack([item["ca_images"] for item in batch]),
        "garment_images": torch.stack([item["garment_images"] for item in batch]),
        "person_poses": torch.stack([item["person_poses"] for item in batch]),
        "garment_poses": torch.stack([item["garment_poses"] for item in batch]),
    }