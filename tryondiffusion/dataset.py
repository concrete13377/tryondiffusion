import pickle 

import cv2
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
    
    def __call__(self, input_image, detect_resolution=512, include_hand=True, include_face=False):
       
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
            body = np.array(xy)
            return body

            # xy=[]
            # for point in poses[0].right_hand:
            #     if point is not None:
            #         xy.append([point.x, point.y])
            #     else:
            #         xy.append([0, 0])
            # right_hand = np.array(xy)


            # xy=[]
            # for point in poses[0].left_hand:
            #     if point is not None:
            #         xy.append([point.x, point.y])
            #     else:
            #         xy.append([0, 0])
            # left_hand = np.array(xy)
            # return body, right_hand, left_hand

        else:
            return None
        

import itertools
class SyntheticTryonDataset(Dataset):
    def __init__(self, image_size=(64,64), pose_size=(18, 2), apply_transform=True):
        self.cache_garment = {}
        self.cache_person = {}
        self.human_parser = HumanParser()
        self.transform = T.Compose([
            # T.Resize(image_size),
            # T.CenterCrop(image_size),
            T.ToTensor(),
        ])
        self.apply_transform = apply_transform
        # self.df = pd.read_csv('/home/roman/tryondiffusion_implementation/tryondiffusion_danny/all_imgs.csv')
        # self.df = pd.read_csv('/home/roman/tryondiffusion_implementation/tryondiffusion_danny/all_imgs_clean_80756_total.csv')
        # self.df = pd.read_csv('/home/roman/tryondiffusion_implementation/tryondiffusion_danny/good_fronts.csv')
        # self.df = pd.read_csv('/home/roman/tryondiffusion_implementation/tryondiffusion_danny/turboturbo.csv')
        self.df = pd.read_csv('/home/roman/tryondiffusion_implementation/tryondiffusion_danny/fronted_filtered.csv')
        

        # print(len(self.df))
        # self.df = self.df[~self.df['pose_is_none']]
        print(len(self.df))
        self.items_reverse_index = {}
        self.items_reverse_index_poses = {}
        for group_idx, group in self.df.groupby(by='item_idx'):
            # self.items_reverse_index[group_idx] = [{"fullpath":fp, "pose_512":pose} for fp, pose in zip(group['fullpath'].values, group['pose_512'].values)]        
            self.items_reverse_index[group_idx] = [{"fullpath":fp} for fp in group['fullpath'].values]        

        self.items_reverse_index3 = {}
        key_idx=0
        for k,v in self.items_reverse_index.items():
            permutations = list(itertools.permutations(v, 2))
            for i in permutations:
                self.items_reverse_index3[key_idx] = i
                key_idx+=1

        print(len(self.items_reverse_index3))
        self.image_size = image_size
        # self.pose_size = pose_size
        self.openpose = MyOpenPoseDetector.from_pretrained("lllyasviel/ControlNet")
        self.openpose.to('cuda')

        
    def __len__(self):
        return len(self.items_reverse_index3)
    

    def get_hand(self, person_image_hp, person_pose, clas=14, c1=6, c2=7):

        kist2 = person_pose[c2]
        lokot2 = person_pose[c1]

        x_e, y_e = lokot2
        x_w, y_w = kist2
        elbow = (int(x_e*768), int(y_e*768))
        wrist = (int(x_w*768), int(y_w*768))

        x_e, y_e = elbow
        x_w, y_w = wrist

        arm_mask = (person_image_hp==clas).cpu().numpy().astype(np.uint8)

        vector = np.array([x_w - x_e, y_w - y_e])

        # Calculate the perpendicular vector (90 degrees rotation)
        perpendicular_vector = np.array([-vector[1], vector[0]])

        # Normalize the perpendicular vector
        perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)

        # TODO % 
        perpendicular_length_base = 50  
        perpendicular_length_top = 200  
        extension_length = 300  

        perpendicular_start_base = np.array([x_w, y_w]) + perpendicular_length_base * perpendicular_vector
        perpendicular_end_base = np.array([x_w, y_w]) - perpendicular_length_base * perpendicular_vector

        extended_wrist_point = np.array([x_w, y_w]) + extension_length * vector / np.linalg.norm(vector)

        perpendicular_start_top = extended_wrist_point + perpendicular_length_top * perpendicular_vector
        perpendicular_end_top = extended_wrist_point - perpendicular_length_top * perpendicular_vector

        perpendicular_start_base = perpendicular_start_base.astype(int)
        perpendicular_end_base = perpendicular_end_base.astype(int)
        perpendicular_start_top = perpendicular_start_top.astype(int)
        perpendicular_end_top = perpendicular_end_top.astype(int)
        extended_wrist_point = extended_wrist_point.astype(int)

        wrist_mask = np.zeros_like(arm_mask)

        polygon_points = np.array([
            perpendicular_start_base,
            perpendicular_end_base,
            perpendicular_end_top,
            perpendicular_start_top
        ])


        # Draw the polygon on the wrist mask
        cv2.fillPoly(wrist_mask, [polygon_points.astype(int)], 255)

        # Combine the wrist mask with the arm mask to get the wrist region
        wrist_region = cv2.bitwise_and(arm_mask, wrist_mask)

        # plt.imshow(arm_mask)
        # plt.show()
        # plt.imshow(wrist_mask)
        # plt.show()
        # plt.imshow(wrist_region)
        # plt.show()

        return wrist_region, person_image_hp



    def get_img(self, img_person):
        person_image = Image.open(img_person).convert('RGB').resize((768, 768), Image.BICUBIC)
        person_image_hp = self.human_parser.forward_img(person_image).squeeze(0)

        op_img = load_image(img_person)
        person_pose = self.openpose(op_img, include_hand=True)

        h1, hp = self.get_hand(person_image_hp, person_pose,  15, 3, 4)
        h2, hp = self.get_hand(person_image_hp, person_pose, 14, 6, 7)

        total = cv2.bitwise_or(h1, h2)
        masksadd=[]
        for i in [2,11,6]:
            newmask = (person_image_hp==i).cpu().numpy().astype(np.uint8)
            total = cv2.bitwise_or(newmask, total)

        # contours, hierarchy = cv2.findContours((person_image_hp==7).numpy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # all_contours = np.vstack(contours)
        # bounding_rect_image = np.zeros((person_image_hp.shape))
        
        # x, y, w, h = cv2.boundingRect(all_contours)
        # cv2.rectangle(bounding_rect_image, (x, y), (x + w, y + h), (255, 255, 255), -1)
        binary_mask = (person_image_hp == 7).cpu().numpy().astype(np.uint8)
        binary_mask2 = (person_image_hp == 4).cpu().numpy().astype(np.uint8)
        binary_mask = cv2.bitwise_or(binary_mask, binary_mask2)
        coords = np.column_stack(np.where(binary_mask > 0))
        
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
        else:
            y_min, x_min, y_max, x_max = 0, 0, 0, 0  # Handle case with no coordinates

        
        bounding_rect_image = np.zeros(person_image_hp.shape, dtype=np.uint8)
        cv2.rectangle(bounding_rect_image, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)
        
        
        total = total.astype(np.uint8)
        bounding_rect_image=bounding_rect_image.astype(np.uint8)
        diff = np.where(total==1, 0, bounding_rect_image)
        xxxxx = cv2.bitwise_and( np.array(person_image),  np.array(person_image), mask=~diff)

        gray_background = np.full(np.array(person_image).shape, 128, dtype=np.uint8)  # 128 for gray color
        gray_background_masked = cv2.bitwise_and(gray_background, gray_background, mask=diff)
        final_image = cv2.add(xxxxx, gray_background_masked)

        return final_image 

    def prepare_clothing_agnostic(self, img_person)-> np.array: 
        # # classes_to_rm=[4,6] 
        # classes_to_rm=[4]      
        # # def get_clothing_agnostic(image, hp_mask, classes_to_rm=[4,6]):
        # bg_color = (255,255,255)
        # assert img.shape[:-1] == hp_mask.shape
        # # cloths_to_rm_mask = np.zeros(hp_mask.shape)
        # # for i in np.unique(res):
        # #     if i in classes_to_rm:
        # #         cloths_to_rm_mask[res==i] = 255
        # cloths_to_rm_mask = np.isin(hp_mask, classes_to_rm)
        # img[cloths_to_rm_mask!=0] = bg_color
        # return img
        res = self.get_img(img_person)
        return res

    def prepare_segmented_garment(self, img, hp_mask)-> np.array:
        # classes_to_rm=[4,6] 
        classes_to_rm=[4, 7]     
        bg_color = (255,255,255)
        assert img.shape[:-1] == hp_mask.shape
        # cloths_to_rm_mask = np.zeros(hp_mask.shape)
        # for i in np.unique(res):
        #     if i in classes_to_rm:
        #         cloths_to_rm_mask[res==i] = 255
        cloths_to_rm_mask = np.isin(hp_mask.cpu(), classes_to_rm)
        img[cloths_to_rm_mask==0] = bg_color
        return img

    def __getitem__(self, idx):
        item = self.items_reverse_index3[idx]
        img_person = item[0]['fullpath']
        person_dict = None
        if img_person in self.cache_person:
            person_dict = self.cache_person[img_person]
        # person_pose = item[0]['pose_512']
        # person_pose = pickle.loads(person_pose)

        img_garment = item[1]['fullpath']
        garment_dict = None
        if img_garment in self.cache_garment:
            garment_dict = self.cache_garment[img_garment]
        
        if person_dict is None:
            person_image = Image.open(img_person).convert('RGB').resize((768, 768), Image.BICUBIC)
            op_img = load_image(img_person)
            person_pose = self.openpose(op_img)
            
            # np_person_image = np.array(person_image)
            person_image_resized = person_image.resize(self.image_size, Image.BICUBIC)
            # person_image_hp = self.human_parser.forward_img(person_image).squeeze(0)
            
            # ca_image = self.prepare_clothing_agnostic(np_person_image, person_image_hp)
            ca_image = self.prepare_clothing_agnostic(img_person)

            person_dict = {
                "person_images": self.transform(person_image_resized),
                "ca_images": self.transform(Image.fromarray(ca_image.astype('uint8')).resize(self.image_size, Image.BICUBIC)),
                "person_poses": person_pose
            }
            self.cache_person[img_person] = person_dict
        
        if garment_dict is None:
            garment_image = Image.open(img_garment).convert('RGB').resize((768, 768), Image.BICUBIC)
            op_img = load_image(img_garment)
            garment_pose  = self.openpose(op_img)
            
            np_garment_image = np.array(garment_image)
            garment_image_hp = self.human_parser.forward_img(garment_image).squeeze(0)
            
            segmented_garment = self.prepare_segmented_garment(np_garment_image, garment_image_hp) 

            garment_dict = {
                "garment_images": self.transform(Image.fromarray(segmented_garment.astype('uint8')).resize(self.image_size, Image.BICUBIC)),
                "garment_poses": garment_pose    
            }
            self.cache_garment[img_garment] = garment_dict
        
        # if idx == 3:
        #     print(f'{person_dict=}')
        #     print(f'{garment_dict=}')
        # sample = {
            # "person_images": person_image_resized,
            # "ca_images": Image.fromarray(ca_image.astype('uint8')).resize(self.image_size, Image.BICUBIC),
            # "garment_images": Image.fromarray(segmented_garment.astype('uint8')).resize(self.image_size, Image.BICUBIC),
            # "person_poses": person_pose,
            # "garment_poses": garment_pose,
        # }
        sample = {
            **person_dict, **garment_dict
        }
        
        # if self.apply_transform:
        #     sample = {
        #         "person_images": self.transform(sample['person_images']),
        #         "ca_images": self.transform(sample['ca_images']),
        #         "garment_images": self.transform(sample['garment_images']),
        #         "person_poses": sample['person_poses'],
        #         "garment_poses": sample['garment_poses']
        #     }
        
        return sample


class SyntheticTryonDatasetFromDisk(Dataset):
    # def __init__(self, max_imgs, path='/mnt/datadrive/asos_dataset/prepared_256/tensors/'):
    # def __init__(self, max_imgs=None, path='/workdir/dataset/dress_code/tensors'):
    # def __init__(self, max_imgs=None, path='/mnt/datadrive/asos_dataset/80756_203404_new_algo/prepared_128/tensors_62kready'):
    def __init__(self, max_imgs=None, path='/mnt/datadrive/asos_dataset/20k_clean_fronts/prepared_128/tensors'):
        self.path = Path(path)
        self.glob=sorted(self.path.rglob('*.pt'), key=lambda x: int(x.stem))
        if max_imgs is not None:
            self.glob = self.glob[:max_imgs]
   
    def __len__(self):
        return len(self.glob)
    
    def __getitem__(self, idx):
        tensor_path = self.glob[idx]
        record = torch.load(tensor_path)
        newrecord = {}
        for k,v in record.items():
            if 'pose' in k:
                if isinstance(v, tuple):
                    v = v[0]
                newrecord[k] = torch.tensor(v, dtype=torch.float)
            else: 
                newrecord[k]=v
        # for k,v in newrecord.items():
        #     print(k, type(v), v.dtype)
        #     try:
        #         print(v.shape)
        #     except Exception as e:
        #         pass
        return newrecord
    

def tryondiffusion_collate_fn(batch):
    return {
        "person_images": torch.stack([item["person_images"] for item in batch]),
        "ca_images": torch.stack([item["ca_images"] for item in batch]),
        "garment_images": torch.stack([item["garment_images"] for item in batch]),
        "person_poses": torch.stack([item["person_poses"] for item in batch]),
        "garment_poses": torch.stack([item["garment_poses"] for item in batch]),
    }