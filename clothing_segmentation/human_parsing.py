from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import torch
from torch import nn
import numpy as np

# model_path = '/mnt/datadrive/hf_models/matei-dorian/segformer-b5-finetuned-human-parsing/'
# image_processor = AutoImageProcessor.from_pretrained(model_path)
# model = SegformerForSemanticSegmentation.from_pretrained(model_path)

# image_path = "/home/roman/tryondiffusion_implementation/tryondiffusion_tryonlabs/tryon/image.png"
# image = Image.open(image_path).convert("RGB") 


class HumanParser:
    def __init__(self):
        self.model_path = '/mnt/datadrive/hf_models/matei-dorian/segformer-b5-finetuned-human-parsing/'
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_path)

    def forward_img(self, image:Image):    
        # image = Image.open(image_path).convert("RGB")
        inputs = self.image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
        # print(f'{logits.shape=}')
        # print(f'{image.size}')
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1], # (height, width)
            mode='bilinear',
            align_corners=False
        )
        # print(f'{upsampled_logits.shape=}')
        
        pred_seg = upsampled_logits.argmax(dim=1)
        return pred_seg