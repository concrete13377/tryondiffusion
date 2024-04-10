# import glob
# import os
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from torchvision import transforms
# from tqdm import tqdm

# from .utils import NormalizeImage
# from .u2net_cloth_segm import U2NET
# from collections import OrderedDict


# def load_cloth_segm_model(device, checkpoint_path, in_ch=3, out_ch=1):
#     if not os.path.exists(checkpoint_path):
#         print("Invalid path")
#         return

#     model = U2NET(in_ch=in_ch, out_ch=out_ch)

#     model_state_dict = torch.load(checkpoint_path, map_location=device)
#     new_state_dict = OrderedDict()
#     for k, v in model_state_dict.items():
#         name = k[7:]  # remove `module.`
#         new_state_dict[name] = v

#     model.load_state_dict(new_state_dict)
#     model = model.to(device=device)

#     print("Checkpoints loaded from path: {}".format(checkpoint_path))

#     return model


# class ClothSegmentot:
#     def __init__(self):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         segm_model = '/home/roman/tryondiffusion_implementation/huggingface-cloth-segmentation/model/cloth_segm.pth'
#         self.net = load_cloth_segm_model(self.device, segm_model, in_ch=3, out_ch=4)
#         self.transform_fn = transforms.Compose(
#             [transforms.ToTensor(),
#             NormalizeImage(0.5, 0.5)]
#         )


#         # TODO batched forward
#     def forward_image(self, img):
#         # passing images after # img = Image.open(os.path.join(inputs_dir, image_name)).convert('RGB')
#         image_tensor = self.transform_fn(img)
#         image_tensor = torch.unsqueeze(image_tensor, 0)
        
#         with torch.no_grad():
#             output_tensor = self.net(image_tensor.to(self.device))
#             output_tensor = F.log_softmax(output_tensor[0], dim=1)
#             output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
#             output_tensor = torch.squeeze(output_tensor, dim=0)
#             output_arr = output_tensor.cpu().numpy()

#         if cls == "all":
#             classes_to_save = []

#             # Check which classes are present in the image
#             for cls_inner in range(1, 4):  # Exclude background class (0)
#                 if np.any(output_arr == cls_inner):
#                     classes_to_save.append(cls_inner)
#         elif cls == "upper":
#             classes_to_save = [1]
#         elif cls == "lower":
#             classes_to_save = [2]
#         else:
#             raise ValueError(f"Unknown cls: {cls}")

#         for cls1 in classes_to_save:
#             alpha_mask = (output_arr == cls1).astype(np.uint8) * 255
#             alpha_mask = alpha_mask[0]  # Selecting the first channel to make it 2D
#             alpha_mask_img = Image.fromarray(alpha_mask, mode='L')
#             alpha_mask_img = alpha_mask_img.resize(img_size, Image.BICUBIC)
#             alpha_mask_img.save(os.path.join(outputs_dir, f'{image_name.split(".")[0]}_{cls1}.jpg'))

#         pbar.update(1)

#     pbar.close()
        