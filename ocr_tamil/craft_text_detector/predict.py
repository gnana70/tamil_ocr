import os
import time
import torch

import cv2
import numpy as np

# import craft_text_detector.craft_utils as craft_utils
# import craft_text_detector.image_utils as image_utils
# import craft_text_detector.torch_utils as torch_utils

from . import craft_utils as craft_utils
from . import image_utils as image_utils
from . import torch_utils as torch_utils

from torchvision.transforms import functional as F
from torchvision.transforms import v2 as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import cv2
import numpy as np


class CraftDataset(Dataset):
    """
    
    Craft Dataset loader

    Args:
        Dataset (list): List of Images
    """
    def __init__(self, data, resize=[960,1080]):
        self.data = data
        self.resize = resize
        transforms = []
        transforms.extend([T.Resize(resize, T.InterpolationMode.BICUBIC),
                    T.ToImage(), 
                    T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform = T.Compose(transforms)
        
    def __getitem__(self, index):
        x = self.data[index]
        height, width, _ = x.shape
        x = Image.fromarray(np.uint8(x)).convert('RGB')
        x = self.transform(x)

        w_ratio = self.resize[0] / width
        h_ratio = self.resize[1] / width

        return x,w_ratio,h_ratio
    
    def __len__(self):
        return len(self.data)


def get_prediction(
    image,
    craft_net,
    refine_net=None,
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
    cuda: bool = False,
    long_size: int = 1280,
    poly: bool = True,
    half: bool = False
):
    """
    Arguments:
        image: path to the image to be processed or numpy array or PIL image
        output_dir: path to the results to be exported
        craft_net: craft net model
        refine_net: refine net model
        text_threshold: text confidence threshold
        link_threshold: link confidence threshold
        low_text: text low-bound score
        cuda: Use cuda for inference
        canvas_size: image size for inference
        long_size: desired longest image size for inference
        poly: enable polygon type
    Output:
        {"masks": lists of predicted masks 2d as bool array,
         "boxes": list of coords of points of predicted boxes,
         "boxes_as_ratios": list of coords of points of predicted boxes as ratios of image size,
         "polys_as_ratios": list of coords of points of predicted polys as ratios of image size,
         "heatmaps": visualizations of the detected characters/links,
         "times": elapsed times of the sub modules, in seconds}
    """
    if isinstance(long_size,int):

        # resize
        img_resized, target_ratio, size_heatmap = image_utils.resize_aspect_ratio(
            image, long_size, interpolation=cv2.INTER_LINEAR
        )
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = image_utils.normalizeMeanVariance(img_resized)
        x = torch_utils.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = torch_utils.Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if cuda:
            if half:
                x = x.cuda().half()
            else:
                x = x.cuda()

        # forward pass
        with torch.inference_mode():
            y, _ = craft_net(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy().astype(np.float32)
        score_link = y[0, :, :, 1].cpu().data.numpy().astype(np.float32)


        # Post-processing
        boxes = craft_utils.getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly
        )

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    else:
        pass

    return boxes
