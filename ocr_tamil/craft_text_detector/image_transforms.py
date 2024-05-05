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

if __name__ == "__main__":
    import os
    base_path = r"C:\Users\gnana\Documents\GitHub\tamil_ocr\temp_images"
    image_paths = [os.path.join(base_path,str(i)+".jpg") for i in range(10)]
    image_list = [cv2.imread(i) for i in image_paths]
    dataset = CraftDataset(image_list)
    dataloader = DataLoader(dataset, batch_size=4)

    for data in dataloader:
        print(data[0].shape,data[1])
        break

    




    