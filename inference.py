# import craft functions
import sys
import os

from craft_text_detector import (
    load_craftnet_model,
    get_prediction,
    export_detected_regions,
    empty_cuda_cache
)
import uuid
import torch
import cv2
import numpy as np

# import related to parseq
import re
from abc import ABC, abstractmethod
from itertools import groupby
from typing import List, Optional, Tuple

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from PIL import Image
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
import matplotlib.pyplot as plt

import skimage
from ocr_utils import tamil_character_to_id,id_to_tamil_character

import warnings
warnings.filterwarnings("ignore")

class CharsetAdapter:
    """Transforms labels according to the target charset."""

    def __init__(self, target_charset) -> None:
        super().__init__()
        self.charset = target_charset ###
        self.lowercase_only = target_charset == target_charset.lower()
        self.uppercase_only = target_charset == target_charset.upper()
#         self.unsupported = f'[^{re.escape(target_charset)}]'

    def __call__(self, label):
        if self.lowercase_only:
            label = label.lower()
        elif self.uppercase_only:
            label = label.upper()
        return label


class BaseTokenizer(ABC):

    def __init__(self, charset: str, specials_first: tuple = (), specials_last: tuple = ()) -> None:
        self._itos = specials_first + tuple(charset+'[UNK]') + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> List[int]:
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids: List[int], join: bool = True) -> str:
        tokens = [self._itos[i] for i in token_ids]
        return ''.join(tokens) if join else tokens

    @abstractmethod
    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        """Encode a batch of labels to a representation suitable for the model.
        Args:
            labels: List of labels. Each can be of arbitrary length.
            device: Create tensor on this device.
        Returns:
            Batched tensor representation padded to the max label length. Shape: N, L
        """
        raise NotImplementedError

    @abstractmethod
    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        """Internal method which performs the necessary filtering prior to decoding."""
        raise NotImplementedError

    def decode(self, token_dists: Tensor, raw: bool = False) -> Tuple[List[str], List[Tensor]]:
        """Decode a batch of token distributions.
        Args:
            token_dists: softmax probabilities over the token distribution. Shape: N, L, C
            raw: return unprocessed labels (will return list of list of strings)
        Returns:
            list of string labels (arbitrary length) and
            their corresponding sequence probabilities as a list of Tensors
        """
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)  # greedy selection
            if not raw:
                probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs


class Tokenizer(BaseTokenizer):
    BOS = '[B]'
    EOS = '[E]'
    PAD = '[P]'

    def __init__(self, charset: str) -> None:
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        super().__init__(charset, specials_first, specials_last)
        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in specials_first + specials_last]

    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        batch = [torch.as_tensor([self.bos_id] + self._tok2ids(y) + [self.eos_id], dtype=torch.long, device=device)
                 for y in labels]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)  # Nothing to truncate.
        # Truncate after EOS
        ids = ids[:eos_idx]
        probs = probs[:eos_idx + 1]  # but include prob. for EOS (if it exists)
        return probs, ids


class CTCTokenizer(BaseTokenizer):
    BLANK = '[B]'

    def __init__(self, charset: str) -> None:
        # BLANK uses index == 0 by default
        super().__init__(charset, specials_first=(self.BLANK,))
        self.blank_id = self._stoi[self.BLANK]

    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        # We use a padded representation since we don't want to use CUDNN's CTC implementation
        batch = [torch.as_tensor(self._tok2ids(y), dtype=torch.long, device=device) for y in labels]
        return pad_sequence(batch, batch_first=True, padding_value=self.blank_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        # Best path decoding:
        ids = list(zip(*groupby(ids.tolist())))[0]  # Remove duplicate tokens
        ids = [x for x in ids if x != self.blank_id]  # Remove BLANKs
        # `probs` is just pass-through since all positions are considered part of the path
        return probs, ids
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()
output_dir = "temp_images"

gpu=False
if torch.cuda.is_available():# load models
    gpu=True
    craft_net = load_craftnet_model(cuda=True,weight_path=os.path.join("model_weights","craft_mlt_25k.pth"))
else:
    craft_net = load_craftnet_model(cuda=False,weight_path=os.path.join("model_weights","craft_mlt_25k.pth"))

def sort_bboxes(contours):
    c = np.array(contours)
    max_height = np.median(c[::, 3]) * 0.5
    
    # Sort the contours by y-value
    by_y = sorted(contours, key=lambda x: x[1])  # y values
    
    line_y = by_y[0][1]       # first y
    line = 1
    by_line = []
    
    # Assign a line number to each contour
    for x, y, w, h in by_y:
        if y > line_y + max_height:
            line_y = y
            line += 1
            
        by_line.append((line, x, y, w, h))
    
    # This will now sort automatically by line then by x
    contours_sorted = [[x, y, w, h] for line, x, y, w, h in sorted(by_line)]

    return contours_sorted
    

def craft_detect(image,text_threshold=0.1,link_threshold=0.2,low_text=0.1,**kwargs):
    size = max(image.shape[0],image.shape[1],1280)
    size = min(size,2560)
    
    # perform prediction
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        text_threshold=text_threshold,
        link_threshold=link_threshold,
        low_text=low_text,
        cuda=gpu,
        long_size=size,
        poly = False
    )

    # print(prediction_result)

    new_bbox = []

    for bb in prediction_result:
        xs = bb[:,0]
        ys = bb[:,1]

        min_x,max_x = min(xs),max(xs)
        min_y,max_y = min(ys),max(ys)
        x,y,w,h = min_x,min_y,max_x-min_x, max_y-min_y
        # if w>0 and h>0:
        new_bbox.append([x,y,w,h])

    ordered_new_bbox = sort_bboxes(new_bbox)

    # print(new_bbox)
    # print("*"*10)
    # print(ordered_new_bbox)

    # index_list = []
    updated_prediction_result = []
    for ordered_bbox in ordered_new_bbox:
        index_val = new_bbox.index(ordered_bbox)
        # index_list.append(index_val)
        updated_prediction_result.append(prediction_result[index_val])
        
    # print(index_list)

    # export detected text regions
    exported_file_paths = export_detected_regions(
        image=image,
        regions=updated_prediction_result ,#["boxes"],
        output_dir=output_dir,
        rectify=False
    )

    torch.cuda.empty_cache()

    return exported_file_paths

def decode_file_name(decode_text,id_to_tamil_character,special_sep_char="~"):
    individual_ids = decode_text.split(f"{special_sep_char}")
    k = id_to_tamil_character.keys()
    tamil_chars = [id_to_tamil_character[i] for i in individual_ids if i in k]
    tamil_word ="".join(tamil_chars)
    return tamil_word
    
# Load model and image transforms
ckpt_path = os.path.join("model_weights","parseq_tamil_v4.ckpt")
parseq = load_from_checkpoint(ckpt_path).to('cpu').eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
eng_parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()

def read_image_input(image):
    if type(image) == str:
        img = cv2.imread(image)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif type(image) == bytes:
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif type(image) == np.ndarray:
        if len(image.shape) == 2:  # grayscale
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            img = image
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBAscale
            img = image[:, :, :3]
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return img


def ocr_predict(image,show_image=False,**kwargs):
# image_path = 'test_images/tamil_text.jpg' # can be filepath, PIL image or numpy array
# # image_path = 'Rationcard.jpg'
# image = cv2.imread(image_path)

    image = read_image_input(image)
    exported_regions = craft_detect(image,**kwargs)

    text = ""
    
    for img_org in exported_regions:
        # img_org = cv2.imread(file)
        img_org = skimage.exposure.rescale_intensity(img_org, in_range='image', out_range='dtype')
        
        img_org = Image.fromarray(np.uint8(img_org)).convert('RGB')
        img = img_transform(img_org.convert('RGB')).unsqueeze(0)
    
        # tamil decode
        logits = parseq(img)
        # Greedy decoding
        pred = logits.softmax(-1)
        label, confidence = parseq.tokenizer.decode(pred)
        label = decode_file_name(label[0],id_to_tamil_character)
    
        # english decode
        logits = eng_parseq(img)
        pred = logits.softmax(-1)
        eng_label, eng_confidence = eng_parseq.tokenizer.decode(pred)
    
        avg_eng_conf = sum(eng_confidence[0].detach().numpy())/len(eng_confidence[0].detach().numpy())
        avg_tam_conf = sum(confidence[0].detach().numpy())/len(confidence[0].detach().numpy())
    
        if avg_eng_conf > avg_tam_conf:
            label = eng_label[0]

        text += label + " "

        if show_image:
            plt.imshow(img_org)
            plt.show()
            
            print(avg_eng_conf)
            print(avg_tam_conf)
            print('Decoded label = {}'.format(label))

    return text[:-1]

if __name__ == "__main__":
    image_path = r"test_images\signboard_2.jpg"
    texts = ocr_predict(image_path)
    with open("output.txt","w",encoding="utf-8") as f:
        f.write(texts)