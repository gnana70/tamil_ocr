from ocr_tamil.strhub.models.utils import load_from_checkpoint
from ocr_tamil.strhub.data.module import SceneTextDataModule
import onnx
import torch
from PIL import Image

# To ONNX
device = "cpu"
ckpt_path = r"ocr_tamil\model_weights\parseq_tamil_v6.ckpt"
onnx_path = r"ocr_tamil\model_weights\parseq_test_tamil.onnx"
img_path = r"test_images\4.jpg"


# parseq = load_from_checkpoint(ckpt_path)

parseq = torch.load(ckpt_path)
# parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
parseq.refine_iters = 0
parseq.decode_ar = False
parseq = parseq.to(device).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

img_org = Image.open(img_path)
# Preprocess. Model expects a batch of images with shape: (B, C, H, W)
img = img_transform(img_org.convert('RGB')).unsqueeze(0)

parseq.to_onnx(onnx_path, img, do_constant_folding=True, opset_version=14)  # opset v14 or newer is required

# check
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model, full_check=True) #==> pass

