import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_49.pth', type=str, help='generator model epoch name')
opt = parser.parse_args(args = [])

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
MODEL_NAME = opt.model_name

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

image = Image.open(r"D:\CODDING STUFF\Sem 6\CV_GAN\SRGAN\0a0d7a87378422e3_imresizer.jpg")
image = ToTensor()(image).unsqueeze(0)
if TEST_MODE:
    image = image.cuda()

# Use torch.no_grad() for inference
with torch.no_grad():
    out = model(image)

out_img = ToPILImage()(out[0].cpu())

# Handle NoneType for image_name argument
image_name = opt.image_name if opt.image_name is not None else "default_image"
#out_img.save(f'out_srf_{UPSCALE_FACTOR}_{image_name}.png')

out = model(image)
out_img = ToPILImage()(out[0].data.cpu())
out_img.save('high.jpg')