import torch
import os
from torchvision.utils import save_image
from utils import get_data
import argparse
from ddfm import Diffusion


# get_file_path
current_file_path = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.dataset_path = os.path.join(current_file_path, '../datasets/flowers')
args.batch_size = 1
args.image_size = 64

dataloader = get_data(args)

diff = Diffusion(device="cpu")

images = next(iter(dataloader))[0]
t = torch.Tensor([50, 100, 150, 200, 300, 600, 700, 999]).long()

noised_image, _ = diff.noise_images(images, t)
save_image(noised_image.add(1).mul(0.5), os.path.join(current_file_path, '../logdir/results/noise.jpg'))
