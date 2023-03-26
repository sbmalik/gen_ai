import argparse
import os
import torch
from utils import *
from model import UNet
from ddfm import Diffusion
from tqdm import tqdm

def train(args):
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size,
                           device=device)
    l = len(dataloader)

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
        
        if i % 20 == 0:
            sampled_images = diffusion.sample(model, n=images.shape[0])
            save_images(sampled_images, 
                        os.path.join(os.path.dirname(__file__), f"../logdir/results/{epoch}.jpg"),
                       )
            torch.save(model.state_dict(), 
                    os.path.join(os.path.dirname(__file__), f"../logdir/results/{epoch}.pt"),
                    )


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 500
    args.batch_size = 4
    args.image_size = 64
    args.dataset_path = os.path.join(os.path.dirname(__file__), '../datasets/flowers')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.lr = 3e-4
    train(args)

if __name__ == '__main__':
    main()