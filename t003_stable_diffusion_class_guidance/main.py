import argparse
import os
import torch
from utils import *
from model import UNet, UNetConditional, EMA
from ddfm import Diffusion
from tqdm import tqdm
import numpy as np
import copy

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
        pbar.set_description(f"Train :: Epoch: {epoch}/{args.epochs}")
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
        
        if epoch % 20 == 0:
            sampled_images = diffusion.sample(model, n=images.shape[0])
            save_images(sampled_images, 
                        os.path.join(os.path.dirname(__file__), f"../logdir/results/{epoch}.jpg"),
                       )
            torch.save(model.state_dict(), 
                    os.path.join(os.path.dirname(__file__), f"../logdir/results/{epoch}.pt"),
                    )

def train_conditional(args):
    device = args.device
    dataloader = get_data(args)
    model = UNetConditional(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size,
                           device=device)
    l = len(dataloader)

    ema = EMA(beta=0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader)
        pbar.set_description(f"Train :: Epoch: {epoch}/{args.epochs}")
        for i, (images, labels) in enumerate(pbar):
            
            images = images.to(device)
            labels = labels.to(device)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            if np.random.random() < 0.1:
                labels = None

            predicted_noise = model(x_t, t, labels)
            loss = mse(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.step_ema(model, ema_model)

            pbar.set_postfix(MSE=loss.item())
        
        if epoch % 20 == 0:
            labels = torch.arange(5).long().to(device)
            # n = images.shape[0]
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            save_images(sampled_images, 
                    os.path.join(os.path.dirname(__file__), f"../logdir/results_cfg/{epoch}.jpg"),
                    )
            torch.save(model.state_dict(), 
                    os.path.join(os.path.dirname(__file__), f"../logdir/results_cfg/{epoch}.pt"),
                    )
            save_images(ema_sampled_images, 
                    os.path.join(os.path.dirname(__file__), f"../logdir/results_cfg/{epoch}_ema.jpg"),
                    )
            torch.save(ema_model.state_dict(), 
                    os.path.join(os.path.dirname(__file__), f"../logdir/results_cfg/{epoch}_ema.pt"),
                    )


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 301
    args.batch_size = 6
    args.image_size = 64
    args.num_classes = 5
    args.dataset_path = os.path.join(os.path.dirname(__file__), '../datasets/flowers')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.lr = 3e-4
    # train(args)
    train_conditional(args)

def infer(args):
    device = args.device
    model = UNet(device="cpu").to(device)
    model.load_state_dict(torch.load(args.model_path))
    diffusion = Diffusion(img_size=args.image_size,
                           device=device)
    sampled_images = diffusion.sample(model, n=args.num_images)
    save_images(sampled_images, args.save_path)

def main_infer():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.num_images = 32
    args.image_size = 64
    # args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = 'cpu'
    args.model_path = os.path.join(os.path.dirname(__file__), '../logdir/results/280.pt')
    args.save_path = os.path.join(os.path.dirname(__file__), '../logdir/results/infer.jpg')
    infer(args)

if __name__ == '__main__':
    main()
    # main_infer()