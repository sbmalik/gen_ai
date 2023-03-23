import torch
from torchmetrics import MeanMetric
from dataloader import *
from config import *
from models import UNet
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc


class SimpleDiffusion:
    def __init__(
        self,
        num_diffusion_timesteps=1000,
        img_shape=(3, 64, 64),
        device="cpu",
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.img_shape = img_shape
        self.device = device

        self.initialize()

    def initialize(self):
        # BETAs & ALPHAs required at different places in the Algorithm.
        self.beta = self.get_betas()
        self.alpha = 1 - self.beta

        self_sqrt_beta = torch.sqrt(self.beta)
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha = 1.0 / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)

    def get_betas(self):
        """linear schedule, proposed in original ddpm paper"""
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.num_diffusion_timesteps,
            dtype=torch.float32,
            device=self.device,
        )


def forward_diffusion(sd: SimpleDiffusion, x0: torch.Tensor, timesteps: torch.Tensor):
    eps = torch.randn_like(x0)  # Noise
    mean = get(sd.sqrt_alpha_cumulative, t=timesteps) * x0  # Image scaled
    std_dev = get(sd.sqrt_one_minus_alpha_cumulative, t=timesteps)  # Noise scaled
    sample = mean + std_dev * eps  # scaled inputs * scaled noise

    return sample, eps  # return ... , gt noise --> model predicts this)


def perform_forward_diffusion():
    sd = SimpleDiffusion(num_diffusion_timesteps=TrainingConfig.TIMESTEPS, device="cpu")

    loader = iter(  # converting dataloader into an iterator for now.
        get_dataloader(
            batch_size=6,
            device="cpu",
        )
    )
    x0s, _ = next(loader)  # get a batch of images
    noisy_images = []
    specific_timesteps = [0, 10, 50, 100, 150, 200, 250, 300, 400, 600, 800, 999]

    for timestep in specific_timesteps:
        timestep = torch.as_tensor(timestep, dtype=torch.long)

        xts, _ = forward_diffusion(sd, x0s, timestep)
        xts = inverse_transform(xts) / 255.0
        xts = make_grid(xts, nrow=1, padding=1)

        noisy_images.append(xts)

    _, ax = plt.subplots(1, len(noisy_images), figsize=(10, 5), facecolor="white")

    for i, (timestep, noisy_sample) in enumerate(zip(specific_timesteps, noisy_images)):
        ax[i].imshow(noisy_sample.squeeze(0).permute(1, 2, 0))
        ax[i].set_title(f"t={timestep}", fontsize=8)
        ax[i].axis("off")
        ax[i].grid(False)

    plt.suptitle("Forward Diffusion Process", y=0.9)
    plt.axis("off")
    plt.show()


# Algorithm 1: Training


def train_one_epoch(
    model,
    sd,
    loader,
    optimizer,
    scaler,
    loss_fn,
    epoch=800,
    base_config=BaseConfig(),
    training_config=TrainingConfig(),
):

    loss_record = MeanMetric()
    model.train()

    with tqdm(total=len(loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{training_config.NUM_EPOCHS}")

        for x0s, _ in loader:
            tq.update(1)

            ts = torch.randint(
                low=1,
                high=training_config.TIMESTEPS,
                size=(x0s.shape[0],),
                device=base_config.DEVICE,
            )
            xts, gt_noise = forward_diffusion(sd, x0s, ts)

            with torch.cuda.amp.autocast():
                pred_noise = model(xts, ts)
                loss = loss_fn(gt_noise, pred_noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            loss_value = loss.detach().item()
            loss_record.update(loss_value)

            tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")

        mean_loss = loss_record.compute().item()

        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")

    return mean_loss


@torch.no_grad()
def reverse_diffusion(
    model,
    sd,
    timesteps=1000,
    img_shape=(3, 64, 64),
    num_images=5,
    nrow=8,
    device="cpu",
    **kwargs,
):

    x = torch.randn((num_images, *img_shape), device=device)
    model.eval()

    if kwargs.get("generate_video", False):
        outs = []

    for time_step in tqdm(
        iterable=reversed(range(1, timesteps)),
        total=timesteps - 1,
        dynamic_ncols=False,
        desc="Sampling :: ",
        position=0,
    ):

        ts = torch.ones(num_images, dtype=torch.long, device=device) * time_step
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

        predicted_noise = model(x, ts)

        beta_t = get(sd.beta, ts)
        one_by_sqrt_alpha_t = get(sd.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_cumulative_t = get(sd.sqrt_one_minus_alpha_cumulative, ts)

        x = (
            one_by_sqrt_alpha_t
            * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )

        if kwargs.get("generate_video", False):
            x_inv = inverse_transform(x).type(torch.uint8)
            grid = make_grid(x_inv, nrow=nrow, pad_value=255.0).to("cpu")
            ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]
            outs.append(ndarr)

    if kwargs.get(
        "generate_video", False
    ):  # Generate and save video of the entire reverse process.
        frames2vid(outs, kwargs["save_path"])
        # display(Image.fromarray(outs[-1][:, :, ::-1])) # Display the image at the final timestep of the reverse process.
        return None

    else:  # Display and save the image at the final timestep of the reverse process.
        x = inverse_transform(x).type(torch.uint8)
        grid = make_grid(x, nrow=nrow, pad_value=255.0).to("cpu")
        pil_image = TF.functional.to_pil_image(grid)
        # pil_image.save(kwargs['save_path'], format=save_path[-3:].upper())
        # display(pil_image)
        return None


def perform_training():
    model = UNet(
        input_channels=TrainingConfig.IMG_SHAPE[0],
        output_channels=TrainingConfig.IMG_SHAPE[0],
        base_channels=ModelConfig.BASE_CH,
        base_channels_multiples=ModelConfig.BASE_CH_MULT,
        apply_attention=ModelConfig.APPLY_ATTENTION,
        dropout_rate=ModelConfig.DROPOUT_RATE,
        time_multiple=ModelConfig.TIME_EMB_MULT,
    )
    model = model.to(BaseConfig.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LR)
    dataloader = get_dataloader(
        batch_size=TrainingConfig.BATCH_SIZE,
        device=BaseConfig.DEVICE,
        pin_memory=True,
        num_workers=TrainingConfig.NUM_WORKERS,
    )
    loss_fn = torch.nn.MSELoss()
    sd = SimpleDiffusion(
        num_diffusion_timesteps=TrainingConfig.TIMESTEPS,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
    )

    scaler = torch.cuda.amp.GradScaler()
    total_epochs = TrainingConfig.NUM_EPOCHS + 1
    log_dir, checkpoint_dir = setup_log_directory(config=BaseConfig())

    generate_video = True
    ext = ".mp4" if generate_video else ".png"

    for epoch in range(1, total_epochs):
        torch.cuda.empty_cache()
        gc.collect()

        # Algorithm 1: Training
        train_one_epoch(model, sd, dataloader, optimizer, scaler, loss_fn, epoch=epoch)

        if epoch % 2 == 0:
            save_path = os.path.join(log_dir, f"{epoch}{ext}")

            # Algorithm 2: Sampling
            reverse_diffusion(
                model,
                sd,
                timesteps=TrainingConfig.TIMESTEPS,
                num_images=32,
                generate_video=generate_video,
                save_path=save_path,
                img_shape=TrainingConfig.IMG_SHAPE,
                device=BaseConfig.DEVICE,
            )

            # clear_output()
            checkpoint_dict = {
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "model": model.state_dict(),
            }
            torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "ckpt.tar"))
            del checkpoint_dict


def perform_inference():
    model = UNet(
        input_channels          = TrainingConfig.IMG_SHAPE[0],
        output_channels         = TrainingConfig.IMG_SHAPE[0],
        base_channels           = ModelConfig.BASE_CH,
        base_channels_multiples = ModelConfig.BASE_CH_MULT,
        apply_attention         = ModelConfig.APPLY_ATTENTION,
        dropout_rate            = ModelConfig.DROPOUT_RATE,
        time_multiple           = ModelConfig.TIME_EMB_MULT,
    )
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "ckpt.tar"), map_location='cpu')['model'])

    model.to(BaseConfig.DEVICE)

    sd = SimpleDiffusion(
        num_diffusion_timesteps = TrainingConfig.TIMESTEPS,
        img_shape               = TrainingConfig.IMG_SHAPE,
        device                  = BaseConfig.DEVICE,
    )

    log_dir = "inference_results"
    os.makedirs(log_dir, exist_ok=True)

    generate_video = True

    ext = ".mp4" if generate_video else ".png"
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"

    save_path = os.path.join(log_dir, filename)

    reverse_diffusion(
        model,
        sd,
        num_images=256,
        generate_video=generate_video,
        save_path=save_path,
        timesteps=1000,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
        nrow=32,
    )
    print(save_path)




if __name__ == "__main__":
    # perform_forward_diffusion()
    perform_training()
    # perform_inference()
