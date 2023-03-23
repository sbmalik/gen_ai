import os
import torch
import torchvision.transforms as TF
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import base64


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_images(images, path, **kwargs):
    grid = make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get(element: torch.Tensor, t: torch.Tensor):
    """
    Get value at index position "t" in "element" and
        reshape it to have the same dimension as a batch of images.
    """
    ele = element.gather(-1, t)
    return ele.reshape(-1, 1, 1, 1)


def setup_log_directory(config):
    """Log and Model checkpoint directory Setup"""

    if os.path.isdir(config.root_log_dir):
        # Get all folders numbers in the root_log_dir
        folder_numbers = [
            int(folder.replace("version_", ""))
            for folder in os.listdir(config.root_log_dir)
        ]

        # Find the latest version number present in the log_dir
        last_version_number = max(folder_numbers)

        # New version name
        version_name = f"version_{last_version_number + 1}"

    else:
        version_name = config.log_dir

    # Update the training config default directory
    log_dir = os.path.join(config.root_log_dir, version_name)
    checkpoint_dir = os.path.join(config.root_checkpoint_dir, version_name)

    # Create new directory for saving new experiment version
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Logging at: {log_dir}")
    print(f"Model Checkpoint at: {checkpoint_dir}")

    return log_dir, checkpoint_dir


def frames2vid(images, save_path):

    WIDTH = images[0].shape[1]
    HEIGHT = images[0].shape[0]

    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     fourcc = 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(save_path, fourcc, 25, (WIDTH, HEIGHT))

    # Appending the images to the video one by one
    for image in images:
        video.write(image)

    # Deallocating memories taken for window creation
    # cv2.destroyAllWindows()
    video.release()
    return


# def display_gif(gif_path):
#     b64 = base64.b64encode(open(gif_path,'rb').read()).decode('ascii')
#     display(HTML(f'<img src="data:image/gif;base64,{b64}" />'))


def get_dataset():
    transforms = TF.Compose(
        [
            TF.ToTensor(),
            TF.Resize(
                (32, 32), interpolation=TF.InterpolationMode.BICUBIC, antialias=True
            ),
            TF.RandomHorizontalFlip(),
            TF.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
        ]
    )

    dataset = datasets.ImageFolder(
        root="/home/sbmalik/workspace/research/gen_ai/datasets/flowers",
        transform=transforms,
    )

    return dataset


def get_dataloader(
    batch_size=32,
    pin_memory=False,
    shuffle=True,
    num_workers=0,
    device="cpu",
):
    dataset = get_dataset()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    device_dataloader = DeviceDataLoader(dataloader, device)
    return device_dataloader


def inverse_transform(tensors):
    """Convert tensors from [-1., 1.] to [0., 255.]"""
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0


if __name__ == "__main__":
    loader = get_dataloader(
        batch_size=8,
        device="cpu",
    )
    plt.figure(figsize=(12, 6), facecolor="white")

    for b_image, _ in loader:
        b_image = inverse_transform(b_image).cpu()
        grid_img = make_grid(
            b_image / 255.0,
            nrow=16,
            padding=True,
            pad_value=1,
            normalize=True,
        )
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.axis("off")
        plt.show()
        break
