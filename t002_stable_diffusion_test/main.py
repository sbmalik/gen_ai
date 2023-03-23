"""
pip install diffusers transformers huggingface-hub

"""
import logging
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from diffusers import StableDiffusionPipeline
from huggingface_hub import login

torch.manual_seed(1)

login("username", "password")

CONFIGS = {
    'h': 512, # Defaut height
    'w': 512, # Defaut width
    'seed': 42 # TORCH manual seed.
}

model_id = 'CompVis/stable-diffusion-v1-4'
# We are loading the FP16 model here as that requires less memory compared to the FP32 model.
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=True,
).to("cuda")

# We delete VAE encoder as we do not need it to generate images.
# This step frees up more than 2GB VRAM that allows us to run 
# Stable Diffusion on a 6GB VRAM GPU.
del pipe.vae.encoder

torch.manual_seed(CONFIGS['seed'])
prompt = "Anthropomorphic blue owl, highly detailed, big green eyes, portrait, \
         detailed armor, unreal engine, cinematic lighting, metal design, 8k, \
         octane render, realistic, redshift render"
pipe(
    prompt, height=CONFIGS['h'], width=CONFIGS['w'], num_inference_steps=150
).images[0]