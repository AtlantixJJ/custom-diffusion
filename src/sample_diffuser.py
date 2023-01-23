# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import argparse, sys, glob, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('./')
sys.path.append('../diffusers/src')
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from src import diffuser_training
from src.lib.visualizer import HtmlPageVisualizer
from src.lib.misc import torch2image, imread_to_tensor, bu


PROMPT_TEMPLATES = {
    "person": "A face photo of {special_token}",
    # expression
    "smile": "A face photo of {special_token}, smiling",
    "sad": "A face photo of {special_token}, looking sad",
    # gender
    "female": "A face photo of {special_token}, female looking",
    "male": "A face photo of {special_token}, male looking",
    # attributes
    "red_eye": "A face photo of {special_token}, the eyes are red",
    "single_eye": "A face photo of {special_token}, one of the eyes is closed",
    "close_mouth": "A face photo of {special_token}, mouth is closed",
    "gray_purple_hair": "A face photo of {special_token}, the hair is gray in the top and purple in the bottom",
    # accessories
    "accessory": "A face photo of {special_token}, wearing yellow eyeglasses, a cyan hat, shinny eye rings, and a golden necklace",
    # pose
    "left": "A face photo of {special_token}, looking to the left",
    "right": "A face photo of {special_token}, facing to the right",
    "up": "A face photo of {special_token}, looking upwards"
}


def sample(pipe, delta_ckpt, data_dir, save_prefix,
        compress=False, freeze_model="crossattn_kv"):
    diffuser_training.load_model(pipe.text_encoder, pipe.tokenizer, pipe.unet, delta_ckpt, compress, freeze_model)
    
    ref_images = torch.stack([imread_to_tensor(f)
        for f in glob.glob(f"{data_dir}/ref_image/*")])
    ref_images = torch2image(bu(ref_images, (256, 256)), "[0,1]")

    n_repeat = len(ref_images)
    n_row = len(PROMPT_TEMPLATES) + 1
    n_col = n_repeat
    page = HtmlPageVisualizer(num_rows=n_row, num_cols=n_col)
    count = 0
    for i, img in enumerate(ref_images):
        page.set_cell(0, i, image=img)

    for p_name, prompt_temp in PROMPT_TEMPLATES.items():
        prompt = prompt_temp.format(special_token="a <new1> person")
        print(f"=> {prompt}")
        #images = pipe([prompt] * 5, num_inference_steps=25, guidance_scale=6.)
        images = pipe([prompt]*5, num_inference_steps=200, guidance_scale=6, eta=1.)
        images = torch2image(bu(images, (256, 256)).clamp(-1, 1))
        for img in images:
            row_idx, col_idx = count // n_col, count % n_col
            page.set_cell(1 + row_idx, col_idx, text=p_name, image=img)
            count += 1
    page.save(f"{save_prefix}.html")


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--model-name', type=str,
        default="../../pretrained/stable-diffusion")
    parser.add_argument('--expr-dir', type=str,
        default="../../data/celebahq/custom_diffusion",
        help='The set of all experiments')
    parser.add_argument('--out-dir', type=str,
        default="../../data/celebahq/custom_diffusion_stacked",
        help='The set of all experiments')
    parser.add_argument('--data-dir', type=str,
        default="../../data/celebahq/id_inpaint",
        help='The directory to image data.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    #scheduler = DPMSolverMultistepScheduler.from_config(
    #    args.model_name, subfolder="scheduler", cache_dir="../../pretrained")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name, safety_checker=None,
        #scheduler=scheduler, cache_dir="../../pretrained",
        torch_dtype=torch.float16).to("cuda")

    model_dirs = glob.glob(f"{args.expr_dir}/*")
    model_dirs.sort()
    for model_dir in model_dirs:
        print(f"=> Sampling for {model_dir}")
        model_name = model_dir.split("/")[-1]
        delta_ckpt = f"{model_dir}/delta.bin"
        save_prefix = f"{args.out_dir}/{model_name}"
        data_dir = f"{args.data_dir}/{model_name}"
        sample(pipe, delta_ckpt, data_dir, save_prefix)
