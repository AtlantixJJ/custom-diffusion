"""Load and sample textual inversion models.
"""
import torch, sys, os, argparse
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, ".")
from src import diffuser_training
sys.path.insert(0, "../diffusers")
sys.path.insert(0, "../diffusers/src")
import diffusers
# import diffusers must happen before vutils, otherwise munmap_chunk error
import torchvision.utils as vutils
from lib.dataset import CelebAHQIDIDataset, PROMPT_TEMPLATES_SMALL, PROMPT_TEMPLATES_CONTROL


def post_process(images):
    image = np.stack(images)
    image = torch.from_numpy(image).permute(0, 3, 1, 2) / 255.
    return F.interpolate(image, (256, 256), mode="bilinear")


def generate_given_prompt(pipe, prompt, masked_image, mask,
    guidance_scale=7.5, negative_prompt=None, seed=None):
    """
    Args:
        pipe: pipeline
        prompt: A text.
        num: Number of images 
    """
    images = []
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    # for classifier-free guidance, we need to concat null text and prompt, so we simply pad the sequence to maximum
    images = pipe(prompt, masked_image, mask,
        num_inference_steps=100, guidance_scale=guidance_scale,
        negative_prompt=negative_prompt, eta=1.0).images
    return post_process(images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load textual inversion model and sample images. Made for face images only.")
    parser.add_argument("--result-dir1", type=str,
        default="../../data/celebahq/SDI_CD",
        help="Path to the experiment directory.")
    parser.add_argument("--result-dir2", type=str,
        default="../../data/celebahq/SDI_CD_stacked",
        help="Path to the experiment directory.")
    parser.add_argument("--cache-dir", type=str,
        default="../../pretrained",
        help="Path to the cache dir of diffusers")
    parser.add_argument("--expr-dir", type=str,
        default="../../data/celebahq/custom_diffusion_inpaint/0000",
        help="Path to the experiment directory.")
    parser.add_argument("--model-path", type=str,
        default="stabilityai/stable-diffusion-2-inpainting",
        help="The name of stable diffusion model.")
    parser.add_argument("--guidance-scale", type=float,
        default=1.0,
        help="The strength of guidance.")
    parser.add_argument("--prompt_set", type=str, default="small",
        help="small / control")
    args = parser.parse_args()

    if os.path.exists(f"{args.expr_dir}/inv_gen"):
        os.system(f"rm -r {args.expr_dir}/inv_gen")
    os.makedirs(f"{args.expr_dir}/inv_gen", exist_ok=True)
    os.makedirs(f"{args.result_dir1}", exist_ok=True)
    os.makedirs(f"{args.result_dir2}", exist_ok=True)

    torch.set_grad_enabled(False)
    scheduler = diffusers.DDIMScheduler.from_config(
        args.model_path, subfolder="scheduler")
    pipe = diffusers.StableDiffusionInpaintPipeline.from_pretrained(
        args.model_path, safety_checker=None, scheduler=scheduler, cache_dir=args.cache_dir)
    pipe = pipe.to("cuda")

    delta_path = f"{args.expr_dir}/delta.bin"
    diffuser_training.load_model(pipe.text_encoder, pipe.tokenizer, pipe.unet,
        delta_path, False, "crossattn")

    person_id = int(args.expr_dir[args.expr_dir.rfind("/")+1:])
    test_ds = CelebAHQIDIDataset(size=(512, 512),
        split="all", loop_data="identity", single_id=person_id,
        inpaint_region=["lowerface", "eyebrow", "wholeface"])
    batch = test_ds[0]

    prompt_temps = PROMPT_TEMPLATES_SMALL if args.prompt_set == "small" \
        else PROMPT_TEMPLATES_CONTROL
    vutils.save_image(batch["ref_image"], f"{args.expr_dir}/inv_gen/reference.png")
    for p_i, (p_name, prompt_temp) in enumerate(prompt_temps.items()):
        prompt = prompt_temp.format("a <new1> person")
        print(f"=> ({p_i}/{len(prompt_temps)}) {prompt}")
        col_imgs = []
        for i, image in enumerate(batch["infer_image"]):
            id_idx = batch["id"]
            image_idx = batch["all_indice"][i]
            image = image.unsqueeze(0).cuda()
            for mask_idx, mask in enumerate(batch["infer_mask"][i]):
                mask = mask[None, :1].cuda()
                masked_image = (image * 2 - 1).cuda() * (1 - mask)
                p_img = generate_given_prompt(pipe,
                    prompt,
                    masked_image, mask, args.guidance_scale,
                    negative_prompt=None,
                    seed=1997)
                name = f"{id_idx}_{image_idx}_{mask_idx}.png"
                if args.prompt_set == "control":
                    name = f"{id_idx}_{image_idx}_{mask_idx}_{p_name}.png"
                vutils.save_image(p_img, f"{args.result_dir1}/{name}.png")