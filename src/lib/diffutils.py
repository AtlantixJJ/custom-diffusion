import sys, torch
import numpy as np
sys.path.insert(0, ".")
sys.path.insert(0, "./src")
import diffusers
from diffusers import UNet2DModel, VQModel
#from diffusers.models.vae import VQModelMask
from diffusers.models.guided_diffusion.unet import UNetModel as UNet_GuidedDiffusion
from diffusers.models.unet_2d import UNet2DOutput


class UNetWrapped(UNet_GuidedDiffusion):
    """To wrap an OpenAI model to work in diffuser."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args, self.kwargs = args, kwargs
        self.device = "cuda"
        self.sample_size = kwargs["image_size"]
        self.in_channels = 3

    def clone(self):
        net = UNetWrapped(*self.args, **self.kwargs)
        net.load_state_dict(self.state_dict())
        return net.to(self.device)

    def forward(self, x, timesteps, y=None):
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.Tensor([timesteps]).to(x)
        res = super().forward(x, timesteps, y)
        return UNet2DOutput(sample=res[:, :3]) # discard learned sigma

    def initialize_cross_attention(self):
        for m in self.modules():
            if "AttentionBlock" in str(type(m)):
                m.initialize_cross_attention()

    def collect_attention_from_ckpt(self):
        qkvs = []
        for m in self.modules():
            if "AttentionBlock" in str(type(m)):
                qkvs.append(m.qkv_ckpt.clone().detach())
        return qkvs

    def collect_variable(self):
        vars = []
        for m in self.modules():
            #if "AttentionBlock" in str(type(m)):
            #    #vars.append(m.pre_att_x)
            #    vars.append(m.qkv.weight) # the parameter of attention
            if "QKV" in str(type(m)):
                vars.append(m.att_score)
        return vars

    def delete_variable(self):
        for m in self.modules():
            if "AttentionBlock" in str(type(m)):
                m.pre_att_delta = None
            if "QKV" in str(type(m)):
                m.att_score_delta = None

    def set_variable(self, vars):
        count = 0
        for m in self.modules():
            if "AttentionBlock" in str(type(m)):
                m.qkv.weight = vars[count]
                count += 1
            if count >= len(vars):
                break

    def set_variable_delta(self, deltas):
        count = 0
        for m in self.modules():
            #if "AttentionBlock" in str(type(m)):
            #    m.pre_att_delta = deltas[count]
            #    m.qkv.weight.add_(deltas[count])
            #    count += 1
            if "QKV" in str(type(m)):
                m.att_score_delta = deltas[count]
                count += 1
            if count >= len(deltas):
                break

    def set_attention_from_list(self, qkvs):
        count = 0
        for m in self.modules():
            if "AttentionBlock" in str(type(m)):
                m.qkv_context = qkvs[count].to(self.device)
                count += 1

    def set_attention_from_ckpt(self):
        for m in self.modules():
            if "AttentionBlock" in str(type(m)):
                m.qkv_context = m.qkv_ckpt.clone().detach()

    def delete_attention_ckpt(self):
        for m in self.modules():
            if "AttentionBlock" in str(type(m)):
                m.qkv_context = None
                m.qkv_ckpt = None

    def get_attention_parameter(self):
        for m in self.modules():
            if "AttentionBlock" in str(type(m)):
                for p in m.cross_q_conv.parameters(): #m.parameters():
                    yield p


def default_guided_diffusion_model():
    return create_model(
        image_size=256,
        num_channels=128,
        num_res_blocks=1,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="16",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False)


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=True,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetWrapped(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=None,#(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def load(score_model, sampler, resolution=256, cache_dir="../../pretrained", pipe_with_vae=True):
    """Load the scheduler, unet, vqvae from model name."""
    model_name = f"{cache_dir}/{score_model}-{resolution}"

    # load and initialize
    scheduler_cls = {
            "ddpm": diffusers.DDPMScheduler,
            "ddim": diffusers.DDIMScheduler,
            "pndm": diffusers.PNDMScheduler,
            "pc":   diffusers.ScoreSdeVeScheduler,
            "dpmsolver++": diffusers.DDPMScheduler, # dummy
        }[sampler]

    pipe_cls = {
            "ddpm": diffusers.DDPMPipeline,
            "ddim": diffusers.DDIMPipeline,
            "pndm": diffusers.PNDMPipeline,
            "pc":   diffusers.ScoreSdeVePipeline,
            "dpmsolver++": diffusers.DPMSolverPipeline, # dummy
        }[sampler]

    if "ldm" in score_model:
        unet = UNet2DModel.from_pretrained(model_name,
            subfolder="unet", cache_dir=cache_dir).cuda()
        scheduler = scheduler_cls.from_config(model_name,
            subfolder="scheduler", cache_dir=cache_dir)
        vqvae = VQModel.from_pretrained(model_name,
            subfolder="vqvae", cache_dir=cache_dir).cuda()
        #vqvae = VQModelMask.from_config(model_name,
        #    subfolder="mask_vqvae", cache_dir=cache_dir).cuda()
        vqvae.init_first_layer()
        vqvae.load_state_dict(torch.load(f"{model_name}/mask_vqvae/model.pth"))
        vqvae.use_mask = True
    else:
        if "ddpm-ffhq" in score_model:
            unet = default_guided_diffusion_model().cuda()
            unet.load_state_dict(torch.load(f"{cache_dir}/ffhq_ilvr.pt"))
        else:
            unet = UNet2DModel.from_pretrained(model_name,
                cache_dir=cache_dir).cuda()
        vqvae = None
        scheduler = scheduler_cls.from_config(model_name,
            cache_dir=cache_dir)

    if sampler == "dpmsolver++":
        scheduler = diffusers.DPMSolverScheduler(unet, scheduler, "dpmsolver++", scheduler.config.clip_sample)

    pipe = pipe_cls(unet=unet, scheduler=scheduler, vqvae=vqvae if pipe_with_vae else None).to("cuda")

    return unet, vqvae, scheduler, pipe


def inpaint_identity(inpaint_pipe, prompt, neg_prompt, iv_imgs, iv_masks, visual_feat,
    num_infer_steps, guidance_scale, max_infer_num):
    masked_images, images = [], []
    if max_infer_num < 0:
        max_infer_num = iv_imgs.shape[0]
    for i in range(min(max_infer_num, iv_imgs.shape[0])):
        images.append([])
        masked_images.append([])
        for j in range(iv_masks.shape[1]):
            masked_image = iv_imgs[i:i+1] * (1 - iv_masks[i:i+1, j])
            image = inpaint_pipe(prompt, masked_image, iv_masks[i:i+1, j],
                extra_feats=visual_feat, num_inference_steps=num_infer_steps,
                guidance_scale=guidance_scale, negative_prompt=neg_prompt)
            images[i].append(image)
            masked_images[i].append(masked_image)
    return masked_images, images
