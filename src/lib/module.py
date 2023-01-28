import torch, torchvision, os, sys
import torch.nn.functional as F
import torchvision.transforms.functional as ttf
from transformers.models.clip.modeling_clip import CLIPAttention
import pytorch_lightning as pl
sys.path.insert(0, "./src")
sys.path.insert(0, ".")
import diffusers, transformers
from lib.diffutils import inpaint_identity
from lib.misc import bu
from lib.dataset import PROMPT_TEMPLATES_SMALL
from diffusers.models.face import FaceNet
from diffusers import MySDInpaint
from diffusers.models.attention import CrossAttention


class VGGLoss(torch.nn.Module):
    """Computes the VGG perceptual loss between two batches of images.
    The input and target must be 4D tensors with three channels
    ``(B, 3, H, W)`` and must have equivalent shapes. Pixel values should be
    normalized to the range 0–1.
    The VGG perceptual loss is the mean squared difference between the features
    computed for the input and target at layer :attr:`layer` (default 8, or
    ``relu2_2``) of the pretrained model specified by :attr:`model` (either
    ``'vgg16'`` (default) or ``'vgg19'``).
    If :attr:`shift` is nonzero, a random shift of at most :attr:`shift`
    pixels in both height and width will be applied to all images in the input
    and target. The shift will only be applied when the loss function is in
    training mode, and will not be applied if a precomputed feature map is
    supplied as the target.
    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.
    :meth:`get_features()` may be used to precompute the features for the
    target, to speed up the case where inputs are compared against the same
    target over and over. To use the precomputed features, pass them in as
    :attr:`target` and set :attr:`target_is_features` to :code:`True`.
    Instances of :class:`VGGLoss` must be manually converted to the same
    device and dtype as their inputs.
    """

    models = {
        'vgg16': torchvision.models.vgg16,
        'vgg19': torchvision.models.vgg19}

    def __init__(self, model='vgg16', layer=15, shift=0, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model = self.models[model](pretrained=True).features[:layer+1]
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.cuda()

    def get_features(self, input):
        return self.model(self.normalize(input))

    def train(self, mode=True):
        self.training = mode

    def forward(self, input, target, mask=None):
        sep = input.shape[0]
        batch = torch.cat([input, target])
        #if self.shift and self.training:
        #    padded = F.pad(batch, [self.shift] * 4, mode='replicate')
        #    batch = transforms.RandomCrop(batch.shape[2:])(padded)
        feats = self.get_features(batch)
        diff = (feats[:sep] - feats[sep:]) ** 2
        if mask:
            resized_mask = F.interpolate(mask, diff.shape[2:],
                mode="bilinear", align_corners=True)[:, 0]
            diff = (diff.sum(1) * resized_mask).view(sep, -1).sum(1)
            area = resized_mask.view(sep, -1).clamp(min=1e-5)
        else:
            return 0.5 * diff.mean()
        return (diff / area).mean()


#class FastPersonalizedInpaintPipeline(pl.LightningModule):
class FastPersonalizedInpaintPipeline(torch.nn.Module):
    """Learning-based Fast Personalized Inpainting Pipeline."""

    PROMPT_TEMPLATES = {
        "A face photo of {special_token}. {vpt_tokens}",
    }

    def __init__(self,
                 from_model_name="stable-diffusion-inpainting",
                 cache_dir="../../pretrained",
                 clip_trans_name="none-0",
                 id_trans_name="none-0",
                 special_token="<person>",
                 num_vpt_token=10,
                 mode="AS",
                 memory_saving=1,
                 device="cuda",
                 args=None):
        super().__init__()
        self.args = args
        self.num_vpt_token = num_vpt_token
        self.special_token = special_token
        self.id_trans_name = id_trans_name
        self.clip_trans_name = clip_trans_name
        self.mode = mode
        self.memory_saving = memory_saving
        self.device = device
        self._init_from_model_name(from_model_name, cache_dir)

    def _set_sancheck(self):
        self.is_sancheck = True

    def _resize_text_embedding(self):
        # Add the placeholder token in tokenizer
        new_tokens = [f"<vpt-tkn{i}>" for i in range(self.num_vpt_token)]
        self.new_tokens = [self.special_token] + new_tokens
        self.tokenizer.add_tokens(self.new_tokens)
        self.new_token_ids = self.tokenizer.convert_tokens_to_ids(
            self.new_tokens)
        self.special_token_id = self.new_token_ids[0]

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.clip_text_enc.resize_token_embeddings(len(self.tokenizer))

        token_embeds = self.clip_text_enc.get_input_embeddings().weight.data
        self.tuned_embedding = TunedEmbedding(token_embeds,
            self.special_token_id, "multi_insert")
        self.clip_text_enc.text_model.embeddings.token_embedding = \
            self.tuned_embedding

    def _init_from_model_name(self, model_name, cache_dir):
        clip_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" #"openai/clip-vit-large-patch14"
        self.face_net = FaceNet(device=self.device)
        self.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(
            model_name, subfolder="scheduler")
        clip = transformers.CLIPModel.from_pretrained(
            clip_name, cache_dir="../../pretrained")
        sd = clip.state_dict()
        clip_text_enc = transformers.CLIPTextModel(clip.config.text_config)
        clip_text_enc.load_state_dict(sd, strict=False)
        clip_image_enc = transformers.CLIPVisionModel(clip.config.vision_config)
        clip_image_enc.load_state_dict(sd, strict=False)
        clip_image_proj = torch.nn.Linear(
            clip.vision_embed_dim, clip.projection_dim, bias=False)
        clip_image_proj.load_state_dict(clip.visual_projection.state_dict())
        del clip, sd
        self.clip_text_enc = clip_text_enc
        self.clip_image_enc = clip_image_enc
        self.clip_image_proj = clip_image_proj
        _c = transformers.CLIPFeatureExtractor.from_pretrained(
            clip_name, cache_dir=cache_dir)
        self.clip_image_proc = lambda x: ttf.normalize(F.interpolate(
            x, (224, 224), mode="bicubic"), _c.image_mean, _c.image_std)
        self.clip_image_func = lambda x: clip_image_proj(clip_image_enc(
            self.clip_image_proc(x))[0]) # (N, C, H, W) -> (N, L, D)
        self.vae = diffusers.AutoencoderKL.from_pretrained(
            model_name, subfolder="vae", cache_dir=cache_dir)
        self.unet = diffusers.UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet", cache_dir=cache_dir)
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer", cache_dir=cache_dir)

        if self.mode == "IS":
            self._resize_text_embedding()

        if "none" in self.id_trans_name:
            self.trans_net = None
        else:
            mode, num_layer = self.id_trans_name.split("-")
            self.trans_net = TransformNetwork(512, clip.projection_dim, mode, int(num_layer))
            self.trans_net.to(self.device).train()
        
        self.vae.to(self.device).eval()
        self.unet.to(self.device).eval()
        self.face_net.to(self.device).eval()
        self.clip_text_enc.to(self.device).eval()
        self.clip_image_enc.to(self.device).eval()
        self.clip_image_proj.to(self.device).eval()

    def _append_visual_feature(self, x, text_feat=None, id_feat=None):
        """Append visual feature after CLIP text encoder."""
        if hasattr(self, "tune_clip") and self.tune_clip in ["proj", "none"]:
            torch.set_grad_enabled(False)
        if False:#self.memory_saving: # no memory saving effect
            N = x.shape[0]
            grad_idx = torch.randint(0, x.shape[0], (1,)).item()
            grad_mask = torch.zeros(N).bool()
            grad_mask[grad_idx] = True
            nograd_x = x[~grad_mask]
            with torch.no_grad():
                nograd_feat = [self.clip_image_enc(nograd_x[i:i+1])
                    for i in range(nograd_x.shape[0])]
            clip_feat = self.clip_image_enc(x[grad_mask])
            idx = 0 if self.mode == "AA" else 1
            clip_feat = clip_feat[idx]
            nograd_feat = torch.cat([f[idx] for f in nograd_feat])
            clip_feat = torch.cat([nograd_feat.requires_grad_(True), clip_feat])
        else:
            clip_feat = self.clip_image_enc(x)
            if self.mode == "AA":
                clip_feat = clip_feat[0]
            elif self.mode == "AS":
                clip_feat = clip_feat[1]
        if hasattr(self, "tune_clip") and self.tune_clip in ["proj", "none"]:
            torch.set_grad_enabled(True)
        clip_feat = self.clip_image_proj(clip_feat)
        if self.trans_net is not None and id_feat is not None:
            trans_id_feat = self.trans_net(id_feat)
            clip_feat = clip_feat.view(id_feat.shape[0], -1, clip_feat.shape[-1])
            clip_feat = torch.cat([clip_feat, trans_id_feat], 1)
        if text_feat is not None:
            clip_feat = clip_feat.view(text_feat.shape[0], -1, clip_feat.shape[-1])
            return torch.cat([text_feat, clip_feat], 1), clip_feat
        return clip_feat

    def _insert_visual_feature(self, text_id, id_feat=None):
        """Insert visual feature before CLIP text encoder."""
        trans_id_feat = self.trans_net(id_feat)
        self.tuned_embedding.set_extra_feature(trans_id_feat)
        return self.clip_text_enc(text_id)[0], trans_id_feat

    def calc_visual_feature(self, x,
            text_id=None, text_feat=None, id_feat=None):
        if self.mode in ["AA", "AS"]:
            return self._append_visual_feature(x, text_feat, id_feat)
        elif self.mode in ["IS"]:
            return self._insert_visual_feature(text_id, id_feat)

    def set_trainable_parameters(self, tune_clip="none", tune_unet="none"):
        self.tune_clip = tune_clip
        self.tune_unet = tune_unet
        self.requires_grad_(False)
        trainable_params = []

        if self.trans_net is not None:
            self.trans_net.requires_grad_(True)
            trainable_params.append({
                "params": self.trans_net.parameters(),
                "lr": 2e-4})
        
        if self.tune_clip != "none":
            self.clip_image_proj.train()
            trainable_params.append({"params": self.clip_image_proj.parameters()})

        if self.tune_clip in ["QKV", "KV", "all"]:
            self.clip_image_enc.train()
            if self.memory_saving:
                self.clip_image_enc.gradient_checkpointing_enable()

        for m in self.clip_image_enc.modules():
            if isinstance(m, CLIPAttention):
                if tune_clip == "QKV":
                    trainable_params.append({"params": m.parameters()})
                elif tune_clip == "KV":
                    trainable_params.extend([
                        {"params": m.k_proj.parameters()},
                        {"params": m.v_proj.parameters()},
                        #{"params": m.out_proj.parameters()}
                        ])

        if "all" in self.tune_clip:
            trainable_params.append({"params":
                self.clip_image_enc.vision_model.encoder.parameters()})

        if self.tune_unet != "none":
            self.unet.train()
            if self.memory_saving:
                self.unet.enable_gradient_checkpointing()    

        if self.tune_unet == "all":
            trainable_params.append({"params": self.unet.parameters()})

        for m in self.unet.modules():
            if isinstance(m, CrossAttention):
                if self.tune_unet == "QKV":
                    trainable_params.append({"params": m.parameters()})
                elif self.tune_unet == "KV":
                    trainable_params.extend([
                        {"params": m.to_k.parameters()},
                        {"params": m.to_v.parameters()},
                        #{"params": m.to_out.parameters()}
                        ])

        for name, params in self.unet.named_parameters():
            if self.tune_unet == "QKV":
                if 'attn2' in name:
                    params.requires_grad = True
            elif self.tune_unet == "KV":
                if 'attn2.to_k' in name or 'attn2.to_v' in name:
                    params.requires_grad = True

        for p_g in trainable_params:
            for p in p_g["params"]:
                p.requires_grad = True

        return trainable_params

    def restore(self, data):
        """Load components from saved weights.
        """
        if self.trans_net is not None and "transform_network" in data:
            self.trans_net.load_state_dict(data["transform_network"])

        if "clip_vision_projection" in data:
            self.clip_image_proj.load_state_dict(
                data["clip_vision_projection"])

        if "clip_vision_model" in data:
            self.clip_image_enc.load_state_dict(
                data["clip_vision_model"], strict=False)

        if "unet" in data:
            self.unet.load_state_dict(data["unet"], strict=False)

    def save(self, accelerator, args):
        """Saving training parameters. Accelerator is required during training. args is also required."""
        dic = {}
        if hasattr(self, "tuned_embedding"):
            new_token_embeds = self.tuned_embedding.weight[self.new_token_ids]
            dic["new_tokens"] = new_token_embeds.clone().detach()
        if self.trans_net is not None:
            dic["transform_network"] = accelerator.unwrap_model(
                self.trans_net).state_dict()
        if self.tune_clip != "none":
            vision_model = accelerator.unwrap_model(self.clip_image_enc)
            proj_model = accelerator.unwrap_model(self.clip_image_proj)
        else:
            vision_model = self.clip_image_enc
            proj_model = self.clip_image_proj
        dic["clip_vision_model"] = vision_model.state_dict()
        dic["clip_vision_projection"] = proj_model.state_dict()
        if self.tune_unet != "none":
            model = accelerator.unwrap_model(self.unet)
        else:
            model = self.unet
        dic["unet"] = model.state_dict()

        torch.save(dic, os.path.join(args.output_dir, "FPIP.bin"))

    def prepare_accelerator(self, accelerator):
        unet = accelerator.prepare(self.unet)
        clip_image_enc, clip_image_proj = \
            accelerator.prepare(self.clip_image_enc, self.clip_image_proj)
        if self.trans_net is not None:
            trans_net = accelerator.prepare(self.trans_net)
        else:
            trans_net = None
        return unet, clip_image_enc, clip_image_proj, trans_net

    def inpaint_dl_generator(self, dl,
            device="cuda", size=256, guidance_scale=1,
            max_infer_num=1, max_samples=5, num_infer_steps=10):
        """Inpaint image in each identity in the dataloader. Return as a generator.
        Note that accelerator DDP cannot work with multiple inference images per identity.
        """
        if max_samples < 0:
            max_samples = len(dl)
        inpaint_pipe = MySDInpaint(self.vae, self.clip_text_enc,
            self.tokenizer, self.unet, self.scheduler,
            safety_checker=None, feature_extractor=None)

        for idx, batch in enumerate(dl):
            cosims, inputs, images, ids, image_ids = [], [], [], [], []
            # (N, 3, H, W), (N, 2, 3, H, W)
            iv_imgs = batch["infer_image"][0].to(device) * 2 - 1
            iv_masks = batch["infer_mask"][0, :, :, :1].to(device)
            rv_imgs = batch["ref_image"].to(device) # (1, N_REF, 3, H, W)
            ref_id_feats = batch["ref_id_feat"].to(device) # (1, N_REF, 512)
            infer_id_feats = batch["infer_id_feat"][0].to(device)
            N, N_REF = rv_imgs.shape[:2]
            rv_imgs = rv_imgs.view(N * N_REF, *rv_imgs.shape[-3:])
            rv_imgs_proc = self.clip_image_proc(rv_imgs)
            infer_id_feats /= infer_id_feats.norm(p=2, dim=-1, keepdim=True)
            prompt = list(PROMPT_TEMPLATES_SMALL.values())[0]
            neg_prompt = prompt.format("a person")
            if self.mode in ["AA", "AS"]:
                prompt = prompt.format("a person")
                visual_feat = self.calc_visual_feature(
                    x=rv_imgs_proc, id_feat=ref_id_feats)
            elif self.mode in ["IS"]:
                prompt = prompt.format("<person>")
                visual_feat = None
                trans_id_feat = self.trans_net(ref_id_feats)
                self.tuned_embedding.set_extra_feature(trans_id_feat)
            x_ins, x_outs = inpaint_identity(
                inpaint_pipe, prompt, neg_prompt, iv_imgs, iv_masks, visual_feat,
                num_infer_steps, guidance_scale, max_infer_num)
            for i in range(len(x_outs)):
                for j in range(len(x_outs[i])):
                    inpaint_id_feat = self.face_net(x_outs[i][j])[0]
                    inpaint_id_feat /= inpaint_id_feat.norm()
                    cosims.append(infer_id_feats[i].dot(inpaint_id_feat))
                    inputs.append(bu(x_ins[i][j], (size, size))[0])
                    images.append(bu(x_outs[i][j], (size, size))[0])
                    ids.append(batch["id"])
                    image_ids.append(batch["all_indice"][i])
            res = [torch.stack(x) for x in [ids, image_ids, inputs, images, cosims]]
            yield res + [bu(rv_imgs, (size, size))]
            if idx >= max_samples - 1:
                break

    def inpaint_dl_generator_sancheck(self, dl,
            device="cuda", size=256, guidance_scale=1,
            max_infer_num=1, max_samples=5, num_infer_steps=10):
        """Inpaint image in each identity in the dataloader. Return as a generator.
        Note that accelerator DDP cannot work with multiple inference images per identity.
        """
        if max_samples < 0:
            max_samples = len(dl)
        inpaint_pipe = MySDInpaint(self.vae, self.clip_text_enc,
            self.tokenizer, self.unet, self.scheduler,
            safety_checker=None, feature_extractor=None)
        
        for idx, batch in enumerate(dl):
            cosims, inputs, images, ids, image_ids = [], [], [], [], []
            # (N, 3, H, W), (N, 2, 3, H, W)
            iv_imgs = batch["infer_image"][0].to(device) * 2 - 1
            iv_masks = batch["infer_mask"][0, :, :, :1].to(device)
            rv_imgs = batch["ref_image"].to(device) # (1, N_REF, 3, H, W)
            ref_id_feats = batch["ref_id_feat"].to(device) # (1, N_REF, 512)
            infer_id_feats = batch["infer_id_feat"][0].to(device)
            N, N_REF = rv_imgs.shape[:2]
            rv_imgs = rv_imgs.view(N * N_REF, *rv_imgs.shape[-3:])
            rv_imgs_proc = self.clip_image_proc(rv_imgs)
            iv_imgs_proc = self.clip_image_proc(iv_imgs / 2 + 0.5)
            gt_id_feats = infer_id_feats / infer_id_feats.norm(p=2, dim=-1, keepdim=True)
            prompt = list(PROMPT_TEMPLATES_SMALL.values())[0]

            x_ins, x_outs = [], []
            if max_infer_num < 0:
                max_infer_num = iv_imgs.shape[0]
            for i in range(min(max_infer_num, iv_imgs.shape[0])):
                x_outs.append([])
                x_ins.append([])
                id_feats = torch.cat(
                    [infer_id_feats[None, i:i+1], ref_id_feats], 1)
                #img = torch.cat([iv_imgs_proc[i:i+1], rv_imgs_proc], 0)
                neg_prompt = prompt.format("a person")
                if self.mode in ["AA", "AS"]:
                    prompt = prompt.format("a person")
                    visual_feat = self.calc_visual_feature(
                        x=iv_imgs_proc[i:i+1], id_feat=id_feats)
                elif self.mode in ["IS"]:
                    prompt = prompt.format("<person>")
                    visual_feat = None
                    trans_id_feat = self.trans_net(id_feats)
                    self.tuned_embedding.set_extra_feature(trans_id_feat)
                for j in range(iv_masks.shape[1]):
                    masked_image = iv_imgs[i:i+1] * (1 - iv_masks[i:i+1, j])
                    image = inpaint_pipe(prompt, masked_image, iv_masks[i:i+1, j],
                        extra_feats=visual_feat, num_inference_steps=num_infer_steps,
                        guidance_scale=guidance_scale, negative_prompt=neg_prompt)
                    x_outs[i].append(image)
                    x_ins[i].append(masked_image)

            for i in range(len(x_outs)):
                for j in range(len(x_outs[i])):
                    inpaint_id_feat = self.face_net(x_outs[i][j])[0]
                    inpaint_id_feat /= inpaint_id_feat.norm()
                    cosims.append(gt_id_feats[i].dot(inpaint_id_feat))
                    inputs.append(bu(x_ins[i][j], (size, size))[0])
                    images.append(bu(x_outs[i][j], (size, size))[0])
                    ids.append(batch["id"])
                    image_ids.append(batch["all_indice"][i])
            res = [torch.stack(x) for x in [ids, image_ids, inputs, images, cosims]]
            yield res + [bu(rv_imgs, (size, size))]
            if idx >= max_samples - 1:
                break

    """ Deprecated. No longer uses Pytorch-Lightning.
    def configure_optimizers(self):
        args = self.args
        self.optimizer = torch.optim.AdamW(
            self.set_trainable_parameters(args.tune_clip, args.tune_unet),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay)
        self.lr_scheduler = diffusers.optimization.get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps)
        return [self.optimizer], [self.lr_scheduler]

    def prepare_for_training(self):
        with torch.no_grad():
            self.input_ids = self.tokenizer("A face photo of <person>",
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt").input_ids.to(self.device)
            neg_text_ids = self.tokenizer("TAhe face photo of a person",
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt").input_ids.to(self.device)
            self.text_feat = self.clip_text_enc(neg_text_ids)

    def training_step(self, batch):
        mask = batch["random_mask"][:, 0, :1] # (N, 1, H, W)
        smask = F.interpolate(mask, scale_factor=1/8)
        iv_img = batch["infer_image"][:, 0] * 2 - 1 # (N, C, H, W)
        rv_imgs = batch["ref_image"]
        ref_id_feat = batch["ref_id_feat"] # (N, N_REF, 512)
        if self.is_sancheck:
            rv_imgs = iv_img.unsqueeze(0) / 2 + 0.5
            ref_id_feat = batch["infer_id_feat"]
        N, N_REF = rv_imgs.shape[:2]
        rv_imgs = rv_imgs.view(N * N_REF, *iv_img.shape[-3:])
        masked_image = iv_img * (1 - mask)
        input_ids = self.input_ids.to(iv_img.device)

        B = iv_img.shape[0]
        T = self.scheduler.config.num_train_timesteps
        with torch.no_grad():
            gt_latents = 0.18215 * self.vae.encode(iv_img).latent_dist.sample()
            mask_latents = 0.18215 * self.vae.encode(masked_image).latent_dist.sample()
            noise = torch.randn_like(mask_latents)
            t = torch.randint(0, T, (B,), device=noise.device)
            noisy_latents = self.scheduler.add_noise(gt_latents, noise, t)
            z_t = torch.cat([noisy_latents, smask, mask_latents], 1)
            rv_imgs = self.clip_image_proc(rv_imgs)

        if True: # drop condition occationally
            context_feat, visual_feat = self.calc_visual_feature(
                x=rv_imgs, text_id=input_ids, id_feat=ref_id_feat)
        else:
            visual_feat = text_feat
            context_feat = text_feat

        pred_eps = self.unet(z_t, t, context_feat).sample
        dsm_loss = F.mse_loss(pred_eps, noise,
            reduction="none").mean([1, 2, 3])

        for i, t_ in enumerate(t):
            dsm_loss_bin = int(t_ / (T // 5))
            self.log(f"dsm_loss_{dsm_loss_bin}", dsm_loss[i])
        self.log("visual_feat_norm", visual_feat.norm(p=2, dim=1).mean())
        return dsm_loss
    """


class TransformNetwork(torch.nn.Sequential):
    def __init__(self, in_dim, out_dim, mode="mlp", num_layer=4, dropout_p=0.1):
        if mode == "trans":
            enc_layer = torch.nn.TransformerEncoderLayer(out_dim, 8,
                dropout=dropout_p, batch_first=True)
            layer_norm = torch.nn.LayerNorm(out_dim)
            trans_enc = torch.nn.TransformerEncoder(enc_layer, num_layer, layer_norm)
            proj_layer = torch.nn.Linear(in_dim, out_dim)
            super().__init__(proj_layer, trans_enc)
        elif mode == "mlp":
            layers = [torch.nn.Linear(in_dim, out_dim)]
            for _ in range(num_layer - 1):
                layers.append(torch.nn.ReLU(inplace=True))
                layers.append(torch.nn.Linear(out_dim, out_dim))
                layers.append(torch.nn.Dropout(dropout_p))
            super().__init__(*layers)
        self.mode = mode
        self.num_layer = num_layer


class TunedEmbedding(torch.nn.Embedding):
    def __init__(self, weight, stoken_id=-1, mode="single_token"):
        super().__init__(weight.shape[0], weight.shape[1])
        self.weight.data.copy_(weight)
        self.mode = mode
        self.stoken_id = stoken_id

    def set_extra_feature(self, feat):
        self.feat = feat
        self.n_dim = len(feat.shape)

    def forward(self, x):
        embed = super().forward(x)
        
        if self.mode == "append":
            pad_token_id = x[0, -1]
            row_idx = (x == pad_token_id).long().argmax(1)
        else:
            stoken_mask = x == self.stoken_id
            # prompt does not contain special token (e.g. in classifier-free)
            if stoken_mask.sum() < 1:
                return embed
            row_idx = [torch.where(m)[0][0].item() for m in stoken_mask]

        if self.mode == "single_token": # add with last token
            if self.n_dim == 1:
                embed[stoken_mask] = self.feat
            else:
                for i in range(embed.shape[0]):
                    v = self.feat[i] if self.n_dim == 2 else self.feat[i, -1]
                    embed[i, row_idx[i]] = v

        elif self.mode in ["multi_insert", "append"]:
            N = self.feat.shape[1]
            assert self.n_dim == 3
            embed_orig = embed.clone()
            for i in range(embed.shape[0]):
                st = row_idx[i]
                embed[i, st + 1 + N:] = embed_orig[i, st + 1 : -N]
                embed[i, st + 1: st + 1 + N] = self.feat[i]
        return embed

