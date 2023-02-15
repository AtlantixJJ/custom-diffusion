import os, json

model_name = "stabilityai/stable-diffusion-2-inpainting"
data_dir = "../../data/celebahq"
fp = "../../data/celebahq/annotation/celebahq-idi-5.json"
test_ids = json.load(open(fp, "r"))["test_ids"]

# CUDA_VISIBLE_DEVICES=7 ipython -i
cmd_format = """accelerate launch --multi_gpu --gpu_ids 6,7 --main_process_port 29501 --num_processes=2 \
src/diffuser_inpainting_training.py \
--pretrained_model_name_or_path={model_name}  \
--instance_data_dir={data_dir}/id_inpaint/{id_name:04d}/ref_image  \
--class_data_dir={data_dir}/image  \
--output_dir={data_dir}/custom_diffusion_inpaint/{id_name:04d}  \
--with_prior_preservation --real_prior --prior_loss_weight=1.0  \
--flip_p 0.5 \
--instance_prompt="photo of a <new1> person"  \
--class_data_dir={data_dir}/annotation/dialog/class_images.txt  \
--class_prompt={data_dir}/annotation/dialog/class_prompts.txt  \
--resolution=512   \
--train_batch_size=1   \
--gradient_accumulation_steps 4 \
--learning_rate=1e-6   \
--lr_warmup_steps=0  \
--max_train_steps=1000 \
--num_class_images=10000  \
--scale_lr \
--modifier_token "<new1>" \
--freeze_model crossattn \
--gradient_checkpointing \
--save_steps 100000
"""
# do not save intermediate results

for i, id_name in enumerate(test_ids):
    out_dir = f"{data_dir}/custom_diffusion_inpaint/{id_name:04d}"
    if os.path.exists(out_dir):
        print(f"!> Skip {out_dir}")
        continue
    cmd = cmd_format.format(data_dir=data_dir,
        id_name=id_name, model_name=model_name)
    print(cmd)
    os.system(cmd)