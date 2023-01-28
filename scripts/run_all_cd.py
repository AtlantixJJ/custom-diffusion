import os

model_name = "../../pretrained/stable-diffusion"
data_dir = "../../data/celebahq"
id_names = os.listdir(f"{data_dir}/id_inpaint")
id_names.sort()
cmd_format = """accelerate launch --multi_gpu --gpu_ids 6,7 --num_processes=2 \
src/diffuser_training.py \
--pretrained_model_name_or_path={model_name}  \
--instance_data_dir={data_dir}/id_inpaint/{id_name}/ref_image  \
--class_data_dir={data_dir}/image  \
--output_dir={data_dir}/custom_diffusion/{id_name}  \
--with_prior_preservation --real_prior --prior_loss_weight=1.0  \
--instance_prompt="face photo of a <new1> person"  \
--class_prompt="A face photo of a person"  \
--resolution=512   \
--train_batch_size=1   \
--learning_rate=1e-5   \
--lr_warmup_steps=0  \
--max_train_steps=750 \
--num_class_images=10000  \
--scale_lr \
--modifier_token "<new1>" \
--gradient_accumulation_steps 4 \
--gradient_checkpointing \
--save_steps 100000
"""
# do not save intermediate results

for i, id_name in enumerate(id_names):
    out_dir = f"{data_dir}/custom_diffusion/{id_name}"
    if os.path.exists(out_dir):
        print(f"!> Skip {out_dir}")
        continue
    cmd = cmd_format.format(data_dir=data_dir,
        id_name=id_name, model_name=model_name)
    print(cmd)
    os.system(cmd)
    if i > 20:
        break
