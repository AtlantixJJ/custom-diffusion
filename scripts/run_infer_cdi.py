import glob, os


data_dir = "../../data/celebahq/"
expr_dirs = glob.glob(f"{data_dir}/custom_diffusion_inpaint/*")
expr_dirs.sort()
cmd = "python3 src/sample_diffuser_inpaint.py --expr-dir {expr_dir} --guidance-scale {guidance_scale} --result-dir1 ../../data/celebahq/SDI_CD_{guidance_scale}"
cmds = []
for expr_dir in expr_dirs:
    for guidance_scale in [1, 7.5]:
        cmds.append(cmd.format(data_dir=data_dir,
            expr_dir=expr_dir, guidance_scale=guidance_scale))

gpus = [3]
slots = [[] for _ in gpus]
for i, cmd in enumerate(cmds):
    gpu_cmd = f"CUDA_VISIBLE_DEVICES={gpus[i % len(gpus)]} {cmd}"
    slots[i % len(gpus)].append(gpu_cmd)
for slot_cmds in slots:
    slot_cmd = " && ".join(slot_cmds) + " &"
    print(slot_cmd)
    os.system(slot_cmd)

