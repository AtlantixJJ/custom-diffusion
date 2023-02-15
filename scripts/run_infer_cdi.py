import glob, os


data_dir = "../../data/celebahq/"
expr_dirs = glob.glob(f"{data_dir}/textual_inversion_inpainting/*")
missing_indices = [i for i, d in enumerate(expr_dirs)
    if not os.path.exists(f"{d}/learned_embeds.bin")]
missing_dirs = [expr_dirs[i] for i in missing_indices]
print("Missing trained files: ", missing_dirs)
expr_dirs = [d for i, d in enumerate(expr_dirs)
    if i not in missing_indices]
expr_dirs.sort()
cmd = "python3 src/sample_diffuser_inpaint.py --expr-dir {expr_dir} --guidance-scale {guidance_scale} --result-dir1 ../../data/celebahq/SDI_CD_{guidance_scale}"
cmds = []
for expr_dir in expr_dirs:
    for guidance_scale in [1, 7.5]:
        cmds.append(cmd.format(data_dir=data_dir,
            expr_dir=expr_dir, guidance_scale=guidance_scale))

