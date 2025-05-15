import os

# 目录结构
PREP_INPUT   = os.path.join('MuralDH', 'Preprocessed', 'input')
PREP_TARGET  = os.path.join('MuralDH', 'Preprocessed', 'target')
RESTORED_DIR = 'MuralDH/Results'

# 输出 Markdown 文件名
OUT_MD = 'COMPARE_CLAUDE.md'

# 获取所有样本（仅 png）
fns = sorted([fn for fn in os.listdir(PREP_INPUT) if fn.lower().endswith('.png')])

with open(OUT_MD, 'w', encoding='utf-8') as f:
    # 表头
    f.write('| Original (target) | Corrupted (input) | Restored50_3 |\n')
    f.write('| :---------------: | :---------------: | :------: |\n')

    for fn in fns:
        inp_path  = os.path.join(PREP_INPUT,   fn).replace('\\', '/')
        tgt_path  = os.path.join(PREP_TARGET,  fn).replace('\\', '/')
        res_path  = os.path.join(RESTORED_DIR, fn).replace('\\', '/')
        f.write(f'| ![]({tgt_path}) | ![]({inp_path}) | ![]({res_path}) |\n')

print(f"Generated comparison markdown → {OUT_MD}")
