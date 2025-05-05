## Steps

### 1. Prepare the Dataset

- Download from https://datadryad.org/landing/show?id=doi%3A10.5061%2Fdryad.bnzs7h4jd
- Unzip `MuralDH.zip` and place the folder under `./Dunhuang/`.

```bash
# Example structure after unzip:
./Dunhuang/MuralDH/
```

---

### 2. Generate Degraded Segmentation Dataset (1000 Image Pairs)

```bash
# From ./Dunhuang/preprocessing, run:
python degrade_segmentation.py
```

**Output structure:**

```
Mural_seg_downscaled/
├── train/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

### 3. Clone LaMa Repository (Outside Dunhuang)

```bash
# Go to the parent folder (where Dunhuang is located):
git clone https://github.com/advimman/lama.git
```

> The folder structure should look like:
>
> ```
> ./Dunhuang/
> ./lama/
> ```

---

### 4. Preprocess `Mural512` Dataset (Split, Mask, and Downscale)

#### Masking Strategy

- Irregular masks: Simulate weathering, cracks, or flaking (e.g., free-form brush or blobs).
- Box/rectangle masks: Simulate localized damage or occlusion.
- Combined strategy: Random choice between irregular and rectangular for diversity. --> What we used here

**Dunhuang-style Mask**:

```bash
# From ./Dunhuang/preprocessing, run:
python preprocess_Mural512.py \
  --source_dir ../MuralDH/Mural512 \
  --output_dir ./Mural512_processed \
  --resize_to 256 256 \
  --use_dh_mask
```

**Lama Mask (flaky/buggy)**:

```bash
pip install -r ../../../lama/requirements.txt;
# From ./Dunhuang/preprocessing, run:
python preprocess_Mural512.py \
  --source_dir ../MuralDH/Mural512 \
  --output_dir Mural512_processed \
  --mask_config ../../../lama/mask_gen_config.yaml \
  --lama_repo_dir ../../../lama \
  --resize_to 256 256
```

**Output structure:**

```
Mural512_processed/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

## MuralDH Dataset:

```
|-- MuralDH (Files: 1, Dirs: 3)
    |-- README.md
    |-- Mural_seg (Files: 0, Dirs: 2)
        |-- test (Files: 0, Dirs: 2)
            |-- images (Files: 201, Dirs: 0)
                |-- 001072.png
                |-- 000436.png
                |-- 000608.png
                |-- 000146.png
                |-- 000973.png
                |-- ... (196 more files)
            |-- labels (Files: 201, Dirs: 0)
                |-- 001072.png
                |-- 000436.png
                |-- 000608.png
                |-- 000146.png
                |-- 000973.png
                |-- ... (196 more files)
        |-- train (Files: 0, Dirs: 2)
            |-- images (Files: 760, Dirs: 0)
                |-- 004863.png
                |-- img_11crop_1_0.png
                |-- 002584.png
                |-- img_157crop_0_1.png
                |-- 001714.png
                |-- ... (755 more files)
            |-- labels (Files: 760, Dirs: 0)
                |-- 004863.png
                |-- img_11crop_1_0.png
                |-- 002584.png
                |-- img_157crop_0_1.png
                |-- 001714.png
                |-- ... (755 more files)
    |-- Mural_SR (Files: 0, Dirs: 2)
        |-- Mural_DataSet (Files: 512, Dirs: 0)
            |-- 0298.png
            |-- 0267.png
            |-- 0501.png
            |-- 0273.png
            |-- 0065.png
            |-- ... (507 more files)
        |-- Mural_DataSet_LR (Files: 0, Dirs: 4)
            |-- X6 (Files: 512, Dirs: 0)
                |-- 0418x6.png
                |-- 0370x6.png
                |-- 0335x6.png
                |-- 0071x6.png
                |-- 0034x6.png
                |-- ... (507 more files)
            |-- X4 (Files: 512, Dirs: 0)
                |-- 0292x4.png
                |-- 0193x4.png
                |-- 0507x4.png
                |-- 0337x4.png
                |-- 0372x4.png
                |-- ... (507 more files)
            |-- X3 (Files: 512, Dirs: 0)
                |-- 0390x3.png
                |-- 0029x3.png
                |-- 0405x3.png
                |-- 0091x3.png
                |-- 0440x3.png
                |-- ... (507 more files)
            |-- X2 (Files: 512, Dirs: 0)
                |-- 0254x2.png
                |-- 0211x2.png
                |-- 0155x2.png
                |-- 0110x2.png
                |-- 0048x2.png
                |-- ... (507 more files)
    |-- Mural512 (Files: 5096, Dirs: 0)
        |-- img_414crop_1_1.png
        |-- img_281crop_0_1.png
        |-- img_144crop_1_1.png
        |-- img_302crop_0_0.png
        |-- img_821crop_0_1.png
        |-- ... (5091 more files)
```
