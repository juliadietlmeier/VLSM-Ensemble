# VLSM-Ensemble: Ensembling CLIP-based Vision-Language Segmentation Models
experimenting on biomedclipseg and clipseg

# To model UNet-D component:
1. prepare the datasets such that you have three split folders **train**, **val** and **test**. The images and masks need to have the same filenames! The split for each dataset is defined in /data/.../anns folders in https://github.com/naamiinepal/medvlsm. This will require reading .json files and re-saving images in our three folders. This is very important - we need to use the same splits as in medvlsm repository!
2. Change paths to your data in train.py
3. To train UNet-D run train.py from the command line interface with $python train.py
4. Record the **average** Dice score and **standard deviation** (computed over the entire test set) for the Table 1
5. Do not make any changes to **UNet_D.py**! This component must be left unchaanged for fair comparison with VLSMs ensembles
6. Modify paths to your test data in predict.py
7. Modify predict.py to save all predictions in a test set (We will need some of those for the paper)
8. Once trained, this will save the checkpoint - then run predict.py from the command line interface

# Filenames for qualitative results in the paper:
**BKAI**: 3e732adb4c5d1580670d78b9d054d694.jpeg

**CheXlocalize**: patient64548_study1_view1_frontal_airspace_opacity.jpg (image) or patient64548_study1_view1_frontal_airspace_opacity.png (mask)

If you have prepared the above three split folders correct, then the BKAI and CheXlocalize filenames should be located in the corresponding test folders

# Acknowledgement
This repo works with the experimental testbench, data splits and text prompts developed by **naamiinepal** in their **medvlsm** GitHub repo https://github.com/naamiinepal/medvlsm. Sincere thanks for their excellent work!

# License
Apache License Version 2.0, January 2004

# Citations
Julia Dietlmeier, Oluwabukola Grace Adegboro, Vayangi Ganepola, Aonghus Lawlor, Claudia Mazo and Noel E. O'Connor. "**VLSM-Ensemble: Ensembling CLIP-based Vision-Language
Models for Enhanced Medical Image Segmentation**", in MIDL 2025

Kanchan Poudel, Manish Dhakal, Prasiddha Bhandari, Rabin Adhikari, Safal Thapaliya and Bishesh Khanal. "**Exploring Transfer Learning in Medical Image Segmentation
using Vision-Language Models**", in MIDL 2024


