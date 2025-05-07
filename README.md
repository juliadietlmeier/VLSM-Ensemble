# VLSM-Ensemble: Ensembling CLIP-based Vision-Language Segmentation Models
[Accepted at MIDL 2025](https://openreview.net/forum?id=IJJJT1vIdX#discussion)

# Architecture
<div align="center">
   <img src="https://github.com/user-attachments/assets/5197ebda-205d-40c8-9a97-5a918284dee3" width="600" />
</div>

# Datasets
For data preparation please follow the instructions posted in the **medvlsm** repository
1. [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)
2. [ClinicDB](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)
3. [BKAI](https://www.kaggle.com/c/bkai-igh-neopolyp/data)
4. [BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
5. [CheXlocalize](https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c)

# To run our ensemble models
The models can be integrated into the **medvlsm** repository https://github.com/naamiinepal/medvlsm with a few simple tricks:

1. To run the BiomedCLIPSeg-A save BiomedCLIPSeg_A.py as medvlsm/src/models/biomedclipseg.py
2. To run the CLIPSeg-B save CLIPSeg_B.py as medvlsm/src/models/clipseg.py
3. To run Ensemble-C save Ensemble_C.py as medvlsm/src/models/biomedclipseg.py and change configurations in ...
   
# To model UNet-D component
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
This repo works with the experimental testbench, data splits and text prompts developed by **naamiinepal** (Poudel et al., 2024) in their **medvlsm** GitHub repo https://github.com/naamiinepal/medvlsm. Sincere thanks for their excellent work!

# Citations
Julia Dietlmeier, Oluwabukola Grace Adegboro, Vayangi Ganepola, Claudia Mazo and Noel E. O'Connor. "**VLSM-Ensemble: Ensembling CLIP-based Vision-Language
Models for Enhanced Medical Image Segmentation**", in MIDL 2025

Kanchan Poudel, Manish Dhakal, Prasiddha Bhandari, Rabin Adhikari, Safal Thapaliya and Bishesh Khanal. "**Exploring Transfer Learning in Medical Image Segmentation
using Vision-Language Models**", in MIDL 2024


