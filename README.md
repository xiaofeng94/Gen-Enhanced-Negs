# Generating Enhanced Negatives for Training Language-Based Object Detectors, CVPR 2024

[arXiv](https://arxiv.org/abs/2401.00094)

This repo offers: 
- How to evaluate [GLIP](https://github.com/microsoft/GLIP) on [OmniLabel benchmark](https://www.omnilabel.org/). 
- TODO: How to evaluate [FIBER](https://github.com/microsoft/FIBER) on [OmniLabel benchmark](https://www.omnilabel.org/). 
- TODO: Generated negative samples for [Flickr30k dataset](https://bryanplummer.com/Flickr30kEntities/), which is used to train detectors in our paper. 


## Evaluating GLIP on OmniLabel
We provide our customized GLIP code in `./GLIP`. We only add necessary code to enable evaluation on [OmniLabel benchmark](https://www.omnilabel.org/). 
You may check the git log in `./GLIP` for what we modified, if you want merge those changes into your own codebase. NOTE: Below is not a necessary step to run the evaluation
```
cd <root folder of this repo>
git checkout 4190ad40f1caa371e73c4168ced9f1adc25d382e
cd ./GLIP
mv .git_old .git
git log
```

### Install GLIP
- You may follow the following instruction to install GLIP to avoid any incompatibility issues.
```
cd <root folder of this repo>/GLIP
# create new environment
conda create -n GLIP_omni python=3.8
conda activate GLIP_omni

# install pytorch v1.9
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# install other required packages
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo
pip install opencv-python pycocotools
pip install transformers==4.30

conda install -c anaconda nltk
conda install -c conda-forge matplotlib inflect scipy

# compile maskrcnn benchmark
python setup.py build develop --user
```

### Install omnilabeltools
Install omnilabeltools as follow or as the [official instruction](https://github.com/samschulter/omnilabeltools).
```
git clone https://www.github.com/samschulter/omnilabeltools
cd omnilabeltools
pip install .
```

### Download pretrained weights
- Official GLIP checkpoints are available at [GLIP Model Zoo](https://github.com/microsoft/GLIP?tab=readme-ov-file#model-zoo).

- Our checkpoints finetuned with negatives are available at [Google Drive]() (coming soon).

- [Optional] Create `OUTPUT` folder to contain the checkpoints.

### Download datasets
- Create `DATASET/omnilabel` folder
- Download OmniLabel benchmark via its [official website](https://www.omnilabel.org/dataset). Put the images in `DATASET/omnilabel/imgs` and .json files in `DATASET/omnilabel/`.
```bazaar
DATASET/
  omnilabel/
    dataset_all_val_v0.1.4.json
    dataset_all_val_v0.1.3.json
    ...
    imgs/
      coco/
        <list of images>
      object365/
        <list of patches>
      openimagesv5/
        <list of images>
```

- [Optional] Download datasets used in original GLIP codebase as [this doc](https://github.com/microsoft/GLIP/blob/main/DATA.md).

### Run evaluation
Replace `<model_weight>` with your path to a checkpoint.
```
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port 47773 \
./tools/test_net_omnilabel.py \
        --config-file ./configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    --weight <model_weight> \
        --task_config ./configs/omnilabel_val_eval.yaml \
        --chunk_size 20 \
        TEST.IMS_PER_BATCH 8 \
    DATASETS.TEST "('omnilabel_val_1.4',)" \
    OUTPUT_DIR OUTPUT/GLIP_omnilabel_eval
```
- Note 1: It may take hours to finish. You may increase `--chunk_size` to speed up but with possible performance drops.
- Note 2: To evaluate on other splits of omnilabel, change `omnilabel_val_1.4` (in `DATASETS.TEST "('omnilabel_val_1.4',)"`) to `omnilabel_val_1.3`, `omnilabel_val_1.3_coco`, `omnilabel_val_1.3_o365`, or `omnilabel_val_1.3_oi_v5`
- Note 3: `omnilabel_val_1.3` is used for OmniLabel Challenge 2023, `omnilabel_val_1.4` for OmniLabel Challenge 2024.
- Note 4: You may change `--nproc_per_node=8` and `TEST.IMS_PER_BATCH 8` to any number to reduce/increase the number of GPUs used in the evaluation.

## Evaluating FIBER on OmniLabel
TODO


## Generated negative samples
TODO


## Acknowledgement

This repository was built on top of [GLIP](https://github.com/microsoft/GLIP/blob/main/DATA.md), [FIBER](https://github.com/microsoft/FIBER), and [alpaca-lora](https://github.com/tloen/alpaca-lora). We thank the effort from our community.

## Citation
If this repository helps your work, please consider to cite our paper:
```BibTeX
@inproceedings{zhao2024generating,
  title={Generating Enhanced Negatives for Training Language-Based Object Detectors},
  author={Zhao, Shiyu and Zhao, Long and Suh, Yumin and Metaxas, Dimitris N and Chandraker, Manmohan and Schulter, Samuel and others},
  booktitle={CVPR},
  pages={13592--13602},
  year={2024}
}
```
