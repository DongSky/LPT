# LPT
The official code of Long-tailed Prompt Tuning

Our code is based on the [unofficial VPT code](https://github.com/DongSky/vpt_reproduce) implemented by DongSky.

This repository will be updated continuously in the near future. 

## Preparing Data

#### Places-LT

Download the original Places365 standard dataset from [here](http://places2.csail.mit.edu/download.html), and then change the path of Places-LT in datasets.py by the current root path of places365standard.

Note that we have stored the train/val/test split of Places-LT in vtab directory (move into phase2 test directory and you will see this dir). 

## Testing LPT

Here we present LPT trained on Places-LT dataset. 

Note that for simplicity during experiments, I stored the whole model into storage... The final size of LPT checkpoint may be slightly larger (negligible) than standard ViT.

LPT (Places-LT): [Google Drive](https://drive.google.com/file/d/1PEd0YUW_BH0hz6s6QE8GRholJl0XdA3F/view?usp=sharing)

Set the checkpoint to the Phase2 test directory, and then execute the following commands:

```shell
CUDA_VISIBLE_DEVICES=0 python eval_pool.py --dataset places365 --split full
```

You will obtain:

```shell
epoch 1, overall: 50.07123%, many-shot: 49.26718%, medium-shot: 52.30573%, few-shot: 46.88312%
```

## TODO
- Training code
- More checkpoints

Give me some time to prepare the code QAQ.