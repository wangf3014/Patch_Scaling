# Scaling Laws in Patchification: An Image Is Worth 50,176 Tokens And More
Official implementation of Scaling Laws in Patchification: An Image Is Worth 50,176 Tokens And More

This repository primarily follows the codebase of [DeiT](https://github.com/facebookresearch/deit) and [Adventurer](https://github.com/wangf3014/Adventurer).

*Arxiv: https://arxiv.org/pdf/2502.03738*

<img src="images\main.png" width="90%" />

## Release
- [Feb.24.2025] 📢 Code and model weights released.

## Models
| Model                   | Input Size | IN-1k Top-1 Acc. | Checkpoint                                                   |
| ----------------------- | ---------- | ---------------- | ------------------------------------------------------------ |
| Adventurer-Base-patch1  | 224        | 84.6             | [Adventurer_base_patch1_224](https://huggingface.co/Wangf3014/Patch_Scaling/resolve/main/adventurer_base_patch1_224.pth) |
| Adventurer-Small-patch1 | 224        | 83.8             | [Adventurer_small_patch1_224](https://huggingface.co/Wangf3014/Patch_Scaling/resolve/main/adventurer_small_patch1_224.pth) |
| Adventurer-Tiny-patch1  | 224        | 81.9             | [Adventurer_tiny_patch1_224](https://huggingface.co/Wangf3014/Patch_Scaling/resolve/main/adventurer_tiny_patch1_224.pth) |
| Adventurer-Base-patch1  | 128        | 82.4             | [Adventurer_base_patch1_128](https://huggingface.co/Wangf3014/Patch_Scaling/resolve/main/adventurer_base_patch1_128.pth) |
| Adventurer-Small-patch1 | 128        | 81.4             | [Adventurer_small_patch1_128](https://huggingface.co/Wangf3014/Patch_Scaling/resolve/main/adventurer_small_patch1_128.pth) |
| Adventurer-Tiny-patch1  | 128        | 80.7             | [Adventurer_tiny_patch1_128](https://huggingface.co/Wangf3014/Patch_Scaling/resolve/main/adventurer_tiny_patch1_128.pth) |
| Adventurer-Base-patch1  | 64         | 80.9             | [Adventurer_base_patch1_64](https://huggingface.co/Wangf3014/Patch_Scaling/resolve/main/adventurer_base_patch1_64.pth) |
| Adventurer-Small-patch1 | 64         | 80.5             | [Adventurer_small_patch1_64](https://huggingface.co/Wangf3014/Patch_Scaling/resolve/main/adventurer_small_patch1_64.pth) |
| Adventurer-Tiny-patch1  | 64         | 77.4             | [Adventurer_tiny_patch1_64](https://huggingface.co/Wangf3014/Patch_Scaling/resolve/main/adventurer_tiny_patch1_64.pth) |

- Find comparisons to the patch size 16*16 models in [Adventurer](https://github.com/wangf3014/Adventurer) repo.

## Install
- Prepare your environment
```bash
conda create -n adventurer python=3.10
source activate adventurer
```
- Install Pytorch with CUDA version >= 11.8
```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```
- Other dependencies, wandb and submitit are optinal
```bash
pip install timm==0.4.12 mlflow==2.9.1 setuptools==69.5.1 wandb submitit
```
- Install causal-conv1d and mamba-2
```bash
pip install causal-conv1d==1.2.1
pip install mamba-ssm==2.0.4
```

## Evaluation
- For evaluations of very small input sizes like 64*64, we recommend changing the --eval-crop-ratio to 0.875
```python
python -m torch.distributed.launch --nproc_per_node=1  --use_env main.py \
    --model adventurer_base_patch1 --input-szie 224 \
    --data-path /PATH/TO/IMAGENET --batch 128 \
    --resume /PATH/TO/CHECKPOINT \
    --eval --eval-crop-ratio 1.0
```
## Training
- We basically follow the multi-stage training strategy of [Mamba-Reg](https://github.com/wangf3014/Mamba-Reg).
- Here we provide single-node, 8-GPU training scripts. For multi-node training, we have integrated the codebase with [submitit](https://github.com/facebookincubator/submitit), which allows conveniently launching distributed jobs on [Slurm](https://slurm.schedmd.com/quickstart.html) clusters. See [Multi-node training](README_MULTI_NODE.md) for more details.
- Stage one, pretrain with 128*128 inputs
```python
python -m torch.distributed.launch --nproc_per_node=8  --use_env main.py \
    --model adventurer_base_patch16 \
    --data-path /PATH/TO/IMAGENET \
    --batch 128 --lr 5e-4 --weight-decay 0.05 \
    --output_dir ./output/adventurer_base_patch16_224/s1_128 \
    --reprob 0.0 --smoothing 0.0  --repeated-aug --ThreeAugment \
    --epochs 300 --input-size 128 --drop-path 0.1 --dist-eval
```
- Stage two, train with 224*224 inputs
```python
python -m torch.distributed.launch --nproc_per_node=8  --use_env main.py \
    --model adventurer_base_patch16 \
    --data-path /PATH/TO/IMAGENET \
    --batch 128 --lr 5e-4 --weight-decay 0.05 \
    --finetune ./output/adventurer_base_patch16_224/s1_128/checkpoint.pth
    --output_dir ./output/adventurer_base_patch16_224/s2_224 \
    --reprob 0.0 --smoothing 0.0  --repeated-aug --ThreeAugment \
    --epochs 100 --input-size 224 --drop-path 0.4 --dist-eval
```
- Stage three, finetune with 224*224 inputs
```python
python -m torch.distributed.launch --nproc_per_node=8  --use_env main.py \
    --model adventurer_base_patch16 \
    --data-path /PATH/TO/IMAGENET \
    --batch 64 --lr 1e-5 --weight-decay 0.1 --unscale-lr \
    --finetune ./output/adventurer_base_patch16_224/s2_224/checkpoint.pth
    --output_dir ./output/adventurer_base_patch16_224/s3_224 \
    --reprob 0.0 --smoothing 0.1 --no-repeated-aug --aa rand-m9-mstd0.5-inc1 \
    --epochs 20 --input-size 224 --drop-path 0.6 --dist-eval
```

## Citation
```bash
@article{wang2025scaling,
  title={Scaling Laws in Patchification: An Image Is Worth 50,176 Tokens And More},
  author={Wang, Feng and Yu, Yaodong and Wei, Guoyizhe and Shao, Wei and Zhou, Yuyin and Yuille, Alan and Xie, Cihang},
  journal={arXiv preprint arXiv:2502.03738},
  year={2025}
}

@article{wang2024causal,
  title={Causal Image Modeling for Efficient Visual Understanding},
  author={Wang, Feng and Yang, Timing and Yu, Yaodong and Ren, Sucheng and Wei, Guoyizhe and Wang, Angtian and Shao, Wei and Zhou, Yuyin and Yuille, Alan and Xie, Cihang},
  journal={arXiv preprint arXiv:2410.07599},
  year={2024}
}
```
