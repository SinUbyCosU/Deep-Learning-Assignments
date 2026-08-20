# Deep-Learning-Assignments

Repository that tracks course assignments and accompanying training scripts.

## Structure

- `Assigment 1/` – resources from the first assignment (unchanged).
- `Assignment2/Assignmnt2.py` – comprehensive robustness training pipeline covering CIFAR-10, Fashion-MNIST, and ImageNet-100 for VGG16, ResNet18, ConvNeXt-Tiny, ViT-B/16, and an MLP baseline.

## Assignment 2 quickstart

Install dependencies (PyTorch, torchvision, matplotlib, scikit-learn, imagecorruptions) and then run, for example:

```bash


python Assignment2/Assignmnt2.py --dataset cifar10 --model resnet18 --epochs 10 --batch_size 64
```
# Assignment 3
uh-oh
# Assignment 4
Additional flags:

- `--use_corrupted_validation` – evaluate with a corrupted validation split.
- `--mlp_validation_suite` – run the clean/corrupted/optimized validation experiments for the MLP baseline.
- `--imagenet_root <path>` – required when training on ImageNet-100.
  
