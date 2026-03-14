"""Comprehensive training script for robustness experiments.

This assignment script covers the following scenarios:
1. Train deep networks (VGG, ResNet, ConvNeXt, ViT) on CIFAR-10, Fashion-MNIST,
   and ImageNet-100 with a consistent 20% validation split when a validation set
   is not predefined.
2. Train an MLP with three validation regimes: clean, corrupted, and optimized
   subset of corruptions. Optionally perturb intermediate feature maps.
3. Evaluate accuracy on clean and corrupted test sets while logging robustness
   metrics and visualizations (feature maps and t-SNE plots) for every setting.

Note: The script focuses on orchestration and reproducibility. Actual wall-clock
training time will depend on available hardware, so consider reducing epochs or
sampling fewer experiments when iterating locally.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder

try:
	from imagecorruptions import corrupt as apply_image_corruption
except ImportError:  # pragma: no cover - optional dependency
	apply_image_corruption = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_STATS = {
	"cifar10": {
		"mean": (0.4914, 0.4822, 0.4465),
		"std": (0.2470, 0.2435, 0.2616),
		"num_classes": 10,
		"image_size": 224,
		"to_rgb": True,
	},
	"fmnist": {
		"mean": (0.5, 0.5, 0.5),
		"std": (0.5, 0.5, 0.5),
		"num_classes": 10,
		"image_size": 224,
		"to_rgb": True,
	},
	"imagenet100": {
		"mean": (0.485, 0.456, 0.406),
		"std": (0.229, 0.224, 0.225),
		"num_classes": 100,
		"image_size": 224,
		"to_rgb": True,
	},
}

CORRUPTION_SET = [
	"gaussian_noise",
	"shot_noise",
	"impulse_noise",
	"defocus_blur",
	"glass_blur",
	"motion_blur",
	"zoom_blur",
	"snow",
	"frost",
	"fog",
	"brightness",
	"contrast",
	"elastic_transform",
	"pixelate",
	"jpeg_compression",
]

MODEL_FACTORIES: Dict[str, Callable[[int], nn.Module]] = {
	"vgg16": lambda num_classes: models.vgg16(num_classes=num_classes),
	"resnet18": lambda num_classes: models.resnet18(num_classes=num_classes),
	"convnext_tiny": lambda num_classes: models.convnext_tiny(num_classes=num_classes),
	"vit_b_16": lambda num_classes: models.vit_b_16(num_classes=num_classes),
}


def build_mlp(num_classes: int, image_size: int = 32) -> nn.Module:
	input_dim = 3 * image_size * image_size
	hidden = 1024
	return nn.Sequential(
		nn.Flatten(),
		nn.Linear(input_dim, hidden),
		nn.ReLU(inplace=True),
		nn.Dropout(0.3),
		nn.Linear(hidden, hidden // 2),
		nn.ReLU(inplace=True),
		nn.Dropout(0.3),
		nn.Linear(hidden // 2, num_classes),
	)


MODEL_FACTORIES["mlp"] = lambda num_classes: build_mlp(num_classes)


@dataclass
class ExperimentConfig:
	dataset: str
	model: str
	epochs: int = 30
	batch_size: int = 128
	lr: float = 3e-4
	weight_decay: float = 5e-4
	val_split: float = 0.2
	corruption_types: Optional[List[str]] = None
	corruption_severity: int = 2
	use_corrupted_validation: bool = False
	optimize_validation_corruption: bool = False
	feature_hook: Optional[str] = None
	log_dir: str = "results"
	split_dir: str = "splits"
	data_dir: str = "data"
	imagenet_root: Optional[str] = None
	num_workers: int = 4
	seed: int = 42
	device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> Path:
	path.mkdir(parents=True, exist_ok=True)
	return path


def tensor_to_uint8(image: Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
	mean_t = torch.tensor(mean).view(-1, 1, 1)
	std_t = torch.tensor(std).view(-1, 1, 1)
	restored = image * std_t + mean_t
	restored = restored.clamp(0.0, 1.0)
	uint8 = (restored * 255.0).byte().permute(1, 2, 0).numpy()
	return uint8


def uint8_to_normalized(array: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> Tensor:
	tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
	mean_t = torch.tensor(mean).view(-1, 1, 1)
	std_t = torch.tensor(std).view(-1, 1, 1)
	return (tensor - mean_t) / std_t


def build_transforms(dataset: str, image_size: int, augment: bool, is_train: bool) -> transforms.Compose:
	stats = DATASET_STATS[dataset]
	ops: List[transforms.Compose] = []
	if stats["to_rgb"]:
		ops.append(transforms.Lambda(lambda img: img.convert("RGB")))
	if augment and is_train:
		if image_size <= 64:
			ops.append(transforms.RandomCrop(image_size, padding=4))
		else:
			ops.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)))
		ops.append(transforms.RandomHorizontalFlip())
		ops.append(transforms.ColorJitter(0.1, 0.1, 0.1, 0.05))
	else:
		ops.append(transforms.Resize((image_size, image_size)))
	ops.append(transforms.ToTensor())
	ops.append(transforms.Normalize(stats["mean"], stats["std"]))
	return transforms.Compose(ops)


def get_split_file(split_dir: Path, dataset: str) -> Path:
	return split_dir / f"{dataset}_split.json"


def get_or_create_split_indices(
	total_length: int,
	dataset_name: str,
	split_dir: Path,
	val_split: float,
	seed: int,
) -> Tuple[List[int], List[int]]:
	ensure_dir(split_dir)
	split_file = get_split_file(split_dir, dataset_name)
	val_count = int(val_split * total_length)
	if val_count == 0:
		raise ValueError("Validation split resulted in zero samples")
	if split_file.exists():
		with split_file.open("r", encoding="utf-8") as fp:
			data = json.load(fp)
		return data["train"], data["val"]
	generator = torch.Generator().manual_seed(seed)
	indices = torch.randperm(total_length, generator=generator).tolist()
	val_idx = indices[:val_count]
	train_idx = indices[val_count:]
	with split_file.open("w", encoding="utf-8") as fp:
		json.dump({"train": train_idx, "val": val_idx}, fp)
	return train_idx, val_idx



class CorruptedDataset(Dataset):
	def __init__(
		self,
		base: Dataset,
		corruption_types: Sequence[str],
		mean: Sequence[float],
		std: Sequence[float],
		severity: int = 2,
		seed: int = 0,
	) -> None:
		self.base = base
		self.corruption_types = list(corruption_types)
		self.mean = mean
		self.std = std
		self.severity = severity
		self.rng = np.random.default_rng(seed)
		if apply_image_corruption is None:
			raise ImportError("imagecorruptions is required for corruption experiments")

	def __len__(self) -> int:
		return len(self.base)

	def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
		img, label = self.base[idx]
		np_img = tensor_to_uint8(img, self.mean, self.std)
		corr = self.rng.choice(self.corruption_types)
		corrupted = apply_image_corruption(np_img, severity=self.severity, corruption_name=corr)
		torch_img = uint8_to_normalized(corrupted, self.mean, self.std)
		return torch_img, label


def load_dataset(config: ExperimentConfig) -> Tuple[Dataset, Dataset, Dataset]:
	dataset_name = config.dataset.lower()
	stats = DATASET_STATS[dataset_name]
	image_size = stats["image_size"]
	train_transform = build_transforms(dataset_name, image_size, augment=True, is_train=True)
	eval_transform = build_transforms(dataset_name, image_size, augment=False, is_train=False)

	if dataset_name == "cifar10":
		train_full = datasets.CIFAR10(config.data_dir, train=True, download=True, transform=train_transform)
		val_source = datasets.CIFAR10(config.data_dir, train=True, download=True, transform=eval_transform)
		test_dataset = datasets.CIFAR10(config.data_dir, train=False, download=True, transform=eval_transform)
	elif dataset_name == "fmnist":
		train_full = datasets.FashionMNIST(config.data_dir, train=True, download=True, transform=train_transform)
		val_source = datasets.FashionMNIST(config.data_dir, train=True, download=True, transform=eval_transform)
		test_dataset = datasets.FashionMNIST(config.data_dir, train=False, download=True, transform=eval_transform)
	elif dataset_name == "imagenet100":
		if not config.imagenet_root:
			raise ValueError("ImageNet-100 requires --imagenet_root pointing to ImageFolder structure")
		train_full = ImageFolder(Path(config.imagenet_root) / "train", transform=train_transform)
		val_source = ImageFolder(Path(config.imagenet_root) / "train", transform=eval_transform)
		test_dataset = ImageFolder(Path(config.imagenet_root) / "val", transform=eval_transform)
	else:
		raise ValueError(f"Unsupported dataset {dataset_name}")

	train_idx, val_idx = get_or_create_split_indices(
		len(train_full),
		dataset_name,
		Path(config.split_dir),
		config.val_split,
		config.seed,
	)
	train_dataset = Subset(train_full, train_idx)
	val_dataset = Subset(val_source, val_idx)

	if config.use_corrupted_validation and config.corruption_types:
		val_dataset = CorruptedDataset(
			val_dataset,
			config.corruption_types,
			stats["mean"],
			stats["std"],
			config.corruption_severity,
			config.seed,
		)

	return train_dataset, val_dataset, test_dataset


def make_dataloaders(
	train_set: Dataset,
	val_set: Dataset,
	test_set: Dataset,
	batch_size: int,
	num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
	val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	return train_loader, val_loader, test_loader


def build_model(config: ExperimentConfig) -> nn.Module:
	dataset_name = config.dataset.lower()
	stats = DATASET_STATS[dataset_name]
	if config.model not in MODEL_FACTORIES:
		raise ValueError(f"Unknown model {config.model}")
	model = MODEL_FACTORIES[config.model](stats["num_classes"])
	return model


def train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	optimizer: optim.Optimizer,
	device: str,
) -> Tuple[float, float]:
	model.train()
	epoch_loss = 0.0
	correct = 0
	total = 0
	for images, labels in loader:
		images = images.to(device)
		labels = labels.to(device)
		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item() * images.size(0)
		preds = outputs.argmax(dim=1)
		correct += (preds == labels).sum().item()
		total += labels.size(0)
	return epoch_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> Tuple[float, float]:
	model.eval()
	epoch_loss = 0.0
	correct = 0
	total = 0
	for images, labels in loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		loss = criterion(outputs, labels)
		epoch_loss += loss.item() * images.size(0)
		preds = outputs.argmax(dim=1)
		correct += (preds == labels).sum().item()
		total += labels.size(0)
	return epoch_loss / total, correct / total


def evaluate_with_corruptions(
	model: nn.Module,
	loader: DataLoader,
	device: str,
	mean: Sequence[float],
	std: Sequence[float],
	corruption_types: Sequence[str],
	severity: int,
) -> Dict[str, float]:
	if apply_image_corruption is None:
		raise ImportError("imagecorruptions is required for corrupted accuracy evaluation")
	model.eval()
	results: Dict[str, float] = {}
	for corruption in corruption_types:
		correct = 0
		total = 0
		for images, labels in loader:
			np_imgs = [tensor_to_uint8(img, mean, std) for img in images]
			corrupted = [
				apply_image_corruption(img, corruption_name=corruption, severity=severity)
				for img in np_imgs
			]
			tensor_batch = torch.stack(
				[uint8_to_normalized(img, mean, std) for img in corrupted]
			).to(device)
			labels = labels.to(device)
			outputs = model(tensor_batch)
			preds = outputs.argmax(dim=1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)
		results[corruption] = correct / total
	return results


def collect_features(
	model: nn.Module,
	loader: DataLoader,
	device: str,
	max_batches: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
	model.eval()
	features: List[np.ndarray] = []
	labels: List[np.ndarray] = []
	batches = 0
	with torch.no_grad():
		for images, labs in loader:
			images = images.to(device)
			outputs = model(images)
			feats = outputs.detach().cpu().numpy()
			features.append(feats)
			labels.append(labs.numpy())
			batches += 1
			if batches >= max_batches:
				break
	return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def plot_tsne(
	embeddings: np.ndarray,
	labels: np.ndarray,
	out_path: Path,
	title: str,
) -> None:
	perplexity = min(30, max(5, embeddings.shape[0] // 5))
	tsne = TSNE(n_components=2, perplexity=perplexity, init="random", learning_rate="auto", n_iter=1000)
	reduced = tsne.fit_transform(embeddings)
	plt.figure(figsize=(6, 5))
	scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab20", s=5, alpha=0.6)
	plt.title(title)
	plt.colorbar(scatter, fraction=0.046, pad=0.04)
	ensure_dir(out_path.parent)
	plt.tight_layout()
	plt.savefig(out_path, dpi=200)
	plt.close()


def capture_feature_maps(
	model: nn.Module,
	images: Tensor,
	layer: nn.Module,
	device: str,
	out_path: Path,
) -> None:
	activations: List[Tensor] = []

	def hook(_module, _inp, output):
		activations.append(output.detach().cpu())

	handle = layer.register_forward_hook(hook)
	with torch.no_grad():
		model(images.to(device))
	handle.remove()
	if not activations:
		return
	fmap = activations[0]
	if fmap.ndim < 4:
		return
	num_maps = min(16, fmap.shape[1])
	grid = fmap[: num_maps].numpy()
	fig, axes = plt.subplots(4, 4, figsize=(6, 6))
	for idx, ax in enumerate(axes.flat):
		if idx < num_maps:
			ax.imshow(grid[idx], cmap="viridis")
		ax.axis("off")
	plt.suptitle("Feature maps")
	ensure_dir(out_path.parent)
	plt.tight_layout()
	plt.savefig(out_path, dpi=200)
	plt.close()


def log_metrics(
	log_dir: Path,
	config: ExperimentConfig,
	train_metrics: List[Dict[str, float]],
	val_metrics: List[Dict[str, float]],
	test_accuracy: float,
	corruption_results: Optional[Dict[str, float]],
) -> None:
	ensure_dir(log_dir)
	result = {
		"config": vars(config),
		"train_epochs": train_metrics,
		"val_epochs": val_metrics,
		"test_accuracy": test_accuracy,
		"corruption_accuracy": corruption_results,
	}
	out_file = log_dir / f"{config.dataset}_{config.model}.json"
	with out_file.open("w", encoding="utf-8") as fp:
		json.dump(result, fp, indent=2)


def optimize_validation_corruptions(
	model: nn.Module,
	val_loader: DataLoader,
	mean: Sequence[float],
	std: Sequence[float],
	corruption_types: Sequence[str],
	severity: int,
	device: str,
	target_drop: float = 0.05,
) -> List[str]:
	if apply_image_corruption is None:
		raise ImportError("imagecorruptions is required for corruption optimization")
	_, baseline = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
	drops: List[Tuple[str, float]] = []
	for corr in corruption_types:
		acc = evaluate_with_corruptions(model, val_loader, device, mean, std, [corr], severity)[corr]
		drops.append((corr, baseline - acc))
	drops.sort(key=lambda item: item[1], reverse=True)
	cumulative = 0.0
	selected: List[str] = []
	for corr, drop in drops:
		selected.append(corr)
		cumulative += drop
		if cumulative >= target_drop:
			break
	return selected or [drops[0][0]]


def run_experiment(config: ExperimentConfig) -> None:
	set_seed(config.seed)
	log_dir = ensure_dir(Path(config.log_dir))
	stats = DATASET_STATS[config.dataset.lower()]
	train_set, val_set, test_set = load_dataset(config)
	train_loader, val_loader, test_loader = make_dataloaders(
		train_set, val_set, test_set, config.batch_size, config.num_workers
	)
	model = build_model(config).to(config.device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
	train_metrics: List[Dict[str, float]] = []
	val_metrics: List[Dict[str, float]] = []
	best_val = -math.inf
	best_state: Optional[Dict[str, Tensor]] = None
	for epoch in range(config.epochs):
		train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.device)
		val_loss, val_acc = evaluate(model, val_loader, criterion, config.device)
		scheduler.step()
		train_metrics.append({"epoch": epoch, "loss": train_loss, "acc": train_acc})
		val_metrics.append({"epoch": epoch, "loss": val_loss, "acc": val_acc})
		if val_acc > best_val:
			best_val = val_acc
			best_state = model.state_dict()
	if best_state:
		model.load_state_dict(best_state)
	_, test_acc = evaluate(model, test_loader, criterion, config.device)
	corruption_results = evaluate_with_corruptions(
		model,
		test_loader,
		config.device,
		stats["mean"],
		stats["std"],
		config.corruption_types or CORRUPTION_SET,
		config.corruption_severity,
	)
	log_metrics(log_dir, config, train_metrics, val_metrics, test_acc, corruption_results)

	embeddings, labels = collect_features(model, val_loader, config.device)
	plot_tsne(embeddings, labels, log_dir / f"tsne_{config.dataset}_{config.model}.png", "Validation t-SNE")
	sample_images, _ = next(iter(test_loader))
	layer = next(model.children())
	capture_feature_maps(
		model,
		sample_images[:8],
		layer if isinstance(layer, nn.Module) else model,
		config.device,
		log_dir / f"feature_maps_{config.dataset}_{config.model}.png",
	)


def run_mlp_validation_regimes(config: ExperimentConfig) -> None:
	regimes = [
		("clean", False, None),
		("corrupted", True, CORRUPTION_SET),
		("optimized", True, None),
	]
	for tag, use_corruption, corr_types in regimes:
		cfg = dataclass_replace(
			config,
			use_corrupted_validation=use_corruption,
			corruption_types=corr_types,
			model="mlp",
		)
		set_seed(cfg.seed)
		train_set, val_set, test_set = load_dataset(cfg)
		train_loader, val_loader, test_loader = make_dataloaders(
			train_set, val_set, test_set, cfg.batch_size, cfg.num_workers
		)
		model = build_model(cfg).to(cfg.device)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
		for epoch in range(cfg.epochs // 2):
			train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
		if tag == "optimized":
			selected = optimize_validation_corruptions(
				model,
				val_loader,
				DATASET_STATS[cfg.dataset.lower()]["mean"],
				DATASET_STATS[cfg.dataset.lower()]["std"],
				CORRUPTION_SET,
				cfg.corruption_severity,
				cfg.device,
			)
			cfg.corruption_types = selected
		run_experiment(cfg)


def dataclass_replace(config: ExperimentConfig, **kwargs) -> ExperimentConfig:
	data = vars(config).copy()
	data.update(kwargs)
	return ExperimentConfig(**data)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Robustness assignment runner")
	parser.add_argument("--dataset", choices=list(DATASET_STATS.keys()), required=True)
	parser.add_argument(
		"--model",
		choices=list(MODEL_FACTORIES.keys()),
		required=True,
		help="Model architecture to train",
	)
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--weight_decay", type=float, default=5e-4)
	parser.add_argument("--val_split", type=float, default=0.2)
	parser.add_argument("--use_corrupted_validation", action="store_true")
	parser.add_argument("--optimize_validation_corruption", action="store_true")
	parser.add_argument("--corruption_types", nargs="*", default=None)
	parser.add_argument("--corruption_severity", type=int, default=2)
	parser.add_argument("--imagenet_root", type=str, default=None)
	parser.add_argument("--log_dir", type=str, default="results")
	parser.add_argument("--split_dir", type=str, default="splits")
	parser.add_argument("--data_dir", type=str, default="data")
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--mlp_validation_suite", action="store_true")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	config = ExperimentConfig(
		dataset=args.dataset,
		model=args.model,
		epochs=args.epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		weight_decay=args.weight_decay,
		val_split=args.val_split,
		corruption_types=args.corruption_types,
		corruption_severity=args.corruption_severity,
		use_corrupted_validation=args.use_corrupted_validation,
		optimize_validation_corruption=args.optimize_validation_corruption,
		log_dir=args.log_dir,
		split_dir=args.split_dir,
		data_dir=args.data_dir,
		imagenet_root=args.imagenet_root,
		num_workers=args.num_workers,
		seed=args.seed,
	)
	if args.mlp_validation_suite:
		run_mlp_validation_regimes(config)
	else:
		run_experiment(config)


if __name__ == "__main__":
	main()
