"""Train ConvNeXt-Tiny on CIFAR-10 for robustness experiments."""

import os

from Assignmnt2 import ExperimentConfig, run_experiment


def main() -> None:

    config = ExperimentConfig(
        dataset="cifar10",
        model="convnext_tiny",
        epochs=int(os.environ.get("EPOCHS", 30)),
        batch_size=int(os.environ.get("BATCH_SIZE", 128)),
        lr=float(os.environ.get("LR", 3e-4)),
        weight_decay=float(os.environ.get("WEIGHT_DECAY", 5e-4)),
        val_split=float(os.environ.get("VAL_SPLIT", 0.2)),
        corruption_severity=int(os.environ.get("CORRUPTION_SEVERITY", 2)),
        log_dir=os.environ.get("LOG_DIR", "results/cifar10_convnext_tiny"),
        split_dir=os.environ.get("SPLIT_DIR", "splits"),
        data_dir=os.environ.get("DATA_DIR", "data"),
        num_workers=int(os.environ.get("NUM_WORKERS", 4)),
        seed=int(os.environ.get("SEED", 42)),
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
