#!/usr/bin/env python3
"""Train a NumPy MLP on EuroSAT RGB without deep-learning frameworks."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_or_build_cache(data_dir: Path, image_size: int, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"eurosat_{image_size}px_uint8.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return data["x"], data["y"], data["paths"].tolist()

    images: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    for label, name in enumerate(CLASS_NAMES):
        class_dir = data_dir / name
        for path in sorted(class_dir.glob("*.jpg")):
            with Image.open(path) as img:
                img = img.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
                images.append(np.asarray(img, dtype=np.uint8))
            labels.append(label)
            paths.append(str(path))

    x = np.stack(images, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(cache_path, x=x, y=y, paths=np.asarray(paths))
    return x, y, paths


def stratified_split(
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for label in range(len(CLASS_NAMES)):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_train = int(len(idx) * train_ratio)
        n_val = int(len(idx) * val_ratio)
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val :].tolist())

    for idxs in (train_idx, val_idx, test_idx):
        rng.shuffle(idxs)
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def prepare_data(data_dir: Path, image_size: int, seed: int, cache_dir: Path) -> tuple[SplitData, list[str]]:
    x_uint8, y, paths = load_or_build_cache(data_dir, image_size, cache_dir)
    x = x_uint8.astype(np.float32).reshape(len(x_uint8), -1) / 255.0
    train_idx, val_idx, test_idx = stratified_split(y, train_ratio=0.7, val_ratio=0.15, seed=seed)

    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True) + 1e-6
    split = SplitData(
        x_train=(x[train_idx] - mean) / std,
        y_train=y[train_idx],
        x_val=(x[val_idx] - mean) / std,
        y_val=y[val_idx],
        x_test=(x[test_idx] - mean) / std,
        y_test=y[test_idx],
        mean=mean,
        std=std,
    )
    return split, [paths[i] for i in test_idx]


def he_or_xavier(rng: np.random.Generator, fan_in: int, fan_out: int, activation: str) -> np.ndarray:
    if activation == "relu":
        scale = math.sqrt(2.0 / fan_in)
    else:
        scale = math.sqrt(1.0 / fan_in)
    return rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32)


class MLP:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str,
        seed: int,
    ) -> None:
        if activation not in {"relu", "tanh", "sigmoid"}:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activation
        self.rng = np.random.default_rng(seed)
        dims = [input_dim] + hidden_dims + [output_dim]
        self.params: dict[str, np.ndarray] = {}
        for i in range(len(dims) - 1):
            act_for_init = activation if i < len(dims) - 2 else "linear"
            self.params[f"W{i+1}"] = he_or_xavier(self.rng, dims[i], dims[i + 1], act_for_init)
            self.params[f"b{i+1}"] = np.zeros((1, dims[i + 1]), dtype=np.float32)

    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(z, 0.0)
        if self.activation == "tanh":
            return np.tanh(z)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

    def _activation_grad(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (z > 0).astype(np.float32)
        if self.activation == "tanh":
            return 1.0 - a * a
        return a * (1.0 - a)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        caches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        a = x
        n_layers = len(self.params) // 2
        for i in range(1, n_layers):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            next_a = self._activate(z)
            caches.append((a, z, next_a))
            a = next_a
        logits = a @ self.params[f"W{n_layers}"] + self.params[f"b{n_layers}"]
        caches.append((a, logits, logits))
        return logits, caches

    def loss_and_grads(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weight_decay: float,
    ) -> tuple[float, dict[str, np.ndarray]]:
        logits, caches = self.forward(x)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        n = x.shape[0]
        data_loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
        l2 = 0.0
        for key, value in self.params.items():
            if key.startswith("W"):
                l2 += 0.5 * weight_decay * float(np.sum(value * value))
        loss = data_loss + l2

        grads: dict[str, np.ndarray] = {}
        dout = probs
        dout[np.arange(n), y] -= 1.0
        dout /= n

        n_layers = len(self.params) // 2
        for i in range(n_layers, 0, -1):
            a_prev, z, a = caches[i - 1]
            grads[f"W{i}"] = a_prev.T @ dout + weight_decay * self.params[f"W{i}"]
            grads[f"b{i}"] = dout.sum(axis=0, keepdims=True)
            if i > 1:
                da_prev = dout @ self.params[f"W{i}"].T
                prev_z = caches[i - 2][1]
                prev_a = caches[i - 2][2]
                dout = da_prev * self._activation_grad(prev_z, prev_a)
        return float(loss), grads

    def predict(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        preds = []
        for start in range(0, len(x), batch_size):
            logits, _ = self.forward(x[start : start + batch_size])
            preds.append(np.argmax(logits, axis=1))
        return np.concatenate(preds)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, activation=self.activation, **self.params)

    def load_params(self, arrays: dict[str, np.ndarray]) -> None:
        for key in self.params:
            self.params[key][...] = arrays[key]


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred == y))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def train_one(
    split: SplitData,
    hidden_dims: list[int],
    activation: str,
    lr: float,
    lr_decay: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
    momentum: float = 0.0,
) -> tuple[MLP, dict[str, list[float]], dict[str, np.ndarray], float]:
    model = MLP(split.x_train.shape[1], hidden_dims, len(CLASS_NAMES), activation, seed)
    history: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val = -1.0
    best_params: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(seed)
    current_lr = lr
    velocity = {key: np.zeros_like(value) for key, value in model.params.items()}

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(len(split.x_train))
        losses = []
        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start : start + batch_size]
            xb = split.x_train[batch_idx]
            yb = split.y_train[batch_idx]
            loss, grads = model.loss_and_grads(xb, yb, weight_decay)
            losses.append(loss)
            for key in model.params:
                velocity[key] = momentum * velocity[key] - current_lr * grads[key]
                model.params[key] += velocity[key]

        train_pred = model.predict(split.x_train)
        val_pred = model.predict(split.x_val)
        val_loss, _ = model.loss_and_grads(split.x_val, split.y_val, weight_decay)
        train_acc = accuracy(train_pred, split.y_train)
        val_acc = accuracy(val_pred, split.y_val)
        history["train_loss"].append(float(np.mean(losses)))
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"epoch {epoch:02d}/{epochs} loss={history['train_loss'][-1]:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} lr={current_lr:.5f}",
            flush=True,
        )
        if val_acc > best_val:
            best_val = val_acc
            best_params = {key: value.copy() for key, value in model.params.items()}
        current_lr *= lr_decay

    model.load_params(best_params)
    return model, history, best_params, best_val


def plot_curves(history: dict[str, list[float]], out_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, out_path: Path) -> None:
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sum, 1)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=140)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_first_layer_weights(model: MLP, image_size: int, out_path: Path) -> None:
    w = model.params["W1"]
    norms = np.linalg.norm(w, axis=0)
    selected = np.argsort(norms)[-16:][::-1]
    fig, axes = plt.subplots(4, 4, figsize=(6, 6), dpi=150)
    for ax, idx in zip(axes.ravel(), selected):
        filt = w[:, idx].reshape(image_size, image_size, 3)
        filt = filt - filt.min()
        filt = filt / (filt.max() + 1e-8)
        ax.imshow(filt)
        ax.set_title(f"h{idx}", fontsize=8)
        ax.axis("off")
    fig.suptitle("First Hidden Layer Weights", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_error_examples(
    x_test: np.ndarray,
    y_test: np.ndarray,
    pred: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    image_size: int,
    out_path: Path,
    max_examples: int = 12,
) -> list[dict[str, str]]:
    wrong = np.flatnonzero(y_test != pred)[:max_examples]
    examples: list[dict[str, str]] = []
    if len(wrong) == 0:
        return examples
    cols = 4
    rows = int(math.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.8 * rows), dpi=140)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, idx in zip(axes_arr, wrong):
        img = x_test[idx : idx + 1] * std + mean
        img = np.clip(img.reshape(image_size, image_size, 3), 0, 1)
        ax.imshow(img)
        true_name = CLASS_NAMES[int(y_test[idx])]
        pred_name = CLASS_NAMES[int(pred[idx])]
        ax.set_title(f"T: {true_name}\nP: {pred_name}", fontsize=8)
        ax.axis("off")
        examples.append({"true": true_name, "pred": pred_name})
    for ax in axes_arr[len(wrong) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("EuroSAT_RGB"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search", action="store_true", help="Run the default hyperparameter search.")
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split, _test_paths = prepare_data(args.data_dir, args.image_size, args.seed, args.out_dir / "cache")
    print(
        f"loaded train={len(split.y_train)} val={len(split.y_val)} test={len(split.y_test)} "
        f"input_dim={split.x_train.shape[1]}",
        flush=True,
    )

    if args.search:
        configs = [
            {
                "hidden_dims": [384, 96],
                "activation": "relu",
                "lr": 0.010,
                "lr_decay": 0.97,
                "weight_decay": 1e-3,
                "momentum": 0.9,
                "seed": 44,
            },
            {
                "hidden_dims": [256, 64],
                "activation": "relu",
                "lr": 0.012,
                "lr_decay": 0.97,
                "weight_decay": 1e-3,
                "momentum": 0.9,
                "seed": 55,
            },
            {
                "hidden_dims": [256, 64],
                "activation": "relu",
                "lr": 0.012,
                "lr_decay": 0.97,
                "weight_decay": 7e-4,
                "momentum": 0.9,
                "seed": 44,
            },
            {
                "hidden_dims": [256, 64],
                "activation": "relu",
                "lr": 0.035,
                "lr_decay": 0.95,
                "weight_decay": 7e-4,
                "momentum": 0.0,
                "seed": 44,
            },
            {
                "hidden_dims": [256, 128],
                "activation": "tanh",
                "lr": 0.015,
                "lr_decay": 0.95,
                "weight_decay": 1e-4,
                "momentum": 0.0,
                "seed": 47,
            },
        ]
    else:
        configs = [
            {
                "hidden_dims": [384, 96],
                "activation": "relu",
                "lr": 0.010,
                "lr_decay": 0.97,
                "weight_decay": 1e-3,
                "momentum": 0.9,
                "seed": 44,
            }
        ]

    runs = []
    best_run = None
    for run_id, cfg in enumerate(configs, start=1):
        print(f"\nrun {run_id}: {cfg}", flush=True)
        model, history, _best_params, best_val = train_one(
            split=split,
            hidden_dims=cfg["hidden_dims"],
            activation=cfg["activation"],
            lr=cfg["lr"],
            lr_decay=cfg["lr_decay"],
            weight_decay=cfg["weight_decay"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=cfg.get("seed", args.seed + run_id),
            momentum=cfg.get("momentum", 0.0),
        )
        run_dir = args.out_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model.save(run_dir / "best_model.npz")
        plot_curves(history, run_dir / "learning_curves.png")
        result = {"run_id": run_id, **cfg, "best_val_acc": best_val, "history": history}
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        runs.append(result)
        if best_run is None or best_val > best_run["best_val_acc"]:
            best_run = {"model": model, "history": history, "config": cfg, "run_id": run_id, "best_val_acc": best_val}

    assert best_run is not None
    model = best_run["model"]
    test_pred = model.predict(split.x_test)
    test_acc = accuracy(test_pred, split.y_test)
    cm = confusion_matrix(split.y_test, test_pred, len(CLASS_NAMES))

    plot_curves(best_run["history"], args.out_dir / "learning_curves.png")
    plot_confusion(cm, args.out_dir / "confusion_matrix.png")
    plot_first_layer_weights(model, args.image_size, args.out_dir / "first_layer_weights.png")
    error_examples = save_error_examples(
        split.x_test,
        split.y_test,
        test_pred,
        split.mean,
        split.std,
        args.image_size,
        args.out_dir / "error_examples.png",
    )
    model.save(args.out_dir / "best_model.npz")

    summary = {
        "image_size": args.image_size,
        "class_names": CLASS_NAMES,
        "split_sizes": {"train": int(len(split.y_train)), "val": int(len(split.y_val)), "test": int(len(split.y_test))},
        "best_run": {
            "run_id": best_run["run_id"],
            "config": best_run["config"],
            "best_val_acc": best_run["best_val_acc"],
            "test_acc": test_acc,
        },
        "search_results": runs,
        "confusion_matrix": cm.tolist(),
        "error_examples": error_examples,
    }
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nbest run={best_run['run_id']} val_acc={best_run['best_val_acc']:.4f} test_acc={test_acc:.4f}")
    print(f"saved outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
