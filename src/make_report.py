#!/usr/bin/env python3
"""Build the homework PDF report from generated experiment outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from PIL import Image as PILImage


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def add_image(story: list, path: Path, width_cm: float, title: str | None = None) -> None:
    if title:
        story.append(Paragraph(title, STYLES["Heading2"]))
    if path.exists():
        with PILImage.open(path) as source:
            src_w, src_h = source.size
        width = width_cm * cm
        height = width * src_h / src_w
        max_height = 20 * cm
        if height > max_height:
            scale = max_height / height
            width *= scale
            height *= scale
        img = RLImage(str(path), width=width, height=height)
        story.append(img)
        story.append(Spacer(1, 0.3 * cm))


STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle(name="Small", parent=STYLES["BodyText"], fontSize=8, leading=10))


def main() -> None:
    out_dir = Path("outputs")
    with open(out_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    report_path = out_dir / "hw1_report.pdf"
    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    story: list = []
    best = summary["best_run"]
    cfg = best["config"]
    repo_url = os.environ.get("GITHUB_REPO_URL", "https://github.com/huang200309/eurosat-numpy-mlp-hw1")
    weights_url = os.environ.get(
        "MODEL_WEIGHTS_URL",
        "https://github.com/huang200309/eurosat-numpy-mlp-hw1/raw/main/outputs/best_model.npz",
    )

    story.append(Paragraph("HW1: Three-Layer NumPy MLP for EuroSAT Land-Cover Classification", STYLES["Title"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("1. Task and Dataset", STYLES["Heading1"]))
    story.append(
        Paragraph(
            "This work implements a multilayer perceptron classifier from scratch with NumPy only. "
            "The model is trained on EuroSAT_RGB, a 10-class remote-sensing land-cover dataset. "
            f"Images are resized to {summary['image_size']}x{summary['image_size']} RGB, flattened, "
            "scaled to [0, 1], and standardized using the training-set mean and standard deviation.",
            STYLES["BodyText"],
        )
    )
    split = summary["split_sizes"]
    story.append(
        Paragraph(
            f"The deterministic stratified split is train/validation/test = "
            f"{split['train']}/{split['val']}/{split['test']}.",
            STYLES["BodyText"],
        )
    )
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("2. Model and Optimization", STYLES["Heading1"]))
    story.append(
        Paragraph(
            "The network uses fully-connected layers input -> hidden1 -> hidden2 -> softmax output. "
            "Forward propagation, back propagation, softmax cross-entropy, L2 weight decay, SGD/momentum updates, "
            "and learning-rate decay are all implemented manually. The validation-set accuracy is used "
            "to save the best model weights during training.",
            STYLES["BodyText"],
        )
    )
    model_table = [
        ["Item", "Value"],
        ["Best run", str(best["run_id"])],
        ["Hidden dimensions", str(cfg["hidden_dims"])],
        ["Activation", cfg["activation"]],
        ["Initial learning rate", str(cfg["lr"])],
        ["Learning-rate decay", str(cfg["lr_decay"])],
        ["L2 weight decay", str(cfg["weight_decay"])],
        ["Momentum", str(cfg.get("momentum", 0.0))],
        ["Best validation accuracy", pct(best["best_val_acc"])],
        ["Test accuracy", pct(best["test_acc"])],
    ]
    table = Table(model_table, hAlign="LEFT", colWidths=[5 * cm, 10 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e8f5")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("3. Hyperparameter Search", STYLES["Heading1"]))
    rows = [["Run", "Hidden", "Act.", "LR", "Decay", "L2", "Mom.", "Best Val. Acc."]]
    for item in summary["search_results"]:
        rows.append(
            [
                str(item["run_id"]),
                str(item["hidden_dims"]),
                item["activation"],
                str(item["lr"]),
                str(item["lr_decay"]),
                str(item["weight_decay"]),
                str(item.get("momentum", 0.0)),
                pct(item["best_val_acc"]),
            ]
        )
    search_table = Table(rows, hAlign="LEFT", repeatRows=1)
    search_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e5e5")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(search_table)
    story.append(PageBreak())

    story.append(Paragraph("4. Training Curves and Test Evaluation", STYLES["Heading1"]))
    add_image(story, out_dir / "learning_curves.png", 16, "Loss and Accuracy Curves")
    story.append(PageBreak())
    add_image(story, out_dir / "confusion_matrix.png", 15, "Confusion Matrix")
    story.append(
        Paragraph(
            "The confusion matrix shows that visually similar categories are the main source of errors. "
            "For example, River and Highway both contain elongated linear structures, while AnnualCrop, "
            "PermanentCrop, Pasture, and HerbaceousVegetation often share green/brown field textures. "
            "The simple MLP has no convolutional inductive bias, so it must learn spatial patterns only "
            "from flattened pixels; this explains why texture-rich or layout-dependent classes remain hard.",
            STYLES["BodyText"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("5. First-Layer Weight Visualization", STYLES["Heading1"]))
    add_image(story, out_dir / "first_layer_weights.png", 13)
    story.append(
        Paragraph(
            "The first hidden-layer weights are reshaped back to RGB image grids. The stronger filters "
            "show broad color and low-frequency spatial preferences instead of sharp local edge detectors. "
            "This is expected for a fully-connected classifier trained on small satellite thumbnails: "
            "early units tend to capture global vegetation, water, soil, and urban color patterns.",
            STYLES["BodyText"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("6. Error Analysis", STYLES["Heading1"]))
    add_image(story, out_dir / "error_examples.png", 16, "Misclassified Test Examples")
    story.append(
        Paragraph(
            "The sampled mistakes confirm the quantitative pattern. Highways can be confused with rivers "
            "because both appear as thin continuous paths. Cropland, pasture, and herbaceous vegetation "
            "also overlap when the image contains mixed land parcels or lacks distinctive structure. "
            "A convolutional model would likely improve these cases, but the assignment intentionally "
            "uses an MLP to practice manual differentiation and optimization.",
            STYLES["BodyText"],
        )
    )

    story.append(Paragraph("7. Reproducibility", STYLES["Heading1"]))
    story.append(
        Paragraph(
            "Run `python3 src/train_numpy_mlp.py --data-dir EuroSAT_RGB --out-dir outputs --search` "
            "from the homework directory to reproduce the training and figures. "
            "Run `python3 src/make_report.py` to regenerate this PDF. "
            "The saved model weights are in `outputs/best_model.npz`.",
            STYLES["BodyText"],
        )
    )
    links_table = Table(
        [["Public GitHub Repo", repo_url], ["Model Weights Download", weights_url]],
        hAlign="LEFT",
        colWidths=[5 * cm, 11 * cm],
    )
    links_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eeeeee")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(links_table)

    doc.build(story)
    print(report_path)


if __name__ == "__main__":
    main()
