import argparse
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from utils import COCODataset, build_model, export_yolo_dataset, resolve_device


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detection model on COCO-format data.")
    parser.add_argument("--coco-json", required=True, help="Path to COCO result.json / annotations.json")
    parser.add_argument("--images-dir", required=True, help="Path to images directory")
    parser.add_argument("--output", required=True, help="Output checkpoint path (.pth)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--val-split", type=float, default=0.0, help="Fraction of data reserved for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--no-pretrained", action="store_true", help="Disable COCO pretrained weights")
    parser.add_argument("--log-every", type=int, default=10, help="Log progress every N batches")
    parser.add_argument(
        "--arch",
        choices=["fasterrcnn", "ssdlite", "yolo"],
        default=None,
        help="Model architecture (default: fasterrcnn, or ssdlite on MPS)",
    )
    parser.add_argument("--max-dim", type=int, default=None, help="Downscale images so the longest side is <= this value (0 disables)")
    parser.add_argument("--min-size", type=int, default=None, help="Model transform min size (default varies by arch/device)")
    parser.add_argument("--max-size", type=int, default=None, help="Model transform max size (default varies by arch/device)")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO image size (default: 640)")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO base model or checkpoint (.pt)")
    return parser.parse_args()


def format_seconds(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def main():
    args = parse_args()
    coco_json = Path(args.coco_json)
    images_dir = Path(args.images_dir)

    device = resolve_device(args.device)
    arch = args.arch
    if arch is None:
        arch = "ssdlite" if device.type == "mps" else "fasterrcnn"
    if arch == "yolo":
        device_arg = "cpu"
        if device.type == "cuda":
            device_arg = "0"
        elif device.type == "mps":
            device_arg = "mps"

        imgsz = args.imgsz or 640
        output_path = Path(args.output)
        dataset_dir = output_path.parent / f"{output_path.stem}_yolo_data"
        data_yaml, class_names = export_yolo_dataset(
            coco_json,
            images_dir,
            dataset_dir,
            val_split=args.val_split,
            seed=args.seed,
        )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise SystemExit("Ultralytics is required for YOLO training. Install with: pip install ultralytics") from exc

        print("Using device:", device)
        print(f"Architecture: yolo | imgsz: {imgsz} | base model: {args.yolo_model}")
        print(f"YOLO dataset: {data_yaml}")

        model = YOLO(args.yolo_model)
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=imgsz,
            batch=args.batch_size,
            device=device_arg,
            project=str(output_path.parent),
            name=output_path.stem,
            exist_ok=True,
        )

        save_dir = Path(getattr(results, "save_dir", ""))
        weights_dir = save_dir / "weights" if save_dir else output_path.parent / output_path.stem / "weights"
        best_path = weights_dir / "best.pt"
        if not best_path.exists():
            best_path = weights_dir / "last.pt"
        if not best_path.exists():
            raise SystemExit(f"YOLO training finished but no weights found in {weights_dir}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, output_path)
        print(f"Saved model to {output_path}")
        return

    max_dim = args.max_dim
    if max_dim is None:
        if arch == "ssdlite":
            max_dim = 320
        else:
            max_dim = 768 if device.type == "mps" else 1024

    min_size = args.min_size
    max_size = args.max_size
    if min_size is None:
        if arch == "ssdlite":
            min_size = 320
        else:
            min_size = 512 if device.type == "mps" else 800
    if max_size is None:
        if arch == "ssdlite":
            max_size = 320
        else:
            max_size = 768 if device.type == "mps" else 1333

    print("Loading dataset...")
    dataset = COCODataset(coco_json, images_dir, train=True, max_dim=max_dim)
    num_classes = len(dataset.class_names) + 1
    print(f"Dataset images: {len(dataset)} | classes: {len(dataset.class_names)}")

    indices = list(range(len(dataset)))
    if args.val_split > 0:
        torch.manual_seed(args.seed)
        indices = torch.randperm(len(dataset)).tolist()
        split = int(len(indices) * (1 - args.val_split))
        train_indices = indices[:split]
        val_indices = indices[split:]
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(COCODataset(coco_json, images_dir, train=False, max_dim=max_dim), val_indices)
    else:
        train_dataset = dataset
        val_dataset = None

    print("Building dataloader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    ssdlite_backbone_weights = None
    if arch == "ssdlite":
        if args.no_pretrained:
            ssdlite_backbone_weights = False
        else:
            ssdlite_backbone_weights = num_classes != 91

    print(f"Using device: {device}")
    print(
        f"Architecture: {arch} | max-dim: {max_dim} | min-size: {min_size} | max-size: {max_size} | "
        f"ssdlite_backbone_weights: {ssdlite_backbone_weights}"
    )

    print("Building model...")
    model = build_model(
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        min_size=min_size,
        max_size=max_size,
        arch=arch,
        ssdlite_backbone_weights=ssdlite_backbone_weights,
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    try:
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            total_batches = len(train_loader)
            epoch_start = time.time()
            print(f"Starting epoch {epoch + 1}/{args.epochs} | batches: {total_batches}")

            for batch_idx, (images, targets) in enumerate(train_loader, start=1):
                if batch_idx == 1:
                    print("First batch loaded.")
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                total_loss += losses.item()

                if args.log_every > 0 and (batch_idx % args.log_every == 0 or batch_idx == total_batches):
                    elapsed = time.time() - epoch_start
                    avg_loss = total_loss / max(1, batch_idx)
                    eta = (elapsed / batch_idx) * (total_batches - batch_idx) if batch_idx else 0
                    lr = optimizer.param_groups[0].get("lr", 0.0)
                    print(
                        f"Epoch {epoch + 1}/{args.epochs} "
                        f"[{batch_idx}/{total_batches}] "
                        f"loss={avg_loss:.4f} "
                        f"lr={lr:.6f} "
                        f"elapsed={format_seconds(elapsed)} "
                        f"eta={format_seconds(eta)}"
                    )

            lr_scheduler.step()
            avg_loss = total_loss / max(1, total_batches)
            print(f"Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f}")

            if val_dataset is not None:
                model.eval()
                with torch.no_grad():
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=collate_fn,
                    )
                    val_batches = 0
                    for images, targets in val_loader:
                        images = [img.to(device) for img in images]
                        _ = model(images)
                        val_batches += 1
                    print(f"Validation batches: {val_batches}")
    except RuntimeError as exc:
        message = str(exc)
        if device.type == "mps":
            print("MPS runtime error detected. Try one of the following:")
            print("- rerun with --device cpu")
            print("- set PYTORCH_ENABLE_MPS_FALLBACK=1 in your shell")
        raise

    checkpoint = {
        "model_state": model.state_dict(),
        "class_names": dataset.class_names,
        "cat_id_to_contig": dataset.cat_id_to_contig,
        "num_classes": num_classes,
        "architecture": arch,
        "ssdlite_backbone_weights": ssdlite_backbone_weights,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
