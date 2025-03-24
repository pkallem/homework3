import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .models import Detector, save_model
from .metrics import DetectionMetric
from .datasets.road_dataset import load_data


def train_detection(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------------
    # 1) Create the model
    # ----------------------------
    model = Detector(in_channels=3, num_classes=3)
    model.to(device)

    # ----------------------------
    # 2) Create losses + metrics
    # ----------------------------
    # We combine segmentation loss + depth loss
    seg_criterion = nn.CrossEntropyLoss()
    depth_criterion = nn.L1Loss()  # can also try nn.MSELoss
    metric = DetectionMetric(num_classes=3)

    # ----------------------------
    # 3) Create optimizer
    # ----------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ----------------------------
    # 4) Load data
    # ----------------------------
    # Just using 'default' pipeline here.
    # If you want data augmentation, define it inside road_dataset/RoadDataset.
    train_loader = load_data(
        dataset_path=args.train_path,
        transform_pipeline="default",
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = load_data(
        dataset_path=args.val_path,
        transform_pipeline="default",
        batch_size=args.batch_size,
        shuffle=False
    )

    # ----------------------------
    # 5) (Optional) Set up logging
    # ----------------------------
    writer = SummaryWriter(log_dir=args.log_dir)
    global_step = 0

    # ----------------------------
    # 6) Training loop
    # ----------------------------
    for epoch in range(args.num_epochs):
        model.train()
        metric.reset()

        # -- Train --
        for batch in train_loader:
            images = batch["image"].to(device)      # (B,3,H,W)
            seg_labels = batch["track"].to(device)  # (B,H,W) with {0,1,2}
            depth_labels = batch["depth"].to(device)# (B,H,W) in [0,1]

            optimizer.zero_grad()
            logits, raw_depth = model(images)  # logits: (B,3,H,W); depth: (B,H,W)

            seg_loss = seg_criterion(logits, seg_labels)
            depth_loss = depth_criterion(raw_depth, depth_labels)
            loss = seg_loss + depth_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # For training metrics, we can accumulate on the fly
                preds = logits.argmax(dim=1)  # (B,H,W)
                metric.add(preds, seg_labels, raw_depth, depth_labels)

            global_step += 1

        train_metrics = metric.compute()

        # -- Validate --
        model.eval()
        metric.reset()
        with torch.inference_mode():
            for batch in val_loader:
                images = batch["image"].to(device)
                seg_labels = batch["track"].to(device)
                depth_labels = batch["depth"].to(device)

                logits, raw_depth = model(images)
                preds = logits.argmax(dim=1)
                metric.add(preds, seg_labels, raw_depth, depth_labels)

        val_metrics = metric.compute()

        # -- Logging --
        writer.add_scalar("Loss/train", loss.item(), epoch)
        writer.add_scalar("IoU/train", train_metrics["iou"], epoch)
        writer.add_scalar("IoU/val", val_metrics["iou"], epoch)

        writer.add_scalar("AbsDepthError/train", train_metrics["abs_depth_error"], epoch)
        writer.add_scalar("AbsDepthError/val", val_metrics["abs_depth_error"], epoch)

        writer.add_scalar("TPDepthError/train", train_metrics["tp_depth_error"], epoch)
        writer.add_scalar("TPDepthError/val", val_metrics["tp_depth_error"], epoch)

        print(f"Epoch {epoch+1}/{args.num_epochs} | "
              f"Train IoU: {train_metrics['iou']:.4f}, Val IoU: {val_metrics['iou']:.4f}, "
              f"TrainDepthErr: {train_metrics['abs_depth_error']:.4f}, "
              f"ValDepthErr: {val_metrics['abs_depth_error']:.4f}")

    # ----------------------------
    # 7) Save your model
    # ----------------------------
    save_model(model)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="road_data/train")
    parser.add_argument("--val_path", type=str, default="road_data/val")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_dir", type=str, default="logs/detection")
    args = parser.parse_args()

    train_detection(args)
