import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Make sure you import your hw3 versions of Classifier, save_model, etc.
# If you have them in a local module called models, do:
from .models import Classifier, save_model
from .metrics import AccuracyMetric
from .datasets.classification_dataset import load_data


def train_classification(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------------
    # 1) Create the model
    # ----------------------------
    model = Classifier(in_channels=3, num_classes=6)
    model.to(device)

    # ----------------------------
    # 2) Create loss/metrics/optimizer
    # ----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_metric = AccuracyMetric()
    val_metric = AccuracyMetric()

    # ----------------------------
    # 3) Load the data
    # ----------------------------
    # Use 'aug' transform pipeline for training (random flips, etc.)
    train_loader = load_data(
        dataset_path=args.train_path,
        transform_pipeline="aug",     # or any custom pipeline you implemented
        batch_size=args.batch_size,
        shuffle=True
    )

    # Use 'default' transform pipeline for validation
    val_loader = load_data(
        dataset_path=args.val_path,
        transform_pipeline="default",
        batch_size=args.batch_size,
        shuffle=False
    )

    # ----------------------------
    # 4) (Optional) Set up logging
    # ----------------------------
    writer = SummaryWriter(log_dir=args.log_dir)
    global_step = 0

    # ----------------------------
    # 5) Training loop
    # ----------------------------
    for epoch in range(args.num_epochs):
        # -- Train --
        model.train()
        train_metric.reset()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # record training accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                train_metric.add(preds, labels)

            global_step += 1

        train_results = train_metric.compute()

        # -- Validate --
        model.eval()
        val_metric.reset()
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                preds = logits.argmax(dim=1)
                val_metric.add(preds, labels)

        val_results = val_metric.compute()

        # -- Logging --
        writer.add_scalar("Loss/train", loss.item(), epoch)
        writer.add_scalar("Accuracy/train", train_results["accuracy"], epoch)
        writer.add_scalar("Accuracy/val", val_results["accuracy"], epoch)

        print(f"Epoch {epoch+1}/{args.num_epochs} "
              f"Train Acc: {train_results['accuracy']:.4f} | "
              f"Val Acc: {val_results['accuracy']:.4f}")

    # ----------------------------
    # 6) Save your model
    # ----------------------------
    save_model(model)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="classification_data/train")
    parser.add_argument("--val_path", type=str, default="classification_data/val")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_dir", type=str, default="logs/classification")
    args = parser.parse_args()

    train_classification(args)
