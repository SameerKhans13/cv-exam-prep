"""Session 04 — Transfer learning skeleton: feature extraction vs fine-tuning (ResNet-18).

Run:  python transfer_learning_skeleton.py --mode feature              # synthetic FakeData, no downloads
      python transfer_learning_skeleton.py --mode finetune --pretrained # downloads ImageNet weights (~45 MB)
      python transfer_learning_skeleton.py --data path/to/imagefolder --pretrained   # real data: root/{train,val}/class/*.jpg
feature  = freeze the backbone, train only the new head (small data / similar domain)
finetune = unfreeze everything, low LR for backbone, higher LR for head (more data / different domain)
"""
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms as T

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["feature", "finetune"], default="feature")
ap.add_argument("--pretrained", action="store_true")
ap.add_argument("--data", default=None)
ap.add_argument("--epochs", type=int, default=2)
a = ap.parse_args()

# 1. Data: augmentation on TRAIN only; ImageNet normalisation because the backbone was trained with it
norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_train = T.Compose([T.RandomResizedCrop(112), T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2), T.ToTensor(), norm])
tf_val = T.Compose([T.Resize(128), T.CenterCrop(112), T.ToTensor(), norm])
if a.data:
    train_ds = datasets.ImageFolder(f"{a.data}/train", tf_train)
    val_ds = datasets.ImageFolder(f"{a.data}/val", tf_val)
else:  # random images: accuracy stays ~chance; the point is the mechanics
    train_ds = datasets.FakeData(160, (3, 128, 128), num_classes=10, transform=tf_train)
    val_ds = datasets.FakeData(40, (3, 128, 128), num_classes=10, transform=tf_val, random_offset=1)
n_cls = len(train_ds.classes) if a.data else 10
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32)

# 2. Model: pretrained backbone + NEW classification head sized for our classes
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if a.pretrained else None)
if a.mode == "feature":
    for p in model.parameters():
        p.requires_grad = False                     # freeze everything...
model.fc = nn.Linear(model.fc.in_features, n_cls)   # ...new head is created trainable (512*n_cls + n_cls params)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"mode={a.mode}  trainable params={trainable:,} / total={sum(p.numel() for p in model.parameters()):,}")

# 3. Optimiser: feature mode -> head only; finetune -> discriminative LRs (backbone 10x smaller)
if a.mode == "feature":
    opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
else:
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    opt = torch.optim.AdamW([{"params": backbone, "lr": 1e-4}, {"params": model.fc.parameters(), "lr": 1e-3}],
                            weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# 4. Train / validate
for epoch in range(a.epochs):
    model.train()
    if a.mode == "feature":                         # frozen backbone: keep BatchNorm running stats frozen too
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    for x, y in train_dl:
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    model.eval()                                    # BN uses running stats, dropout off
    with torch.no_grad():
        correct = sum((model(x).argmax(1) == y).sum().item() for x, y in val_dl)
    print(f"epoch {epoch + 1}: last train loss={loss.item():.3f}  val acc={correct / len(val_ds):.3f}")
