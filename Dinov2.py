import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import v2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL
from tqdm import tqdm
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# Parameters
BATCH_SIZE = 64
EPOCHS = 35
PATIENCE = 10
LEARNING_RATE = 1e-4 
LEARNING_RATE_BACKBONE = 1e-5 
CLASS_NUM = 5
NAMING = "DINOv2_vitl14" 
DINOV2_MODEL_NAME = 'dinov2_vitl14'

# Set Random Seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# DINOv2 specific normalization
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)

train_transforms = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
    v2.RandomRotation(degrees=20),
    v2.RandomHorizontalFlip(),
    v2.ToTensor(),
    v2.ConvertImageDtype(torch.float),
    v2.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD)
])

test_transforms = v2.Compose([
    v2.Resize(size=224, interpolation=v2.InterpolationMode.BICUBIC, max_size=None, antialias=True),
    v2.CenterCrop(size=(224, 224)),
    v2.ToTensor(),
    v2.ConvertImageDtype(torch.float),
    v2.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD)
])
cutmix = v2.CutMix(num_classes=CLASS_NUM, alpha=1.0)


class myDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = os.path.join(self.img_dir, row['image_id'])
        image = PIL.Image.open(img_name).convert('RGB')
        label = int(row['label'])

        if self.transform:
            image = self.transform(image)

        return image, label


df = pd.read_csv('data/train.csv')
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)

train_dataset = myDataset(train_df, img_dir='data/train_images', transform=train_transforms)
val_dataset = myDataset(val_df, img_dir='data/train_images', transform=test_transforms)


class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes=CLASS_NUM, dinov2_model_name=DINOV2_MODEL_NAME, freeze_backbone=True):
        super(DINOv2Classifier, self).__init__()
        self.dinov2_model_name = dinov2_model_name
        try:
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', self.dinov2_model_name, pretrained=True)
        except Exception as e:
            print(f"Error loading DINOv2 model: {e}")
            print("Please ensure you have an internet connection and the 'facebookresearch/dinov2' repository is accessible.")
            print("You might also need to install dependencies like 'timm'. Try: pip install timm")
            raise

        if 'vitb' in self.dinov2_model_name:
            embed_dim = 768
        elif 'vits' in self.dinov2_model_name:
            embed_dim = 384
        elif 'vitl' in self.dinov2_model_name:
            embed_dim = 1024
        elif 'vitg' in self.dinov2_model_name:
            embed_dim = 1536
        else:
            # Fallback, or you can raise an error
            print(f"Warning: Unknown DINOv2 variant {self.dinov2_model_name}. Assuming embed_dim=768 (ViT-B).")
            embed_dim = 768

        self.classifier_head = nn.Linear(embed_dim, num_classes)

        if freeze_backbone:
            for param in self.dinov2.parameters():
                param.requires_grad = False

            for param in self.classifier_head.parameters():
                param.requires_grad = True
        else:
            for param in self.dinov2.parameters():
                param.requires_grad = True
            for param in self.classifier_head.parameters():
                param.requires_grad = True


    def get_parameter_size(self):
        print('On GPU', os.environ["CUDA_VISIBLE_DEVICES"])
        print('NAMING:', NAMING)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")

    def forward(self, x):
        features = self.dinov2(x)
        return self.classifier_head(features)


def train(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0

    bar = tqdm(enumerate(train_loader), total=len(train_loader))

    for i, (inputs, labels) in bar:
        if cutmix is not None and cutmix.num_classes > 0:
            inputs, labels = cutmix(inputs, labels)
            labels = labels.long()

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        bar.set_description(f"Training Loss: {loss.item():.5f}")

    return running_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    bar = tqdm(enumerate(val_loader), total=len(val_loader))

    with torch.no_grad():
        for i, (inputs, labels) in bar:
            inputs, labels = inputs.to(device), labels.to(device).long()

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            bar.set_description(f"Validate Loss: {loss.item():.5f}")

    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def train_model(seed=42, freeze_backbone_initially=True, unfreeze_epoch=None):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)

    model = DINOv2Classifier(num_classes=CLASS_NUM,
                             dinov2_model_name=DINOV2_MODEL_NAME,
                             freeze_backbone=freeze_backbone_initially).to(device)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    if freeze_backbone_initially:
        print("Initially training only the classifier head.")
        optimizer = optim.AdamW(model.classifier_head.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    else:
        print("Initially training the entire model (backbone + classifier head).")
        optimizer = optim.AdamW([
            {'params': model.dinov2.parameters(), 'lr': LEARNING_RATE_BACKBONE},
            {'params': model.classifier_head.parameters(), 'lr': LEARNING_RATE}
        ], weight_decay=1e-5)

    model.get_parameter_size()

    no_improvement_epochs = 0
    train_losses = []
    val_losses = []
    max_acc = 0
    
    current_epochs_list = [] # For plotting

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        if freeze_backbone_initially and unfreeze_epoch is not None and epoch == unfreeze_epoch:
            print(f"Unfreezing DINOv2 backbone at epoch {epoch + 1} and re-initializing optimizer.")
            for param in model.dinov2.parameters():
                param.requires_grad = True

            optimizer = optim.AdamW([
                {'params': model.dinov2.parameters(), 'lr': LEARNING_RATE_BACKBONE},
                {'params': model.classifier_head.parameters(), 'lr': LEARNING_RATE}
            ], weight_decay=1e-5)
            print("Optimizer re-initialized for full model training.")
            model.get_parameter_size()


        train_loss = train(model, train_loader, criterion,
                           optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        current_epochs_list.append(epoch +1)


        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Acc:  {val_acc:.4f}")

        no_improvement_epochs += 1
        if val_acc > max_acc:
            print(f"Saving model, Best Accuracy: {val_acc:.4f}")
            torch.save(model.state_dict(),
                       f'model_{NAMING}_seed{seed}.pth')
            max_acc = val_acc
            no_improvement_epochs = 0

        if no_improvement_epochs >= PATIENCE:
            print("Early stopping")
            break

    print(f"Best Accuracy: {max_acc:.4f}")

    # plot
    """
    if current_epochs_list: # Check if training actually ran for some epochs
        plt.figure(figsize=(10, 6))
        plt.plot(curren
