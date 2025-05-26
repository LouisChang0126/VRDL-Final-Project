# dinov2_pipeline_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import os
import pandas as pd
import numpy as np
import PIL
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import pipeline  # [MODIFIED]

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# Parameters
BATCH_SIZE = 4 #64
EPOCHS = 35
PATIENCE = 10
LEARNING_RATE = 1e-4
CLASS_NUM = 5
NAMING = "DINOv2_PIPELINE"  # [MODIFIED]

# Set Random Seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

train_transforms = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
    v2.RandomRotation(degrees=20),
    v2.RandomHorizontalFlip(),
    v2.ToTensor(),
    v2.ConvertImageDtype(torch.float)
])

test_transforms = v2.Compose([
    v2.Resize(size=224, interpolation=v2.InterpolationMode.BICUBIC, max_size=None, antialias=True),
    v2.CenterCrop(size=(224, 224)),
    v2.ToTensor(),
    v2.ConvertImageDtype(torch.float)
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

# Read labels
df = pd.read_csv('data/train.csv')
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
train_dataset = myDataset(train_df, img_dir='data/train_images', transform=train_transforms)
val_dataset = myDataset(val_df, img_dir='data/train_images', transform=test_transforms)

# [MODIFIED] Replace model with Hugging Face pipeline using DINOv2
classifier = pipeline(
    task="image-classification",
    model="facebook/dinov2-base",
    device=0,
    top_k=CLASS_NUM
)

def validate_with_pipeline(pipeline_fn, val_loader):
    correct = 0
    total = 0
    bar = tqdm(enumerate(val_loader), total=len(val_loader))

    for _, (inputs, labels) in bar:
        batch_preds = []
        for img_tensor in inputs:
            img = v2.ToPILImage()(img_tensor.cpu())
            outputs = pipeline_fn(img)
            pred_label = int(outputs[0]['label'].split('_')[-1]) if 'label' in outputs[0] else 0
            batch_preds.append(pred_label)

        total += len(labels)
        correct += sum([pred == label.item() for pred, label in zip(batch_preds, labels)])
        bar.set_description(f"Pipeline Validation Acc: {correct / total:.5f}")

    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    set_seed(42)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    val_acc = validate_with_pipeline(classifier, val_loader)
    print(f"Pipeline Validation Accuracy: {val_acc:.4f}")
