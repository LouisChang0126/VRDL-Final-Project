import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from torchvision.transforms import v2

BATCH_SIZE = 32
EPOCH = 20
PATIENCE = 5
LEARNING_RATE = 1e-5
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.75, 1.3333)),
        v2.RandomRotation(degrees=20),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        v2.ToTensor(),
        v2.ConvertImageDtype(torch.float),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ]),
    'val': v2.Compose([
        v2.CenterCrop(size=(224, 224)),
        v2.ToTensor(),
        v2.ConvertImageDtype(torch.float),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        
    ])
}

# Prompts for the 5 classes
prompts = [
    "This is a plant suffering from Cassava Bacterial Blight",
    "This is a plant suffering from Cassava Brown Streak Disease",
    "This is a plant suffering from Cassava Green Mottle",
    "This is a plant suffering from Cassava Mosaic Disease",
    "This is a healthy plant"

    # 爛2% 0.8687
    # "A cassava leaf suffering from Cassava Bacterial Blight",
    # "A cassava leaf suffering from Cassava Brown Streak Disease",
    # "A cassava leaf suffering from Cassava Green Mottle",
    # "A cassava leaf suffering from Cassava Mosaic Disease",
    # "A healthy cassava leaf"

    # 爛1%
    # "This is a plant with Cassava Bacterial Blight",
    # "This is a plant with Cassava Brown Streak Disease",
    # "This is a plant with Cassava Green Mottle",
    # "This is a plant with Cassava Mosaic Disease",
    # "This is a healthy plant"
]

# Custom Dataset
class CassavaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, tokenizer=None, prompts=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.prompts = prompts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        # Get the corresponding prompt (do not move to device here)
        text = self.prompts[label]
        text_tokens = self.tokenizer([text])  # Keep on CPU

        return image, text_tokens, label

# Load model and preprocessors
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
model = model.to(device)
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')

# Data loading
train_dataset = CassavaDataset(
    csv_file='data/train.csv',
    img_dir='data/train_images',
    transform=preprocess_train,
    tokenizer=tokenizer,
    prompts=prompts
)

# Split dataset (90-10 split)
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

# Loss function (CLIP-like contrastive loss)
class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(CLIPLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features):
        image_features = nn.functional.normalize(image_features, dim=-1)
        text_features = nn.functional.normalize(text_features, dim=-1)
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(device)
        loss_i2t = self.criterion(logits, labels)
        loss_t2i = self.criterion(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2

# Training setup
criterion = CLIPLoss(temperature=0.07)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# Training loop
best_accuracy = 0.0
best_model_state = None
early_stop_counter = 0

# Pre-tokenize prompts for validation
tokenized_prompts = tokenizer(prompts).to(device)

for epoch in range(EPOCH):
    model.train()
    train_loss = 0.0
    train_preds, train_labels = [], []

    for images, texts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCH}"):
        images, texts, labels = images.to(device), texts.to(device), labels.to(device)

        optimizer.zero_grad()
        image_features = model.encode_image(images)  # Use encode_image instead of visual
        text_features = model.encode_text(texts.squeeze(1))  # Use encode_text instead of text
        loss = criterion(image_features, text_features)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        with torch.no_grad():
            image_features = nn.functional.normalize(image_features, dim=-1)
            text_features = nn.functional.normalize(model.encode_text(tokenized_prompts), dim=-1)
            logits = torch.matmul(image_features, text_features.t())
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

    train_accuracy = accuracy_score(train_labels, train_preds)
    train_loss /= len(train_loader)

    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0.0

    with torch.no_grad():
        for images, texts, labels in val_loader:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            image_features = model.encode_image(images)  # Use encode_image
            text_features = model.encode_text(texts.squeeze(1))  # Use encode_text
            loss = criterion(image_features, text_features)
            val_loss += loss.item()

            image_features = nn.functional.normalize(image_features, dim=-1)
            text_features = nn.functional.normalize(model.encode_text(tokenized_prompts), dim=-1)
            logits = torch.matmul(image_features, text_features.t())
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(val_labels, val_preds)
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCH}:")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_model_state = model.state_dict()
        early_stop_counter = 0
        print(f"Best Accuracy update: {val_accuracy:.4f}")
    else:
        early_stop_counter += 1

    if early_stop_counter >= PATIENCE:
        print("Early stopping triggered")
        break

    scheduler.step()

model.load_state_dict(best_model_state)
torch.save(model.state_dict(), 'best_cassava_bioclip_clip.pth')
print(f"Best model saved with validation accuracy: {best_accuracy:.4f}")