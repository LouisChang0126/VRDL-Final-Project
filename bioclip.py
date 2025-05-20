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
from sklearn.model_selection import train_test_split

seed = 42
BATCH_SIZE = 32
EPOCH = 40
PATIENCE = 10
LEARNING_RATE = 1e-5
NAMING = "BioCLIP"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(seed)

train_transforms = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
    v2.ToTensor(),
    v2.ConvertImageDtype(torch.float),
    v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

val_transforms = v2.Compose([
    v2.Resize(size=224, interpolation=v2.InterpolationMode.BICUBIC, max_size=None, antialias=True),
    v2.CenterCrop(size=(224, 224)),
    v2.ToTensor(),
    v2.ConvertImageDtype(torch.float),
    v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])


prompts = [
    "This is a plant suffering from Cassava Bacterial Blight",
    "This is a plant suffering from Cassava Brown Streak Disease",
    "This is a plant suffering from Cassava Green Mottle",
    "This is a plant suffering from Cassava Mosaic Disease",
    "This is a healthy plant"
]


model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
model = model.to(device)
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')


class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.prompts = prompts

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = os.path.join(self.img_dir, row['image_id'])
        image = Image.open(img_name).convert('RGB')
        label = int(row['label'])

        if self.transform:
            image = self.transform(image)

        text = self.prompts[label]
        text_tokens = self.tokenizer([text])

        return image, text_tokens, label

df = pd.read_csv('data/train.csv')
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)

train_dataset = CassavaDataset(train_df, img_dir='data/train_images', transform=preprocess_train)
val_dataset = CassavaDataset(val_df, img_dir='data/train_images', transform=preprocess_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

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

criterion = CLIPLoss(temperature=0.07)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
scaler = torch.amp.GradScaler()

best_accuracy = 0.0
best_model_state = None
early_stop_counter = 0
tokenized_prompts = tokenizer(prompts).to(device)

print(f'start running {NAMING}')
for epoch in range(EPOCH):
    model.train()
    train_loss = 0.0

    for images, texts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCH}"):
        images, texts, labels = images.to(device), texts.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts.squeeze(1))
            loss = criterion(image_features, text_features)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0.0

    with torch.no_grad():
        for images, texts, labels in val_loader:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts.squeeze(1))
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
    print(f"Train Loss: {train_loss:.4f}")
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
torch.save(model.state_dict(), f'model_{NAMING}_seed{seed}.pth')
print(f"Best model saved with validation accuracy: {best_accuracy:.4f}")