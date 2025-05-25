import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import open_clip
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
import copy
from math import sqrt
import math

# Configuration
SEEDS = [42, 123]
ENCODER_LR = 2e-6
HEAD_LR = 1e-4
BATCH_SIZE = 32
EPOCHS = 40
PATIENCE = 10
NAMING = "BioCLIP_ModelStock"
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
PROMPTS = [
    "A cassava plant with bacterial blight disease, showing wilted leaves and lesions",
    "A cassava plant affected by brown streak disease, with brown streaks on stems",
    "A cassava plant with green mottle, exhibiting mottled green leaves",
    "A cassava plant suffering from mosaic disease, with yellowing mosaic patterns",
    "A healthy cassava plant with vibrant green leaves"
]
EPS = 1e-8

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_transforms():
    """Define train and validation transforms."""
    train_transforms = v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(15),
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
    
    return train_transforms, val_transforms

class CassavaDataset(Dataset):
    """Dataset for cassava disease classification."""
    def __init__(self, df, img_dir, transform=None, prompts=PROMPTS):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
        self.prompts = prompts

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = os.path.join(self.img_dir, row['image_id'])
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        label = int(row['label'])

        if self.transform:
            image = self.transform(image)

        text = self.prompts[label]
        text_tokens = self.tokenizer([text])[0]

        return image, text_tokens, label

class CLIPLoss(nn.Module):
    """CLIP contrastive loss."""
    def __init__(self, temperature=0.07):
        super(CLIPLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features):
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        if torch.any(torch.isnan(image_features)) or torch.any(torch.isnan(text_features)):
            print("NaN detected in features")
            return torch.tensor(float('nan'), device=DEVICE)
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        if torch.any(torch.isnan(logits)):
            print("NaN detected in logits")
            return torch.tensor(float('nan'), device=DEVICE)
        labels = torch.arange(logits.size(0)).to(DEVICE)
        loss_i2t = self.criterion(logits, labels)
        loss_t2i = self.criterion(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2

class CLIPClassifier(nn.Module):
    """CLIP with a classification head."""
    def __init__(self, clip_model, num_classes=5):
        super(CLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(clip_model.visual.output_dim, num_classes)

    def forward(self, images):
        image_features = self.clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        logits = self.classifier(image_features)
        return image_features, logits

def load_data():
    """Load and split dataset."""
    df = pd.read_csv('data/train.csv')
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
    
    train_transforms, val_transforms = get_transforms()
    train_dataset = CassavaDataset(train_df, img_dir='data/train_images', transform=train_transforms)
    val_dataset = CassavaDataset(val_df, img_dir='data/train_images', transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)
    
    return train_loader, val_loader

def compute_angle(state_dict_1, state_dict_2, ref_state_dict, add_ignore_keys=[]):
    """Compute angle between two state dicts relative to a reference."""
    ignore_keys = [
        'positional_embedding', 'text_projection', 'logit_scale',
        'token_embedding.weight', 'ln_final.weight', 'ln_final.bias'
    ]
    ignore_keys += ['clip_model.' + key for key in ignore_keys]
    ignore_keys.extend(add_ignore_keys)

    return_dict = {}
    with torch.no_grad():
        for key in ref_state_dict:
            if key in ignore_keys:
                continue
            if key not in state_dict_1 or key not in state_dict_2:
                continue
            state_dict_1_val = state_dict_1[key]
            state_dict_2_val = state_dict_2[key]
            ref_val = ref_state_dict[key]

            devices = {state_dict_1_val.device, state_dict_2_val.device, ref_val.device}
            if len(devices) > 1:
                raise RuntimeError(f"Device mismatch for key {key}: found devices {devices}")

            if not (state_dict_1_val.shape == state_dict_2_val.shape == ref_val.shape):
                continue

            vector1 = (state_dict_1_val - ref_val).clone().detach().float()
            vector2 = (state_dict_2_val - ref_val).clone().detach().float()

            if torch.any(torch.isnan(vector1)) or torch.any(torch.isnan(vector2)):
                print(f"NaN detected in vectors for key {key}")
                continue

            cosine_val = torch.sum(vector1 * vector2) / (sqrt(torch.sum(vector1 ** 2) * torch.sum(vector2 ** 2)) + EPS)
            cosine_val = torch.clamp(cosine_val, min=-1., max=1.)
            angle = np.rad2deg(torch.acos(cosine_val).detach().cpu())
            if not torch.isnan(angle):
                return_dict[key] = angle

    return return_dict

def compute_ratio(angle_dict, k=2):
    """Compute interpolation ratio based on angles."""
    ratio_dict = {}
    for key, angle in angle_dict.items():
        angle_rad = np.deg2rad(angle)
        ratio = k * np.cos(angle_rad) / ((k - 1) * np.cos(angle_rad) + 1 + EPS)
        if not np.isnan(ratio):
            ratio_dict[key] = ratio
    return ratio_dict

def merge_weights(w1, w2, w0, ratio):
    """Merge two fine-tuned weights with reference to pre-trained weights using ratios."""
    w12 = {}
    for key in w1.keys():
        if key not in w2 or key not in w0:
            w12[key] = w1[key].clone()
        else:
            w12[key] = (w1[key].clone() + w2[key].clone()) / 2.

    w_merge = copy.deepcopy(w12)
    for key, r in ratio.items():
        if key in w_merge and key in w0:
            w_merge[key] = w12[key].clone() * r + w0[key].clone() * (1. - r)
            if torch.any(torch.isnan(w_merge[key])):
                print(f"NaN detected in merged weights for key {key}")
                w_merge[key] = w12[key].clone()
    return w_merge

def train_epoch(seed, model, train_loader, val_loader, contrastive_criterion, classification_criterion, optimizer, scaler, epoch, tokenized_prompts):
    """Train a single model for one epoch and return its state."""
    set_seed(seed)
    
    model.train()
    train_loss = 0.0
    
    for images, texts, labels in tqdm(train_loader, desc=f"Seed {seed} | Epoch {epoch+1}/{EPOCHS}"):
        images, texts, labels = images.to(DEVICE), texts.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            image_features, logits = model(images)
            text_features = model.clip_model.encode_text(texts.squeeze(1))
            contrastive_loss = contrastive_criterion(image_features, text_features)
            classification_loss = classification_criterion(logits, labels)
            if torch.isnan(contrastive_loss) or torch.isnan(classification_loss):
                print(f"NaN loss detected: contrastive={contrastive_loss.item()}, classification={classification_loss.item()}")
                continue
            loss = 0.5 * contrastive_loss + 0.5 * classification_loss
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    val_loss, val_accuracy = evaluate_model(model, val_loader, contrastive_criterion, classification_criterion, tokenized_prompts)
    
    return train_loss, val_loss, val_accuracy, model.state_dict()

def evaluate_model(model, val_loader, contrastive_criterion, classification_criterion, tokenized_prompts):
    """Evaluate model on validation set."""
    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0.0
    
    with torch.no_grad():
        for images, texts, labels in val_loader:
            images, texts, labels = images.to(DEVICE), texts.to(DEVICE), labels.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                image_features, logits = model(images)
                text_features = model.clip_model.encode_text(texts.squeeze(1))
                contrastive_loss = contrastive_criterion(image_features, text_features)
                classification_loss = classification_criterion(logits, labels)
                if torch.isnan(contrastive_loss) or torch.isnan(classification_loss):
                    print(f"Validation NaN loss: contrastive={contrastive_loss.item()}, classification={classification_loss.item()}")
                loss = 0.5 * contrastive_loss + 0.5 * classification_loss
                
                val_loss += loss.item()
                text_features = F.normalize(model.clip_model.encode_text(tokenized_prompts), dim=-1)
                image_features = F.normalize(image_features, dim=-1)
                logits = torch.matmul(image_features, text_features.t())
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_accuracy = accuracy_score(val_labels, val_preds)
    
    return val_loss, val_accuracy

def train_model(train_loader, val_loader, pretrained_state_dict):
    """Train two models with fixed seeds, merging weights with Model Stock after each epoch."""
    models = {}
    writers = {}
    optimizers = {}
    scalers = {}
    
    # Initialize pretrained_state_dict as mutable
    current_pretrained_state_dict = copy.deepcopy(pretrained_state_dict)
    
    for seed in SEEDS:
        set_seed(seed)
        clip_model, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        clip_model = clip_model.to(DEVICE)
        models[seed] = CLIPClassifier(clip_model).to(DEVICE)
        models[seed].load_state_dict(current_pretrained_state_dict, strict=False)
        
        optimizers[seed] = optim.AdamW([
            {'params': models[seed].clip_model.parameters(), 'lr': ENCODER_LR},
            {'params': models[seed].classifier.parameters(), 'lr': HEAD_LR}
        ])
        
        scalers[seed] = torch.amp.GradScaler()
        writers[seed] = SummaryWriter(f'runs/seed_{seed}')
    
    periodic_merge_writer = SummaryWriter('runs/Periodic_Merge')
    
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
    tokenized_prompts = tokenizer(PROMPTS).to(DEVICE)
    
    contrastive_criterion = CLIPLoss(temperature=0.07)
    classification_criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizers[SEEDS[0]], T_max=20)
    
    best_accuracy = 0.0
    best_model_state = None
    early_stop_counter = 0
    
    for epoch in range(EPOCHS):
        model_states = []
        
        for seed in SEEDS:
            train_loss, val_loss, val_accuracy, model_state = train_epoch(
                seed, models[seed], train_loader, val_loader,
                contrastive_criterion, classification_criterion,
                optimizers[seed], scalers[seed], epoch, tokenized_prompts
            )
            
            writers[seed].add_scalar('Loss/Train', train_loss, epoch)
            writers[seed].add_scalar('Loss/Val', val_loss, epoch)
            writers[seed].add_scalar('Accuracy/Val', val_accuracy, epoch)
            
            print(f"Seed {seed} | Epoch {epoch+1}/{EPOCHS}:")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            model_states.append(model_state)
        
        # Model Stock weight merging
        angle = compute_angle(model_states[0], model_states[1], current_pretrained_state_dict)
        ratio = compute_ratio(angle)
        avg_state_dict = merge_weights(model_states[0], model_states[1], current_pretrained_state_dict, ratio)
        
        # Update pretrained_state_dict with merged weights
        current_pretrained_state_dict = copy.deepcopy(avg_state_dict)
        periodic_merge_writer.add_scalar('ModelStock/Reference_Update', 1, epoch)
        
        clip_model, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        ensemble_model = CLIPClassifier(clip_model).to(DEVICE)
        ensemble_model.load_state_dict(avg_state_dict)
        
        ensemble_loss, ensemble_accuracy = evaluate_model(
            ensemble_model, val_loader, contrastive_criterion, classification_criterion, tokenized_prompts
        )
        
        print(f"Logging periodic merge metrics to TensorBoard for epoch {epoch+1}")
        periodic_merge_writer.add_scalar('Loss/Val', ensemble_loss, epoch)
        periodic_merge_writer.add_scalar('Accuracy/Val', ensemble_accuracy, epoch)
        
        print(f"Periodic Merge | Epoch {epoch+1}/{EPOCHS}:")
        print(f"Periodic Merge Loss: {ensemble_loss:.4f}, Periodic Merge Accuracy: {ensemble_accuracy:.4f}")
        
        for seed in SEEDS:
            models[seed].load_state_dict(avg_state_dict)
        
        if ensemble_accuracy > best_accuracy:
            best_accuracy = ensemble_accuracy
            best_model_state = copy.deepcopy(avg_state_dict)
            early_stop_counter = 0
            print(f"Periodic Merge | Best Accuracy update: {ensemble_accuracy:.4f}")
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= PATIENCE:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
        
        scheduler.step()
    
    torch.save(best_model_state, f'model_{NAMING}_ensemble.pth')
    print(f"Final ensemble model saved with validation accuracy: {best_accuracy:.4f}")
    
    for seed in SEEDS:
        writers[seed].close()
    periodic_merge_writer.close()
    
    return best_model_state

def evaluate_ensemble(val_loader, state_dict):
    """Evaluate the final ensembled model."""
    clip_model, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    model = CLIPClassifier(clip_model).to(DEVICE)
    model.load_state_dict(state_dict)
    
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
    tokenized_prompts = tokenizer(PROMPTS).to(DEVICE)
    
    model.eval()
    val_preds, val_labels = [], []
    
    with torch.no_grad():
        for images, _, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                image_features, _ = model(images)
                text_features = F.normalize(model.clip_model.encode_text(tokenized_prompts), dim=-1)
                image_features = F.normalize(image_features, dim=-1)
                logits = torch.matmul(image_features, text_features.t())
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
    
    ensemble_accuracy = accuracy_score(val_labels, val_preds)
    print(f"Final Ensemble Validation Accuracy: {ensemble_accuracy:.4f}")

def main():
    """Main function to orchestrate training and ensembling."""
    print(f"Starting {NAMING} training...")
    
    # Load pretrained BioCLIP weights and move to DEVICE
    clip_model, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    clip_model = clip_model.to(DEVICE)
    pretrained_model = CLIPClassifier(clip_model).to(DEVICE)
    pretrained_state_dict = pretrained_model.state_dict()
    
    train_loader, val_loader = load_data()
    
    best_model_state = train_model(train_loader, val_loader, pretrained_state_dict)
    
    print("\nEvaluating final ensemble model...")
    evaluate_ensemble(val_loader, best_model_state)

if __name__ == "__main__":
    main()