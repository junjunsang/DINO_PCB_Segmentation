import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

NUM_WORKERS = 8 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = './PCB_Dataset/'
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'train.csv')
VAL_DATA_PATH = os.path.join(BASE_DIR, 'valid.csv')
MODEL_SAVE_DIR = "models/base_line_v4_fixed/"

EPOCHS = 30
LR = 0.001
IMG_SIZE = 512
BATCH_SIZE = 16 
ENCODER = 'resnet50'
WEIGHTS = 'imagenet'


def get_data_path(csv_path):
    path = csv_path.replace("/home/somusan/OpencvUni/opencvblog/robotics-series/yolop_idd/", "")
    full_path = os.path.join(BASE_DIR, path)
    if os.path.exists(full_path):
        return full_path
    return csv_path

class RoadsDataset(torch.utils.data.Dataset):
    def __init__(self, df, augmentation=None):
        self.image_paths = [get_data_path(i) for i in df["image"].tolist()]
        self.mask_paths = [get_data_path(i) for i in df["mask"].tolist()]
        self.augmentation = augmentation
    
    def __getitem__(self, i):
        image = cv2.imread(self.image_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)
        
        mask = (mask > 0).astype('float32') 
        mask = np.expand_dims(mask, axis=-1)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
      
        if mask.shape[-1] == 1:
            mask = mask.permute(2, 0, 1) 

        return image, mask
        
    def __len__(self):
        return len(self.image_paths)

def get_transforms(img_size):
    train_transform = album.Compose([
        album.Resize(img_size, img_size),
        album.HorizontalFlip(p=0.5),
        album.Normalize(),
        ToTensorV2()
    ])
    val_transform = album.Compose([
        album.Resize(img_size, img_size),
        album.Normalize(),
        ToTensorV2()
    ])
    return train_transform, val_transform

if __name__ == '__main__':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)

    t_trans, v_trans = get_transforms(IMG_SIZE)
    train_ds = RoadsDataset(train_df, t_trans)
    val_ds = RoadsDataset(val_df, v_trans)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    model = smp.Unet(encoder_name=ENCODER, encoder_weights=WEIGHTS, classes=1, activation=None).to(device)

    dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
    bce_loss = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR)
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        # Training
        model.train()
        t_loss, t_iou = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for img, msk in pbar:
            img, msk = img.to(device), msk.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = dice_loss(out, msk) + bce_loss(out, msk)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item()
            tp, fp, fn, tn = smp.metrics.get_stats(out, msk.long(), mode='binary', threshold=0.5)
            t_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        v_loss, v_iou = 0, 0
        with torch.no_grad():
            for img, msk in tqdm(val_loader, desc="[Valid]", leave=False):
                img, msk = img.to(device), msk.to(device)
                out = model(img)
                loss = dice_loss(out, msk) + bce_loss(out, msk)
                v_loss += loss.item()
                tp, fp, fn, tn = smp.metrics.get_stats(out, msk.long(), mode='binary', threshold=0.5)
                v_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()

        t_loss, v_loss = t_loss/len(train_loader), v_loss/len(val_loader)
        t_iou, v_iou = t_iou/len(train_loader), v_iou/len(val_loader)
        
        scheduler.step(v_loss)
        print(f"Loss: {t_loss:.4f}/{v_loss:.4f} | IoU: {t_iou:.4f}/{v_iou:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_model.pth"))
            print(">> Best Model Saved")

    print("Finished.")