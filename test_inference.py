import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as album
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = './PCB_Dataset/'
TEST_DATA_PATH = os.path.join(BASE_DIR, 'test.csv')  
MODEL_PATH = "models/base_line_v4_fixed/best_model.pth" 
RESULT_DIR = "./test_results/" 
IMG_SIZE = 512


def get_data_path(path):
    return path.replace("\\", "/")

def get_test_augmentation():
    return album.Compose([
        album.Resize(IMG_SIZE, IMG_SIZE),
        album.Normalize(),
        ToTensorV2()
    ])


if __name__ == '__main__':
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        print(f"Created result directory: {RESULT_DIR}")

    #  모델 로드
    model = smp.Unet(
        encoder_name='resnet50', 
        classes=1, 
        activation=None
    ).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"모델 파일이 없습니다")
        exit()

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    # 2. Test 데이터 로드
    if not os.path.exists(TEST_DATA_PATH):
        print(f"CSV 파일이 없습니다")
        exit()
        
    test_df = pd.read_csv(TEST_DATA_PATH)
    transform = get_test_augmentation()

    print(f"Starting")
    
    count = 0
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_path = get_data_path(row['image'])
        
     
        if not os.path.exists(img_path): 
            img_path = img_path.replace("./PCB_Dataset/", "PCB_Dataset/")
            if not os.path.exists(img_path):
                continue
        
        # 원본 이미지 읽기
        image_src = cv2.imread(img_path)
        if image_src is None: continue

        image_rgb = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
        h, w, _ = image_src.shape

        # 전처리
        image_tensor = transform(image=image_rgb)['image'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype('uint8')
        
        # 예측 마스크를 원본 크기로 복구
        pred_mask_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 원본 위에 Overlay
        overlay = image_src.copy()
        overlay[pred_mask_resized == 1] = [0, 0, 255] 
        combined = cv2.addWeighted(image_src, 0.7, overlay, 0.3, 0)

        # 결과 저장
        file_name = os.path.basename(img_path)
        save_path = os.path.join(RESULT_DIR, f"res_{file_name}")
        cv2.imwrite(save_path, combined)
        count += 1

    print(f"Inference completed. {count} images saved in {RESULT_DIR}")