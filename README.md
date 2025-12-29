기존 LearnOpenCV의 복잡한 내용은 싹 걷어내고, **사용자님의 프로젝트(PCB 결함 탐지 + uv 환경)**에 딱 맞춘 심플하고 세련된 `README.md`입니다.

그대로 복사해서 `README.md` 파일에 붙여넣으시면 됩니다.

---

### 📄 `README.md` (복사해서 사용)

```markdown
# PCB Defect Segmentation using DINO & U-Net

This project explores the application of **Self-Supervised Transformers (DINO)** and **U-Net** for industrial anomaly detection. 
Originally inspired by road segmentation techniques, this project reinterprets the concept to detect microscopic defects on Printed Circuit Boards (PCBs).

## 📌 Project Overview
- **Goal**: Detect defects (Missing hole, Mouse bite, Open circuit, Short, etc.) on PCBs.
- **Approach**: Binary Segmentation (Background vs. Defect).
- **Model**: U-Net with ResNet50 backbone (pretrained on ImageNet).
- **Environment**: Managed by `uv` for fast and reliable dependency management.

## 📂 Project Structure
```bash
├── notebooks
│   ├── 01_data_prep.ipynb    # Converts YOLO txt to Binary Mask & Generates CSV
│   ├── 02_training.ipynb     # Model training with PyTorch
│   └── 03_inference.ipynb    # Visualization of results
├── PCB_Dataset               # (Ignored by Git) Dataset folder
├── models                    # (Ignored by Git) Saved model weights
├── pyproject.toml            # Project dependencies
├── uv.lock                   # Exact version lock file
└── README.md

```

## 🚀 How to Run

This project uses **[uv](https://github.com/astral-sh/uv)** for package management.

### 1. Clone & Setup

```bash
# Clone the repository
git clone <YOUR_REPO_URL>
cd dino-pcb-segmentation

# Sync dependencies (Creating virtual environment)
uv sync

```

### 2. Prepare Dataset

Download the [PCB Defect Dataset (Roboflow)](https://universe.roboflow.com/object-detection-dt-wzpc6/pcb-dataset-defect) and place it in the `PCB_Dataset/` folder.
Then run `notebooks/01_data_prep.ipynb` to convert YOLO labels to segmentation masks.

### 3. Training

Run `notebooks/02_training.ipynb` to train the U-Net model.

```python
# Key Hyperparameters
IMG_SIZE = 480
BATCH_SIZE = 16
EPOCHS = 25

```

## 📊 Dataset Info

* **Source**: [Roboflow Universe - PCB Dataset Defect](https://universe.roboflow.com/object-detection-dt-wzpc6/pcb-dataset-defect)
* **Original Classes**: 6 types (Missing hole, Mouse bite, Open circuit, Short, Spur, Spurious copper)
* **Processed Class**: Binary (0: Background, 1: Defect)

## 🛠 Dependencies

Major libraries used in this project:

* `torch`, `torchvision`
* `segmentation-models-pytorch`
* `opencv-python`
* `albumentations`
* `uv` (Package Manager)

```

---