from pathlib import Path
from typing import Callable, Sequence, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet50
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pydicom
import numpy as np
import os
from catboost import CatBoostClassifier

# --- 1. Настройка устройства ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 2. Класс Dataset для DICOM изображений ---
class DICOMDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], labels: Sequence[int], transform: Optional[Callable] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        dicom_file = pydicom.dcmread(str(self.image_paths[idx]))
        img = dicom_file.pixel_array
        img = cv2.resize(img, (224, 224))  # Преобразуем размер изображения

        # Если изображение ч/б, добавим фиктивный канал
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        
        # Преобразуем к формату albumentations
        img = np.repeat(img, 3, axis=-1).astype(np.uint8)  # Преобразуем в 3 канала
        
        if self.transform:
            img = self.transform(image=img)["image"]

        label = self.labels[idx]
        return img, label


# --- 3. Аугментации ---
transform = A.Compose([
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])


# --- 4. Модель ---
class CustomResNet(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# --- 5. Тренировка ---
def train_model(model, train_loader, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(imgs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    print("Training completed.")


# --- 6. Основной код ---
if __name__ == "__main__":
    # Пути к данным
    data_dir = Path(r"D:\ds_clav_fracture_train_good\block_0000_anon")
    image_paths = list(data_dir.rglob("*.dcm"))  # Рекурсивный поиск DICOM файлов
    print(f"Found {len(image_paths)} DICOM files.")  # Проверяем количество файлов

    # Проверка на пустой список
    if len(image_paths) == 0:
        raise ValueError("No DICOM files found. Check the dataset path and structure.")

    # Пример создания меток
    # Заменить логику на свою, если метки определяются иначе.
    labels = [0 if "benign" in str(p).lower() else 1 for p in image_paths]
    
    # Создаем датасет и загрузчик данных
    dataset = DICOMDataset(image_paths, labels, transform)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Создаем и тренируем модель
    model = CustomResNet(num_classes=1)
    train_model(model, train_loader, epochs=10, lr=1e-4)
