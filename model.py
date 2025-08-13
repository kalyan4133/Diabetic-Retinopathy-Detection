import os
import random
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import Counter

# Directory paths
data_dir = r'K:\DR\data\images'
model_path = r'K:\DR\data\model_vit.pth'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a custom dataset to use all images without a limit
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Loop through each subdirectory and collect images
        for label, sub_dir in enumerate(sorted(os.listdir(root_dir))):  # Sort to ensure consistent ordering
            sub_dir_path = os.path.join(root_dir, sub_dir)
            if os.path.isdir(sub_dir_path):
                images = os.listdir(sub_dir_path)  # No limit, include all images
                for img in images:
                    self.image_paths.append(os.path.join(sub_dir_path, img))
                    self.labels.append(label)

        # Debugging output to check labels
        print("Class distribution:", Counter(self.labels))  # Print class distribution

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Data augmentation transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the custom dataset
dataset = CustomImageDataset(data_dir, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Load a pre-trained Vision Transformer model and modify the last layer
from torchvision.models import vit_b_16, ViT_B_16_Weights

model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
num_ftrs = model.heads.head.in_features
model.heads.head = nn.Linear(num_ftrs, 5)  # 5 classes for retinopathy severity

model = model.to(device)

# Compute class weights to handle class imbalance
label_counts = Counter(dataset.labels)
total_samples = sum(label_counts.values())
class_weights = [total_samples / label_counts[i] for i in range(5)]
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lowered learning rate

# Train the model with early stopping
def train_model(model, criterion, optimizer, num_epochs=10, patience=3):
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)  # Save the best model
            print(f'Best model saved with loss: {best_loss:.4f}')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    print('Training complete!')

# Call the function to train the model
if __name__ == "__main__":
    train_model(model, criterion, optimizer, num_epochs=10)
