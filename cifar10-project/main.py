"""
CIFAR-10 CNN Classification - Quick Start Version
Simplified for easy testing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

print("="*60)
print("CIFAR-10 Image Classification Project")
print("="*60)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Data transforms
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load data
print("\nDownloading CIFAR-10 dataset...")
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ===== BASELINE MODEL =====
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ===== IMPROVED MODEL =====
class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Block 1
        self.conv1a = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(64)
        
        # Block 2
        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(128)
        
        # Block 3
        self.conv3a = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3b = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout_fc = nn.Dropout(0.5)
        
    def forward(self, x):
        # Block 1
        x = self.relu(self.bn1a(self.conv1a(x)))
        x = self.relu(self.bn1b(self.conv1b(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.relu(self.bn2a(self.conv2a(x)))
        x = self.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)
        x = self.dropout2(x)
        
        # Block 3
        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.relu(self.bn3b(self.conv3b(x)))
        x = self.pool(x)
        x = self.dropout3(x)
        
        # Classifier
        x = x.view(-1, 256 * 4 * 4)
        x = self.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

# ===== TRAINING FUNCTION =====
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss/len(loader), 100.*correct/total

# ===== EVALUATION FUNCTION =====
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss/len(loader), 100.*correct/total

# ===== TRAIN MODEL =====
def train_model(model, name, epochs=30):
    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    train_accs, test_accs = [], []
    best_acc = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'{name}_best.pth')
            print(f"✓ New best: {best_acc:.2f}%")
    
    return train_accs, test_accs, best_acc

# ===== MAIN EXECUTION =====
print("\n" + "="*60)
print("Starting Training Process")
print("="*60)

# Train baseline
baseline = BaselineCNN()
print(f"\nBaseline parameters: {sum(p.numel() for p in baseline.parameters()):,}")
baseline_train, baseline_test, baseline_best = train_model(baseline, "Baseline", epochs=30)

# Train improved
improved = ImprovedCNN()
print(f"\nImproved parameters: {sum(p.numel() for p in improved.parameters()):,}")
improved_train, improved_test, improved_best = train_model(improved, "Improved", epochs=30)

# Plot results
print("\n" + "="*60)
print("Generating plots...")
print("="*60)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(baseline_train, 'o-', label='Baseline Train', alpha=0.7)
plt.plot(baseline_test, 's-', label='Baseline Test', alpha=0.7)
plt.plot(improved_train, 'o-', label='Improved Train', alpha=0.7)
plt.plot(improved_test, 's-', label='Improved Test', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Progress')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
models = ['Baseline', 'Improved']
accuracies = [baseline_best, improved_best]
colors = ['skyblue', 'lightcoral']
plt.bar(models, accuracies, color=colors)
plt.ylabel('Best Test Accuracy (%)')
plt.title('Model Comparison')
plt.ylim(0, 100)
for i, v in enumerate(accuracies):
    plt.text(i, v+2, f'{v:.2f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results.png', dpi=300)
print("✓ Saved results.png")

# Final summary
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Baseline CNN: {baseline_best:.2f}%")
print(f"Improved CNN: {improved_best:.2f}%")
print(f"Improvement:  +{improved_best-baseline_best:.2f}%")
print("="*60)
print("\nTraining complete! Check results.png for visualization.")
print("Models saved as Baseline_best.pth and Improved_best.pth")
print("="*60)