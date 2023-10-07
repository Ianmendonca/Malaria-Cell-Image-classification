import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torchvision.models as models

# mean and standard deviation value for ResNet50
mean = [0.4751, 0.4270, 0.3992]
std = [0.3097, 0.3083, 0.3183] 

data_dir = "E:\\Masters\\Semester 4\\Malaria Detection\\archive\\cell_images"

''' Using   Dataloader to solve memory problem.'''

dataset = ImageFolder(data_dir,transform = transforms.Compose([
    transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean,std)])) # Loading the data into the system

batch_size = 128
val_size = int(0.2*len(dataset))
train_size = len(dataset)- val_size
train_data, val_data = random_split(dataset, [train_size, val_size])
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(val_data, batch_size=64,shuffle=True)

import torch
import torch.nn as nn
from torchvision import models

# Load pre-trained ResNet-50
resnet50 = models.resnet50(pretrained=True)

# Freeze all layers except the last one
for param in resnet50.parameters():
    param.requires_grad = False

num_classes = 1  # For binary classification
resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)

criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
optimizer = torch.optim.SGD(resnet50.fc.parameters(), lr=0.001, momentum=0.9)

num_epochs = 1
for epoch in range(num_epochs):
    resnet50.train()  # Set the model to training mode
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = resnet50(inputs)
        loss = criterion(outputs.view(-1), labels.float())  # BCEWithLogitsLoss expects float labels
        loss.backward()
        optimizer.step()

    # Validation loop
    resnet50.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct_1 = 0
    correct_2 = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            outputs = resnet50(inputs)
            val_loss += criterion(outputs.view(-1), labels.float()).item()  # BCEWithLogitsLoss expects float labels
            predicted = (torch.sigmoid(outputs) > 0.5).float() # Convert logits to binary predictions
            total += labels.size(0)
            print(total)
            correct_1 += (predicted.view(-1) == labels.float()).sum().item()
            print(correct_1)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_loss/len(valloader):.4f}, Accuracy: {(correct_1/total)*100:.2f}%")
