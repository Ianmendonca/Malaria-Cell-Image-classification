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

''' Using  Dataloader to solve memory problem
    Loading the data into the system and use the same mean and standard 
    deviationnvalue used in training ResNet 50 model.
    Resize the images according to ResNet50 model'''

dataset = ImageFolder(data_dir,transform = transforms.Compose([
    transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean,std)])) 

# Creating validation and training datasets
val_size = int(0.2*len(dataset))
train_size = len(dataset)- val_size
train_data, val_data = random_split(dataset, [train_size, val_size])
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(val_data, batch_size=64,shuffle=True)

# Load pre-trained ResNet-50
resnet50 = models.resnet50(pretrained=True)

# Freeze all layers except the last one
for param in resnet50.parameters():
    param.requires_grad = False

# For binary classification
num_classes = 1  

# Add a linear layer to the end of ResNet  model 
resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)

# Use BCEWithLogitsLoss for binary classification
criterion = nn.BCEWithLogitsLoss()

# These parameters can be changed to optimise the model
optimizer = torch.optim.SGD(resnet50.fc.parameters(), lr=0.001, momentum=0.9)

num_epochs = 10
for epoch in range(num_epochs):
    # Set the model to training mode
    resnet50.train()  
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = resnet50(inputs)
        # BCEWithLogitsLoss expects float labels
        loss = criterion(outputs.view(-1), labels.float())  
        loss.backward()
        optimizer.step()

    # Validation loop
    resnet50.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    best_val_loss = float('inf')
    total = 0
    stop_loss = 2
    no_improv = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            outputs = resnet50(inputs)
            val_loss += criterion(outputs.view(-1), labels.float()).item()
            # Convert logits to binary predictions
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted.view(-1) == labels.float()).sum().item()

        if val_loss < best_val_loss:
            best_val_loss = val_boss
            no_improv = 0
            # save the model
            torch.save(resnet50.state_dict(), 'best_model.pth')
        else:
            no_improv +=1
            if no_improv >= stop_loss:
                print('No improvement in the model')
                break

    print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_loss/len(valloader):.4f}, Accuracy: {(correct/total)*100:.2f}%")

# load the saved model for further evaluation
best_model = models.resnet50(pretrained=False)
best_model.fc = nn.Linear(best_model.fc.in_features, num_classes)
best_model.load_state_dict(torch.load('best_model.pth'))

# Set the model for Evaluation
best_model.eval()
