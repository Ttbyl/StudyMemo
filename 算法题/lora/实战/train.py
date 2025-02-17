'''
Author: Wangzhibo && ttbylzb11@gmail.com
Date: 2025-02-14 15:19:51
LastEditors: Wanzhiboo && ttbylzb11@gmail.com
LastEditTime: 2025-02-14 16:11:04
FilePath: /实战/train.py
Description: 

Copyright (c) 2025 by ttbylzb11@gmail.com, All Rights Reserved. 
'''
from datasets import CIFAR100
from resnet import ResNet18
from lora import lora
import torch
import torch.nn as nn

# dataloader:
train_dataset = CIFAR100(root='./data', train=True)
test_dataset = CIFAR100(root='data', train=False)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False)

# model:
model = ResNet18()
print("model:ResNet18")
# 冻结参数:
# for param in model.parameters():
#     param.requires_grad = False
# 替换最后一层:
model.fc = nn.Linear(512, 100)
for param in model.fc.parameters():
    param.requires_grad = True
# lora:
lora = lora(224*224*3, 100)
if torch.cuda.is_available():
    model = model.cuda()
    lora = lora.cuda()


# loss and optimizer:
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training:
for epoch in range(10):
    model.train()  # Set model to train mode
    total_correct = 0
    total_samples = 0
    for i, (images, labels) in enumerate(train_loader):
        correct = 0
        samples = 0
        # move tensors to GPU if available:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        # forward pass:
        outputs = model(images) + lora(images)
        loss = criterion(outputs, labels)
        # backward and optimize:
        loss.backward()
        optimizer.step()

        # Accuracy calculation
        _, predicted = torch.max(outputs, 1)
        samples += labels.size(0)
        correct += (predicted == labels).sum().item()

        total_correct += correct
        total_samples += samples

        # Print loss and accuracy
        if (i + 1) % 100 == 0:  # Print every 100 steps
            accuracy = 100 * correct / samples
            print(
                f"Epoch [{epoch+1}/10], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
    # Print epoch accuracy
    epoch_accuracy = 100 * total_correct / total_samples
    print(f"Epoch [{epoch+1}/10], Accuracy: {epoch_accuracy:.2f}%")


# save model:
torch.save(model.state_dict(), 'model.pt')
torch.save(lora.state_dict(), 'lora.pt')

# Evaluation:
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for i, (images, labels) in test_loader:
        outputs = model(images) + lora(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
