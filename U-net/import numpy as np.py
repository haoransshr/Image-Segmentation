import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

import torch
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

dataset = datasets.ImageFolder("Data/without_augmentation", transform=transform)
# dataset = datasets.ImageFolder("Data/with_augmentation", transform=transform)

indices = list(range(len(dataset)))

split = int(np.floor(0.80 * len(dataset)))  # train_size
validation = int(np.floor(0.10 * len(dataset)))  # validation

print(0, validation, split, len(dataset))

print(f"length of train size :{split}")
print(f"length of validation size :{validation}")
print(f"length of test size :{len(dataset) - split - validation}")

np.random.shuffle(indices)

train_indices, validation_indices, test_indices = (
    indices[:split],  # 训练集索引
    indices[split : split + validation],  # 验证集索引
    indices[split + validation :],  # 测试集索引
)

train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(validation_indices)
test_sampler = SubsetRandomSampler(test_indices)

targets_size = len(dataset.class_to_idx)

# Load pretrained VGG16 model
vgg_model = models.vgg16(pretrained=True)
for params in vgg_model.parameters():
    params.requires_grad = False

n_features = vgg_model.classifier[0].in_features
vgg_model.classifier = nn.Sequential(
    nn.Linear(n_features, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, targets_size),
)

class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # conv2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            # conv3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            # conv4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(16 * 16 * 256, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

cnn_model = CNN(targets_size)
cnn_model.to(device)

from torchsummary import summary

summary(cnn_model, (3, 256, 256))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters())

def batch_gd(model, criterion, train_loader, validation_loader, epochs):
    train_losses = np.zeros(epochs)
    validation_losses = np.zeros(epochs)

    for e in range(epochs):
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_loss)

        validation_loss = []
        for inputs, targets in validation_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = criterion(output, targets)
            validation_loss.append(loss.item())

        validation_loss = np.mean(validation_loss)

        train_losses[e] = train_loss
        validation_losses[e] = validation_loss

        dt = datetime.now() - t0

        print(
            f"Epoch : {e+1}/{epochs} Train_loss:{train_loss:.3f} Validation_loss:{validation_loss:.3f} Duration:{dt}"
        )

    return train_losses, validation_losses

batch_size = 64
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler
)
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=test_sampler
)
validation_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=validation_sampler
)

train_losses, validation_losses = batch_gd(
    cnn_model, criterion, train_loader, validation_loader, 10
)

torch.save(cnn_model.state_dict(), "cnn_without.pt")
# torch.save(cnn_model.state_dict(), 'cnn_with.pt')

targets_size = 39
cnn_model.load_state_dict(torch.load("cnn_without.pt"))
# cnn_model.load_state_dict(torch.load("cnn_with.pt"))
cnn_model.eval()

%matplotlib inline

plt.plot(train_losses, label="train_loss")
plt.plot(validation_losses, label="validation_loss")
plt.xlabel("No of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

def calculate_metrics(model, loader):
    all_targets = []
    all_predictions = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    return accuracy, precision, recall, f1, all_targets, all_predictions

def plot_roc_curve(all_targets, all_predictions, n_classes):
    all_targets_bin = label_binarize(all_targets, classes=range(n_classes))
    all_predictions_bin = label_binarize(all_predictions, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_targets_bin[:, i], all_predictions_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

train_acc, train_precision, train_recall, train_f1, train_targets, train_predictions = calculate_metrics(cnn_model, train_loader)
test_acc, test_precision, test_recall, test_f1, test_targets, test_predictions = calculate_metrics(cnn_model, test_loader)
validation_acc, validation_precision, validation_recall, validation_f1, validation_targets, validation_predictions = calculate_metrics(cnn_model, validation_loader)

print(f"Train Accuracy : {train_acc}\nTrain Precision : {train_precision}\nTrain Recall : {train_recall}\nTrain F1 : {train_f1}")
print(f"Test Accuracy : {test_acc}\nTest Precision : {test_precision}\nTest Recall : {test_recall}\nTest F1 : {test_f1}")
print(f"Validation Accuracy : {validation_acc}\nValidation Precision : {validation_precision}\nValidation Recall : {validation_recall}\nValidation F1 : {validation_f1}")

plot_roc_curve(test_targets, test_predictions, targets_size)

# Evaluate pretrained VGG model
vgg_model.to(device)

vgg_train_losses, vgg_validation_losses = batch_gd(
    vgg_model, criterion, train_loader, validation_loader, 10
)

torch.save(vgg_model.state_dict(), "vgg_without.pt")
# torch.save(vgg_model.state_dict(), 'vgg_with.pt')

vgg_model.load_state_dict(torch.load("vgg_without.pt"))
# vgg_model.load_state_dict(torch.load("vgg_with.pt"))
vgg_model.eval()

vgg_train_acc, vgg_train_precision, vgg_train_recall, vgg_train_f1, vgg_train_targets, vgg_train_predictions = calculate_metrics(vgg_model, train_loader)
vgg_test_acc, vgg_test_precision, vgg_test_recall, vgg_test_f1, vgg_test_targets, vgg_test_predictions = calculate_metrics(vgg_model, test_loader)
vgg_validation_acc, vgg_validation_precision, vgg_validation_recall, vgg_validation_f1, vgg_validation_targets, vgg_validation_predictions = calculate_metrics(vgg_model, validation_loader)

print(f"VGG Train Accuracy : {vgg_train_acc}\nVGG Train Precision : {vgg_train_precision}\nVGG Train Recall : {vgg_train_recall}\nVGG Train F1 : {vgg_train_f1}")
print(f"VGG Test Accuracy : {vgg_test_acc}\nVGG Test Precision : {vgg_test_precision}\nVGG Test Recall : {vgg_test_recall}\nVGG Test F1 : {vgg_test_f1}")
print(f"VGG Validation Accuracy : {vgg_validation_acc}\nVGG Validation Precision : {vgg_validation_precision}\nVGG Validation Recall : {vgg_validation_recall}\nVGG Validation F1 : {vgg_validation_f1}")

plot_roc_curve(vgg_test_targets, vgg_test_predictions, targets_size)
