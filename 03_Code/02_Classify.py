import torch
from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import pandas as pd
import numpy as np


abspath = "/home/vang/Downloads/"
filepath_data = "Embeddings_2025-01-15_23-11-50.csv"
totalpath_data = abspath + filepath_data
data = pd.read_csv(totalpath_data, header=None)

filepath_labels = "Lu_sR50_2025-01-06_01-40-21_output.csv"
totalpath_labels = abspath + filepath_labels
initial_labels = pd.read_csv(totalpath_labels, header=None)[:][0]
labels = initial_labels.map({'C': 0, 'D': 1}).to_numpy()
print(labels)
num_classes = len(np.unique(labels))  # Replace with the number of your classes

# Assume `features` is a torch.Tensor with the same dimensions as expected by the ResNet model
# Example: `features` should have the shape (N, C, H, W), where
# - N is the batch size,
# - C is the number of channels (3 for RGB images),
# - H is the height,
# - W is the width.

# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

# Step 2: Ensure your features are on the same device as the model (e.g., CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#features = torch.tensor(data.values, dtype=torch.float32)  # Convert DataFrame to tensor
#features = features.to(device)

# Convert to tensor and reshape to pseudo-images
features = torch.tensor(data.values, dtype=torch.float32)  # (54, 100)
features = features.view(-1, 1, 10, 10)  # Reshape to (N, C, H, W)

# Modify the first convolutional layer to accept 1 channel instead of 3
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# Modify the fully connected layer for your dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)

import torch
import torch.optim as optim

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if isinstance(labels, np.ndarray):
    labels = torch.tensor(labels, dtype=torch.long)  # Convert to PyTorch tensor

# Example training loop
model.train()
for epoch in range(1):  # Replace with the number of epochs you want
    optimizer.zero_grad()
    outputs = model(features)  # Assuming `features` is your training data
    loss = criterion(outputs, labels)  # Assuming `labels` are your true labels
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# Assuming you have a list of your class names
custom_classes = ["Control", "Dementia"]  # Replace with your actual class names

assert len(custom_classes) == num_classes, "Mismatch between custom_classes and num_classes!"


# Perform predictions
model.eval()
with torch.no_grad():
    predictions = model(features).softmax(dim=1)
    for idx, prediction in enumerate(predictions):
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        print(f"Sample {idx}: {custom_classes[class_id]}: {100 * score:.1f}%")







# Step 3: Use the model and print the predicted category
# Ensure the features have the shape (N, 3, H, W) as required by the model
#prediction = model(features).softmax(dim=1)  # dim=1 as predictions are per class
#class_ids = prediction.argmax(dim=1)  # Get the class IDs for each item in the batch

#for idx, class_id in enumerate(class_ids):
#    score = prediction[idx, class_id].item()
#    category_name = weights.meta["categories"][class_id]
#    print(f"Sample {idx}: {category_name}: {100 * score:.1f}%")








def oldSteps():
    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")