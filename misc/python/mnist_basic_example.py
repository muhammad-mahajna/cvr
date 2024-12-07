# %%
# Imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    # Check if CUDA or MPS is available
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA for NVIDIA GPU
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Use Apple's MPS
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")  # Fallback to CPU
        print("Using CPU")

    print(f"Running on device: {device}")
except Exception as e:
    print(f"Error in setting up device: {e}")

# %%
# Data prep

try:
    # Define transformations: Convert to Tensor and normalize images
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize between [-1, 1]
    ])

    # Download and load the training and testing datasets
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)
except Exception as e:
    print(f"Error in loading data: {e}")

# %%
# Define FCN model
try:
    class FCN(nn.Module):
        def __init__(self):
            super(FCN, self).__init__()
            # Input layer: 28x28=784 flattened neurons, output 128 neurons
            self.fc1 = nn.Linear(28*28, 128)
            # Hidden layer: 128 neurons, output 64 neurons
            self.fc2 = nn.Linear(128, 64)
            # Output layer: 64 neurons, 10 output classes (digits 0-9)
            self.fc3 = nn.Linear(64, 10)
        
        def forward(self, x):
            # Flatten to a 28*28=784 vector
            x = x.view(-1, 28*28)
            # Activation: RelU
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Initialize the model, define the loss function and optimizer
    model_fcn = FCN()
    model_fcn.to(device=device)

    criterion_fcn = nn.CrossEntropyLoss()
    optimizer_fcn = torch.optim.Adam(model_fcn.parameters(), lr=0.001)
except Exception as e:
    print(f"Error in initializing FCN model: {e}")

# %%
# Define CNN model
try:
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            # 1 input channel (grayscale), 32 output channels, 3x3 kernel
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            # 32 input channels, 64 output channels, 3x3 kernel
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            # Fully connected layers
            self.fc1 = nn.Linear(64 * 12 * 12, 128)  # 12x12 from Conv2D layer
            self.fc2 = nn.Linear(128, 10)
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu_(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Initialize the CNN model, criterion, and optimizer
    model_cnn = CNN()
    model_cnn.to(device=device)

    criterion_cnn = nn.CrossEntropyLoss()
    optimizer_cnn = torch.optim.Adam(model_cnn.parameters(), lr=0.001)
except Exception as e:
    print(f"Error in initializing CNN model: {e}")

# %%
# Training loop for FCN
try:
    num_epochs = 5
    loss_arr = np.array([])
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Move data (images and labels) to GPU
            images, labels = images.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer_fcn.zero_grad()
            
            # Forward pass
            outputs = model_fcn(images)
            loss = criterion_fcn(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer_fcn.step()

            running_loss += loss.item()

        loss_arr = np.append(loss_arr, loss.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
except Exception as e:
    print(f"Error in FCN training loop: {e}")

print('Finished FCN Training')

# %%
# Training loop for CNN
try:
    num_epochs = 5
    loss_arr = np.array([])
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Move data (images and labels) to GPU
            images, labels = images.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer_cnn.zero_grad()
            
            # Forward pass
            outputs = model_cnn(images)
            loss = criterion_cnn(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer_cnn.step()

            running_loss += loss.item()

        loss_arr = np.append(loss_arr, loss.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
except Exception as e:
    print(f"Error in CNN training loop: {e}")

print('Finished CNN Training')

# %%
# Evaluate the models
try:
    model_cnn.eval()  # Set model to evaluation mode (disables dropout, batchnorm, etc.)

    correct_cnn = 0
    correct_fcn = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation for testing
        for images, labels in test_loader:
            # Move data (images and labels) to GPU
            images, labels = images.to(device), labels.to(device)

            outputs_cnn = model_cnn(images)
            _, predicted = torch.max(outputs_cnn.data, 1)
            total += labels.size(0)
            correct_cnn += (predicted == labels).sum().item()

            outputs_fcn = model_fcn(images)
            _, predicted_fcn = torch.max(outputs_fcn.data, 1)
            correct_fcn += (predicted_fcn == labels).sum().item()

    accuracy_cnn = 100 * correct_cnn / total
    accuracy_fcn = 100 * correct_fcn / total
    print(f'Accuracy of the CNN on the 10,000 test images: {accuracy_cnn:.2f}%')
    print(f'Accuracy of the FCN on the 10,000 test images: {accuracy_fcn:.2f}%')
except Exception as e:
    print(f"Error during evaluation: {e}")

# %%
# Saving model
try:
    torch.save(model_cnn.state_dict(), 'mnist_cnn_model.pth')
    model_cnn.load_state_dict(torch.load('mnist_cnn_model.pth'))
except Exception as e:
    print(f"Error in saving/loading the model: {e}")
