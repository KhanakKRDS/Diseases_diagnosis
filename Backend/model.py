import torch # PyTorch library (Handles tensors, GPU acceleration, math operations)
import torch.nn as nn # Neural network module (Defines layers, loss functions,nn.Linear, nn.Conv2d,nn.ReLU)
import torch.optim as optim # Optimization algorithms (SGD, Adam)
from torchvision import datasets, transforms # Datasets and image transformations
from torch.utils.data import DataLoader # Data loading and batching
import os # Operating system interfaces (File paths, directory management)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#directory
data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "Dataset") #path to the dataset
train_dir = os.path.join(data_dir, "train") #training images
val_dir = os.path.join(data_dir, "validation") #validation images
test_dir = os.path.join(data_dir, "test") #testing images


# Image transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)), #resize images to 128x128 pixels
    transforms.Grayscale(num_output_channels=3), #x-rays are grayscale
    transforms.ToTensor(), #convert images to PyTorch tensors
])

#load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
print("Classes:", train_dataset.classes)
print("Number of training images:", len(train_dataset))
print("Number of validation images:", len(val_dataset))
print("Number of testing images:", len(test_dataset))


#dataloaders for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

images, labels = next(iter(train_loader))#get a batch of training data
print("Image batch shape:", images.shape)#should be like [8, 3, 128, 128] (batch size, channels, height, width)
print("Label batch:", labels) #labels corresponding to the images in the batch


#number of disease classes
num_classes = len(train_dataset.classes) #normal, pneumonia, TB


# Define the CNN architecture
class DiseaseDiagnosisModel(nn.Module): #nn.Module contains the architecture of CNN
    def __init__(self, num_classes):
        super(DiseaseDiagnosisModel, self).__init__()

        self.features = nn.Sequential(
            #convolutional layers to extract features from images
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            #relu activation introduces non-linearity
            nn.ReLU(inplace=True),
            #max pooling reduces computational load and helps capture dominant features
            nn.MaxPool2d(kernel_size=2, stride=2), #reduce into half size

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(), #randomly disables some neurons during training to prevent overfitting
            nn.Linear(128 * 16 * 16, 512), #flattened feature map size after conv layers (128 channel * 16 height * 16 width = 32768)
            nn.ReLU(inplace=True),
            nn.Dropout(), # another dropout layer for extra regularization
            nn.Linear(512, num_classes)
        )

    def forward(self, x): #forward is like the main() funtion
        x = self.features(x) # extract important visual patterns
        x = x.view(x.size(0), -1) # flatten the feature map(turns the image into one long list of numbers)
        x = self.classifier(x) # decides which disease it is
        return x

model = DiseaseDiagnosisModel(num_classes).to(device) #move model to GPU if available
print(model) #print the model architecture


#loss function
criterion = nn.CrossEntropyLoss() #suitable for multi-class classification
labels = labels.to(device) #move labels to GPU if available


#optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001) #optim.Adam minimizes the loss function during training


#train
num_epochs = 10
for epoch in range(num_epochs):
    model.train() #set model to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device) #move data to GPU if available

        optimizer.zero_grad() #clear previous gradients
        outputs = model(images) #forward pass
        loss = criterion(outputs, labels) #compute loss
        loss.backward() #backward pass
        optimizer.step() #update weights

        running_loss += loss.item() * images.size(0) #accumulate loss

    epoch_loss = running_loss / len(train_dataset) #average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}") #output will look like Epoch [1/10], Loss: 0.5678

#validation
model.eval()
correct = 0
total = 0
with torch.no_grad(): #no need to compute gradients during validation
    for images, labels in val_loader: #iterate through validation data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)#forward pass
        _, predicted = torch.max(outputs, 1)#get the class with highest score
        total += labels.size(0)#total number of labels
        correct += (predicted == labels).sum().item()#count correct predictions

val_accuracy = 100 * correct / total #print accuracy
print(f"Validation Accuracy: {val_accuracy:.2f}%") #output will look like Validation Accuracy: 85.50%

#test
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.2f}%")

