import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Define the advanced CNN model class
class AdvancedModel(nn.Module):
    def __init__(self):
        super(AdvancedModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.3)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.4)
        self.flatten = nn.Flatten()
        self.batchnorm2 = nn.BatchNorm1d(128 * 3 * 3)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.dropout3 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = self.batchnorm1(x)
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.batchnorm2(x)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

# Create a new instance of the model
model = AdvancedModel()

# Load the state_dict
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define the class names
class_names = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}

# Function to predict the class of an input image and return the class name
def predict_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Perform the prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return class_names[predicted.item()]
