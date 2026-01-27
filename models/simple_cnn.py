import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim

from utils.train_test_metrics import DEVICE


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def init_model_cnn(learning_rate=0.001):
    print("Init model...")

    torch.cuda.empty_cache()

    model = CNN().to(DEVICE)
    model_name = "CNN_MNIST"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    print("Done initializing model.")
    print(
        f"Model ID: {id(model)}, Optimizer ID: {id(optimizer)}, Criterion ID: {id(criterion)}"
    )

    return model, model_name, criterion, optimizer, transform


def load_model_cnn(model_pth_path):
    print("Load model...")

    model, model_name, criterion, optimizer, transform = init_model_cnn()

    model.load_state_dict(
        torch.load(model_pth_path, weights_only=True, map_location=DEVICE)
    )

    print("Done loading model.")

    return model, model_name, criterion, optimizer, transform
