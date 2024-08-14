import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # Create an instance of the model
    model = SmallNet()

    # Wrap the model with DataParallel
    model = nn.DataParallel(model)

    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Print the model to see its structure
    print(str(model))

    # Create a sample input tensor
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 32, 32).to(device)

    # Perform a forward pass
    output = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
