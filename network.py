import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageDataCNN(nn.Module):
    def __init__(self):
        super(ImageDataCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 50 * 50, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 50 * 50)  # Adjust the size based on the input image size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedDataCNN(nn.Module):
    def __init__(self, num_image_channels, num_non_image_features):
        super(MixedDataCNN, self).__init__()
        # Image processing path
        self.conv1 = nn.Conv2d(num_image_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Non-image processing path
        self.fc_non_image = nn.Linear(num_non_image_features, 64)

        # Fully connected layers for combined data
        self.fc_combined = nn.Linear(64 * 50 * 50 + 64, 128)
        self.fc_final = nn.Linear(128, 1)

    def forward(self, x_image, x_non_image):
        # Image data processing
        x_image = self.pool(F.relu(self.conv1(x_image)))
        x_image = self.pool(F.relu(self.conv2(x_image)))
        x_image = x_image.view(-1, 64 * 50 * 50)

        # Non-image data processing
        x_non_image = F.relu(self.fc_non_image(x_non_image))

        # Combine paths
        x_combined = torch.cat((x_image, x_non_image), dim=1)

        # Final processing
        x_combined = F.relu(self.fc_combined(x_combined))
        x_combined = self.fc_final(x_combined)

        return x_combined


# -------------------------------------------- Models with dropout (for reference) --------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F

class ImageDataCNN_dropout(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(ImageDataCNN_dropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 50 * 50, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)  # Add dropout after the first convolutional layer
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 50 * 50)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Add dropout before the final fully connected layer
        x = self.fc2(x)
        return x


class MixedDataCNN_dropout(nn.Module):
    def __init__(self, num_image_channels, num_non_image_features):
        super(MixedDataCNN_dropout, self).__init__()
        # Image processing path
        self.conv1 = nn.Conv2d(num_image_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Non-image processing path
        self.fc_non_image = nn.Linear(num_non_image_features, 64)

        # Fully connected layers for combined data
        self.fc_combined = nn.Linear(64 * 50 * 50 + 64, 128)
        self.dropout = nn.Dropout(0.70)
        self.fc_final = nn.Linear(128, 1)

    def forward(self, x_image, x_non_image):
        # Image data processing
        x_image = self.pool(F.relu(self.conv1(x_image)))
        x_image = self.pool(F.relu(self.conv2(x_image)))
        x_image = x_image.view(-1, 64 * 50 * 50)
        

        # Non-image data processing
        x_non_image = F.relu(self.fc_non_image(x_non_image))

        # Combine paths
        x_combined = torch.cat((x_image, x_non_image), dim=1)

        # Final processing
        x_combined = F.relu(self.fc_combined(x_combined))
        x_combined = self.dropout(x_combined)
        x_combined = self.fc_final(x_combined)

        return x_combined