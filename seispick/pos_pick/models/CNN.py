import torch.nn as nn


class ConvNet_64_64(nn.Module):
    def __init__(self):
        super(ConvNet_64_64, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3),
                                    nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3),
                                    nn.MaxPool2d(kernel_size=2))

        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3),
                                    nn.MaxPool2d(kernel_size=2))

        # self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(6 * 6 * 64, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out



class ConvNet_64_96(nn.Module):
    def __init__(self):
        super(ConvNet_64_96, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3),
                                    nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3),
                                    nn.MaxPool2d(kernel_size=2))

        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3),
                                    nn.MaxPool2d(kernel_size=2))

        # self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(6 * 10 * 64, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class ConvNet_64_300(nn.Module):
    def __init__(self):
        super(ConvNet_64_300, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, 
                                              kernel_size=(3, 3), 
                                              stride=(3, 1), 
                                              padding=0),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 
                                              kernel_size=(3, 3), 
                                              stride=(1, 1), 
                                              padding=0),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, 
                                              kernel_size=3, 
                                              stride=(1, 1), 
                                              padding=0),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        # self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(6 * 11 * 64, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
