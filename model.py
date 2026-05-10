import torch
from torch import nn


class CNNBody(nn.Module):
    
    def __init__(self, dropout_p: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(5, 10, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(10, 25, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(25, 40, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(40 * 3 * 3, 50),
            nn.Tanh(),
            nn.Dropout(p=dropout_p),
            nn.Linear(50, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def conv_block(in_ch, out_ch, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class AudioCNN(nn.Module):
    """Base class — subclasses only need to define self.stem."""

    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        self.body = CNNBody(dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return x
    
class Model1(AudioCNN):
    #Stem: (1, 128, 173) → (5, 32, 32)#

    def __init__(self, dropout_p: float = 0.5):
        super().__init__(dropout_p)
        self.stem = nn.Sequential(
            conv_block(1, 16, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv_block(16, 32, kernel_size=3, stride=1, padding=1),
            conv_block(32, 64, kernel_size=3, stride=1, padding=1),

            nn.AdaptiveAvgPool2d((32, 32)),

            conv_block(64, 5, kernel_size=1)
            )


class Model2(AudioCNN):
    #Stem: (1, 257, 173) → (5, 32, 32)#

    def __init__(self, dropout_p: float = 0.5):
        super().__init__(dropout_p)
        self.stem = nn.Sequential(
                conv_block(1, 16, kernel_size=5, stride=2, padding=2),
                nn.MaxPool2d(kernel_size=2, stride=2),

                conv_block(16, 32, kernel_size=3, stride=(2,1), padding=1),
                conv_block(32, 64, kernel_size=3, stride=1, padding=1),

                nn.AdaptiveAvgPool2d((32, 32)),

                conv_block(64, 5, kernel_size=1)
        )


class Model3(AudioCNN):
    #Stem: (1, 128, 87) → (5, 32, 32)#

    def __init__(self, dropout_p: float = 0.5):
        super().__init__(dropout_p)
        self.stem = nn.Sequential(
            conv_block(1, 16, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv_block(16, 32, kernel_size=3, stride=1, padding=1),
            conv_block(32, 64, kernel_size=3, stride=1, padding=1),

            nn.AdaptiveAvgPool2d((32, 32)),

            conv_block(64, 5, kernel_size=1)
        )


class Model4(AudioCNN):
    #Stem: (1, 513, 87) → (5, 32, 32)#

    def __init__(self, dropout_p: float = 0.5):
        super().__init__(dropout_p)
        self.stem = nn.Sequential(
            conv_block(1, 16, kernel_size=5, stride=(2,1), padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv_block(16, 32, kernel_size=3, stride=(2,1), padding=1),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),

            conv_block(32, 64, kernel_size=3, stride=1, padding=1),

            nn.AdaptiveAvgPool2d((32, 32)),

            conv_block(64, 5, kernel_size=1)
        )



class Model5(AudioCNN):
    #Stem: (1, 128, 44) → (5, 32, 32)#

    def __init__(self, dropout_p: float = 0.5):
        super().__init__(dropout_p)
        self.stem = nn.Sequential(
            conv_block(1, 16, kernel_size=(5,3), stride=(2,1), padding=(2,1)),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv_block(16, 32, kernel_size=3, stride=1, padding=1),
            conv_block(32, 64, kernel_size=3, stride=1, padding=1),

            nn.AdaptiveAvgPool2d((32, 32)),

            conv_block(64, 5, kernel_size=1)
        )


class Model6(AudioCNN):
    #Stem: (1, 1013, 44) → (5, 32, 32)

    def __init__(self, dropout_p: float = 0.5):
        super().__init__(dropout_p)
        self.stem = nn.Sequential(
                conv_block(1, 16, kernel_size=(5,3), stride=(2,1), padding=(2,1)),
                nn.MaxPool2d(kernel_size=2, stride=2),

                conv_block(16, 32, kernel_size=3, stride=(2,1), padding=1),
                nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),

                conv_block(32, 64, kernel_size=3, stride=(2,1), padding=1),
                conv_block(64, 64, kernel_size=3, stride=1, padding=1),

                nn.AdaptiveAvgPool2d((32, 32)),

                conv_block(64, 5, kernel_size=1)
        )
