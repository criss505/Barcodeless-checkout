from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import os

app = Flask(__name__)

class Block50(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Block50, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.SiLU()
        )
        self.downsample = downsample

    def forward(self, x):
        out = self.layers(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x = torch.add(x, out)
        x = nn.SiLU()(x)

        return x

class ResNet50(nn.Module):
    def __init__(self, num_classes=31):
        super(ResNet50, self).__init__()

        # init
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # blocks
        self.layer1 = self._add_block(64, 3, 1)
        self.layer2 = self._add_block(128, 4, 2)
        self.layer3 = self._add_block(256, 6, 2)
        self.layer4 = self._add_block(512, 3, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Block50.expansion, num_classes)

    def _add_block(self, out_channels, blocks, stride):
        if stride != 1 or self.in_channels != out_channels * Block50.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * Block50.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Block50.expansion)
            )
        else:
            downsample = None

        layers = [Block50(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * Block50.expansion

        for block in range(1, blocks):
            layers.append(Block50(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# load the ResNet model to CPU
resnet_model = ResNet50(num_classes=14)
# model_path = 'resnet_model.pth'
model_path = 'trained_model.pth'
resnet_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
resnet_model.eval()

# preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    img = Image.open(file.stream)
    img_array = np.array(img)

    # preprocess
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    # predict
    with torch.no_grad():
        output = resnet_model(input_batch)
    _, predicted_class = torch.max(output, 1)
    
    return jsonify({'class_id': predicted_class.item()})
	
@app.route('/health-check', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
