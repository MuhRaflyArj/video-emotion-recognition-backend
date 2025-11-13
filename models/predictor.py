import torch
from torch import nn
import numpy as np
import torchvision.models as models

class EfficientNetV2ModelS(nn.Module):
    def __init__(self, num_classes, num_input_channels):
        super(EfficientNetV2ModelS, self).__init__()

        self.efficientnet = models.efficientnet_v2_s(weights=None)

        original_first_layer = self.efficientnet.features[0][0]
        
        first_layer = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=original_first_layer.bias
        )
        
        self.efficientnet.features[0][0] = first_layer
        
        in_features = self.efficientnet.classifier[1].in_features
        hidden_dim = 512
        
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.6, inplace=False),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)
    
def predict_emotion(image_sequence, model_path, device='cpu'):
    num_classes = 5
    num_input_channels = 18
    
    model = EfficientNetV2ModelS(num_classes=num_classes, num_input_channels=num_input_channels)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        inputs = torch.from_numpy(image_sequence).float()
        
        if inputs.ndim == 5 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
            
        logits = model(inputs.to(device))
        probs = torch.softmax(logits, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

    return pred_class.item(), confidence.item()