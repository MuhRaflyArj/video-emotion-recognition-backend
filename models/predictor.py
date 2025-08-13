import torch
from torch import nn
import numpy as np

class EmotiMesh_Net(nn.Module):
    def __init__(
        self, 
        input_size=1404, 
        hidden_size=192, 
        num_layers=2, 
        num_classes=5,
        cnn_out_channels=128, 
        kernel_size=7, 
        stride=1, padding=2,
        lstm_dropout=0.25, 
        classifier_dropout=0.25
    ):
        
        super(EmotiMesh_Net, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_bn = nn.BatchNorm1d(input_size)

        # 1D Conv block
        self.conv1 = nn.Conv1d(in_channels=1, 
                               out_channels=cnn_out_channels, 
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate output size from CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_size)
            cnn_out_length = self.pool1(self.conv1(dummy_input)).shape[2]
        
        self.cnn_output_size = cnn_out_channels * cnn_out_length

        # Bi-LSTM block
        self.lstm = nn.LSTM(input_size=self.cnn_output_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            bidirectional=True,
                            dropout=lstm_dropout if num_layers > 1 else 0)
        
        # Classifier head
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn_fc1 = nn.BatchNorm1d(hidden_size)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(classifier_dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        x_reshaped = x.view(batch_size * seq_length, -1)
        x_bn = self.input_bn(x_reshaped)
        
        x_cnn_input = x_bn.view(batch_size * seq_length, 1, -1)
        cnn_out = self.pool1(self.relu1(self.conv1(x_cnn_input)))
        
        lstm_input = cnn_out.view(batch_size, seq_length, -1)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(lstm_input, (h0, c0)) 
        last_time_step_output = lstm_out[:, -1, :]
        
        out = self.dropout(self.relu_fc1(self.bn_fc1(self.fc1(last_time_step_output))))
        out = self.fc2(out)
        
        return out
    
    
def predict_emotion(arr, model_path, device='cpu'):
    model = EmotiMesh_Net()

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        input_tensor = torch.tensor(np.array(arr), dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_tensor)
        probability = torch.softmax(output, dim=1)

        confidence, predicted_class = torch.max(probability, dim=1)

    return predicted_class.item(), confidence.item()