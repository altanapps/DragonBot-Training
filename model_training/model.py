import torch
import torch.nn as nn

class PredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PredictionModel, self).__init__()
        self.num_layers = num_layers  
        self.hidden_layer_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq):
        # Initializing hidden state for first input
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)

        # Forward pass through LSTM layer
        lstm_out, _ = self.lstm(input_seq, (h0, c0))

        # Only take the output from the final timestep
        lstm_out = lstm_out[:, -1, :]

        # Pass through the linear layer
        predictions = self.linear(lstm_out)
        return predictions
    