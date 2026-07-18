import torch
import torch.nn as nn

class SignLSTM(nn.Module):
    def __init__(self, input_size=255, hidden_size=128, num_layers=2, num_classes=114, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self,x):
        lstm_out, (hidden, cell) = self.lstm(x)

        final_hidden = hidden[-1]
        out = self.dropout(final_hidden)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    batch_size = 4
    seq_len = 40
    input_size = 225
    num_classes = 114

    dummy_input = torch.randn(batch_size, seq_len, input_size)
    model = SignLSTM(input_size=input_size, num_classes=num_classes)

    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {num_classes})")