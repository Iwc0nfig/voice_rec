# FILE: model.py

import torch
import torch.nn as nn

class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_mels, n_class, n_hidden, n_layers,
                 dropout=0.3, cnn_dropout=0.3, activation='gelu'):
        super().__init__()
        
        # Choose activation function
        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        elif activation == 'mish':
            act_fn = nn.Mish()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 1-D CNN front-end: (batch, channels= n_mels, time)
        self.cnn1d = nn.Sequential(
            # First layer: capture local acoustic patterns
            nn.Conv1d(n_mels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            act_fn,
            nn.Dropout(cnn_dropout),
            
            # Second layer: broader temporal context
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            act_fn,
            nn.Dropout(cnn_dropout),
            
            # Third layer: even broader context
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            act_fn,
            nn.Dropout(cnn_dropout)
        )

        # Flatten channel dimension only
        self.projection = nn.Sequential(
            nn.Linear(512, 2*n_hidden),
            nn.LayerNorm(2*n_hidden),
            act_fn,
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=2*n_hidden,
            hidden_size=n_hidden,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        self.classifier  = nn.Linear(n_hidden*2, n_class)
        self.log_softmax = nn.LogSoftmax(dim=-1)


    def forward(self, x):                    # x: (batch, time, n_mels)
        x = x.transpose(1, 2)               # ➜ (batch, n_mels, time)
        cnn_out = self.cnn1d(x)             # (batch, 512, time)
        cnn_out = cnn_out.transpose(1, 2)   # ➜ (batch, time, 512)
        proj_out = self.projection(cnn_out) # (batch, time, 2*n_hidden)
        lstm_out, _ = self.lstm(proj_out)
        logits = self.classifier(lstm_out)
        return self.log_softmax(logits)