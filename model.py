import torch
import torch.nn as nn

class SpeechRecognitionModel(nn.Module):
    def __init__(self,n_mfcc, n_class, n_hidden,n_layers, dropout,cnn_dropout):
        super(SpeechRecognitionModel,self).__init__()
        self.n_mfcc = n_mfcc
        self.n_class = n_class
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.cnn_dropout = cnn_dropout

        self.cnn2d = nn.Sequential(
            # First block
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # Reduce frequency only
            nn.Dropout2d(self.cnn_dropout),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # Reduce frequency only
            nn.Dropout2d(self.cnn_dropout),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), # Reduce both time and frequency: (T/2, F/8)
            nn.Dropout2d(self.cnn_dropout),

            # Forth block
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)), # Reduce both time and frequency: (T/4, F/8)
            nn.Dropout2d(self.cnn_dropout),
        )

        self.cnn_output_size = 6

        self.lstm = nn.LSTM(
                input_size = None
                hidden_size = self.n_hidden,
                num_layers = self.n_layers,
                bidirectional=True,
                dropout=self.dropout,
                batch_first=True
            )

        self.classifier = nn.Linear(self.n_hidden*2 , self.n_class)
        self.log_softmax = nn.LogSoftmax(dim=2)


    def _calculate_cnn_output_size()

    def forward(self,x):

        batch_size, time_steps, n_mels = x.shape
        x = x.unsqueeze(1)
        
        cnn_out = self.cnn2d(x)
        
        batch_size, channels, new_time_steps, freq_bins = cnn_out.shape
        cnn_out = cnn_out.permute(0, 2, 1, 3)
        cnn_out = cnn_out.reshape(batch_size, new_time_steps, channels * freq_bins)

        lstm_out , _ = self.lstm(cnn_out)
        logits = self.classifier(lstm_out)
        log_probs = self.log_softmax(logits)
        return log_probs
    

