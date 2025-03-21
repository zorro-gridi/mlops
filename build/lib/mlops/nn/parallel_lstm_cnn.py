import torch
from torch import nn
import logging


"""
@Desc:
    CNN parallelizism LSTM 模型的组合
@Url: https://medium.com/@mijanr/different-ways-to-combine-cnn-and-lstm-networks-for-time-series-classification-tasks-b03fc37e91b6
"""




class Parallel_LSTM_CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Parallel_LSTM_CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=hidden_size, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(out_features=128),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(128 * 2, num_classes)


    def forward(self, X):
        X_cnn = X.permute(0, 2, 1)
        cnn_out = self.cnn(X_cnn)

        # lstm takes input of shape (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(X)
        # lstm_out:  (batch_size, seq_len, hidden_size)
        n, l, h = lstm_out.shape
        lstm_fc = nn.Linear(l * h, 128)
        # 注意 lstm 的 fc layer 层
        lstm_out = lstm_fc(lstm_out.reshape(n, l * h))
        # 最后一层使用 cat 合并输入到线性层
        out = torch.cat([cnn_out, lstm_out], dim=1)
        out = self.fc(out)
        return out



if __name__ == '__main__':
    test = Parallel_LSTM_CNN(32, 64, 10, 1)
    input = torch.randn(100, 50, 32)
    logging.warning(f'--------> input shape: {input.shape}')
    output = test(input)
    logging.warning(f'--------> output shape: {output.shape}')
