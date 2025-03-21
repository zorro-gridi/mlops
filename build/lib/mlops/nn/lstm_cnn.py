import torch
from torch import nn
import logging


"""
@Desc:
    LSTM + CNN 模型的组合
@Url: https://medium.com/@mijanr/different-ways-to-combine-cnn-and-lstm-networks-for-time-series-classification-tasks-b03fc37e91b6
"""


class LSTM_CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_CNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # same padding = (kernel_size - 1) / 2
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # flatten
            nn.Flatten(),
            # LazyLinear 自动推理获取上游隐藏层输出的矩阵维度
            nn.LazyLinear(out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes),
        )

    def forward(self, X):
        out, _ = self.lstm(X)
        out = out.permute(0, 2, 1)
        out = self.cnn(out)
        return out



if __name__ == '__main__':
    test = LSTM_CNN(32, 64, 1, 10)
    input = torch.randn(100, 50, 32)
    logging.warning(f'--------> input shape: {input.shape}')
    out = test(input)
    logging.warning(f'--------> output shape: {out.shape}')
