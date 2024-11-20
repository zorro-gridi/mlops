import torch
from torch import nn
import logging


"""
@Desc:
    CNN + LSTM 模型的组合
@Url: https://medium.com/@mijanr/different-ways-to-combine-cnn-and-lstm-networks-for-time-series-classification-tasks-b03fc37e91b6
"""



class CNN_LSTM(nn.Module):
    def __init__(self,
                 input_size=16,
                 hidden_size=128,
                 output_size=1,
                 num_layers=1,
                 dropout=0.1,
                 model_loss_func=None):
        super(CNN_LSTM, self).__init__()
        # Sequential 表达一个构件
        self.cnn = nn.Sequential(
            # nn.Conv1d 是一个卷积核
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.loss_fn = model_loss_func

        self.input_size = input_size
        self.output_size = output_size


    def forward(self, X):
        # logging.warning(f' X shape: {X.shape}')
        # lstm input shape: (N, L, H-in), cnn Conv1d input shape: (N, H-in, L)
        # 因此需要 permute 调换1，2维的位置
        x_cnn = X.permute(0, 2, 1)
        X_cnn_out = self.cnn(x_cnn)
        # logging.warning(f'X_cnn_out shape: {X_cnn_out.shape}')

        # cnn 的输出 shape 与 输入一致，因此需要将位置调回来，再输入 lstm
        X_lstm_in = X_cnn_out.permute(0, 2, 1)
        X_lstm_out, _ = self.lstm(X_lstm_in)
        # logging.warning(f'X_lstm_out shape: {X_lstm_out.shape}')

        # 在 fc 输入中使用一层 dropout
        X_lstm_out = self.dropout(X_lstm_out)
        # logging.warning(f'X_lstm_out dropout shape: {X_lstm_out.shape}')

        # n: batch_size, h: hiden_size, l: seq_len
        n, l, h = X_lstm_out.shape
        # 因为 cnn 会压缩 输入的序列长度，把全连接层放在最后一层，可以满足自动获取 lstm 输出的序列长度
        fc_layer = nn.Linear(l * h, self.output_size)
        out = fc_layer(X_lstm_out.reshape(n, l * h))
        return out



if __name__ == '__main__':
    test = CNN_LSTM()
    print(type(test))

    # 20 为 batch_sizze, 16 为 input_channels, 50 为 seq_len
    X = torch.randn(20, 50, 16)
    output = test(X)
    logging.warning(f'output shape: {output.shape}')

    # pool of kernel_size=3, stride=2
    m = nn.MaxPool1d(kernel_size=3, stride=2)
    input = torch.randn(20, 16, 50)
    output = m(input)
    # output shape: (20, 16, 24)
    logging.warning(f'output shape: {output.shape}')
