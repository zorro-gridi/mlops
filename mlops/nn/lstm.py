import torch
from torch import nn
import logging



class LstmModel(nn.Module):
    def __init__(self,
                 input_size=8,
                 seq_len=16,
                 hidden_size=32,
                 output_size=1,
                 num_layers=1,
                 dropout=0.2,
                 model_loss_func=None,
                ):
        super(LstmModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropput = nn.Dropout(p=dropout)
        # 对于回归问题， 需要将 seq_len 和 hidden_size 合并，不像分类任务，取预测列别德 argmax logit prob
        self.loss_fn = model_loss_func
        # 全连接层
        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        # 将batched的三维矩阵转为二维liner连接层
        self.fc_layer = nn.Linear(seq_len * hidden_size, hidden_size)

        # 此处还可以写多层神经网络
        dnn_layers = []
        for i in range(num_layers):
            if i == num_layers - 1:
                # 最后一层为输出层
                dnn_layers.append(nn.ReLU())
                # 输出层
                dnn_layers.append(nn.Linear(hidden_size, output_size))
            else:
                # 此处指定义 1 个 num_layer， 因此用不到
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(hidden_size, hidden_size))
                if dropout:
                    dnn_layers.append(self.dropput)
        # 组深网
        self.dnn = nn.Sequential(*dnn_layers)


    def forward(self, x_input):
        # lstm input shape: [B(batch_size), L(sen_len), H-in(intput_size)]
        # torch built-in lstm 模型 output 返回结果格式: output, (h_n, c_n)
        x_output, (hn, cn) = self.lstm(x_input)
        # x_output shape: [N, L, D * H-out(默认为 H-hidden), D 表示是否双向 lstm]
        n, l, h = x_output.shape

        x_output = self.dropput(x_output)
        # 全连接层
        x_output = self.fc_layer(x_output.reshape(n, l * h))
        # lstm 是记忆网络，所以最后一个数据就是包含了前面所有的信息
        x_output = self.dnn(x_output)
        return x_output



if __name__ == '__main__':
    X = torch.randn(20, 50, 16)
    lstm_inst = LstmModel(input_size=16, seq_len=50)

    output = lstm_inst(X)
    logging.warning(f'X shape: {X.shape}')
    logging.warning(f'ouput shape: {output.shape}')

    logging.warning(f'序列长度: {lstm_inst.seq_len}')
