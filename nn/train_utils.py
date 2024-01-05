import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR



def train_func(model, optimizer, train_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    # 全部样本数据（或者全部 batch）训练一次，称为一个epoch
    training_loss = 0

    for idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        # 梯度重置
        optimizer.zero_grad()

        # for regression
        y_pred = model(data)
        y_pred = y_pred.reshape(target.shape)
        loss = model.loss_fn(y_pred, target)
        # 更新梯度
        loss.backward()
        optimizer.step()

        # 梯度计算，更新模型的参数
        optimizer.step()
        training_loss += loss.item()

    training_losses = round(training_loss / len(train_dataloader), 8)
    return training_losses


def test_func(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(model, nn.Module):
        model.eval()

    test_loss = 0
    # 不用更新梯度
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            y_pred = model(data)
            y_pred = y_pred.reshape(target.shape)

            batch_loss = model.loss_fn(y_pred, target)
            test_loss += batch_loss.item()

    metric_loss = round(test_loss / len(dataloader), 8)
    return metric_loss