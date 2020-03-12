import time

import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import classification_report

from plot_helper import plot_curve



def _train_epoch(model, optimizer, epoch, x, y, batch_size):
    """
        进行1轮模型训练
        param model: 模型
        param optimier: 优化器
        param epoch:当前轮次
        param x:输入数据，需要为Tensor
        param y:标签,必须为Tensor
        param batch_size:批处理大小
    """
    model.train()
    start_time = time.time()
    loss = 0
    correct = 0
    for index in range(0, len(x), batch_size):
        data = x[index : index + batch_size]
        label = y[index : index + batch_size]
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1)[1]
        correct += pred.eq(label.data).cpu().sum()
        
    print("Train epoch: {} \t Time elapse: {}s".format(epoch, int(time.time() - start_time)))
    print("Train Loss: {}  \t 训练准确率: {}".format(round(loss.item(), 4), round(correct.item() / len(x), 4)))
    
    loss_epoch = round(loss.item(), 4)
    acc_epoch  = round(correct.item()/len(x),4)

    return loss_epoch,acc_epoch

def _test_epoch(is_val, model, epoch, x, y, batch_size):
    """ 
        进行1轮模型测试
        param model: 模型
        param optimier: 优化器
        param epoch:当前轮次
        param x:输入数据，需要为Tensor
        param y:标签,必须为Tensor
        param batch_size:批处理大小
    """
    model.eval()
    loss = 0
    correct = 0
    test_result = []
    for index in range(0, len(x), batch_size):
        data = x[index : index + batch_size]
        label = y[index : index + batch_size]
        output = model(data)
        loss += F.cross_entropy(output, label, size_average=False).item()
        pred = output.data.max(1)[1]
        correct += pred.eq(label.data).cpu().sum()
        test_result += [i for i in pred.cpu().numpy()]

    if(is_val):
        print("Val loss: {} \t 验证准确率: {}".format(round(loss / len(x), 6), round(correct.item() / len(x), 6)))
    else:
        print("test loss: {} \t 测试准确率: {}".format(round(loss / len(x), 6), round(correct.item() / len(x), 6)))
        print(classification_report(y.cpu(), test_result, digits=6))
        
    print()
    loss_epoch = round(loss / len(x), 4)
    acc_epoch  = round(correct.item() / len(x), 4)

    return loss_epoch,acc_epoch
        
def train(model,train_x,train_y,val_x,val_y, epochs, batch_size,optimizer=None):
    """
        模型训练并将模型参数，并将训练过程中每次在验证集上效果有提升时的参数进行保存
        param model: 待训练模型
        param optimizer: 优化器
        param train_x: 训练数据，Tensor类型
        param train_y: 训练数据标签，Tensor类型
        param val_x: 验证集数据，Tensor类型
        param val_y: 验证集标签，Tensor类型
        param epochs: 要训练的轮数
        param batch_size: 批处理大小
        param optimizer: 优化器

    """
    max_val_score = 0
    
    train_loss_list,train_acc_list = [],[]
    val_loss_list,val_acc_list = [],[]
    
    if optimizer is None:
        optimizer = optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=0.00001)
        
    for epoch in range(1, epochs + 1):
        
        train_loss_epoch,train_acc_epoch = _train_epoch(model, optimizer, epoch, train_x, train_y, batch_size)
        if val_x is not None:
            val_loss_epoch,val_acc_epoch = _test_epoch(True, model, epoch, val_x, val_y, batch_size)
        
        train_loss_list.append(train_loss_epoch)
        train_acc_list.append(train_acc_epoch)
        if val_x is not None:
            val_loss_list.append(val_loss_epoch)
            val_acc_list.append(val_acc_epoch)
            
        if epoch%5==0:
            state = model.state_dict()
            torch.save(state, './model/model-epoch-{}.state'.format(epoch))
    
    plt_curve(range(1,epochs + 1),[train_loss_list,val_loss_list],["train","test"],"Loss Curve","epoch","Loss")
    plt_curve(range(1,epochs + 1),[train_acc_list,val_acc_list],["train","test"],"Acc Curve","epoch","Acc")
    
    
        
def test(model, test_x, test_y, batch_size):
    """
        在测试集上上进行效果测试
        param model: 待测试模型
        param test_x: 测试集数据，Tensor
        param test_y: 测试集标签，Tensor
        param batch_size: 批处理大小
        
    """
    _test_epoch(False, model, 1, test_x, test_y, batch_size)
    
def predict(model, x, batch_size):
    """
        对指定数据进行预测
        param model: 进行预测的模型
        param x: 要进行预测的数据向量，Tensor
        param batch_size: 批处理大小
        return result:预测的label值，list
    """
    
    model.eval()
    result = []
    for index in range(0, len(x), batch_size):
        data = x[index : index + batch_size]
        output = model(data)
        pred = output.data.max(1)[1]
        result += [i for i in pred.cpu().numpy()]
    return result