import net
import h5py
import numpy as np
import torch

def read_h5_data(file_name):
    f = h5py.File(file_name, 'r')
    data = f['data'][:]
    label = f['label'][:]
    return data, label

file_name = '/home/data/dataset/modelnet40_ply_hdf5_2048/train_files.txt'
# 读取文件内的文件名到字符串列表中
file_list = []
with open(file_name) as f:
    for line in f:
        file_list.append(line.strip())


# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = net.Net().to(device)  # 将模型移动到GPU
net.load_state_dict(torch.load('model.pth')) #加载参数

criterion = torch.nn.CrossEntropyLoss().to(device)  # 将损失函数移动到GPU
optimizer = torch.optim.Adam(net.parameters(), lr=0.000001)

for i in range(10000):

    # 读取h5格式点云数据
    file_name = file_list[i%4]
    [data, labels] = read_h5_data(file_name)

    # 所有点云直接存gpu
    data_gpu = torch.from_numpy(data).float().to(device)
    lable_gpu = torch.tensor(labels, dtype=torch.long).to(device)  # 将标签移动到GPU

    for epoch in range(10):  # 遍历所有训练轮次
        # 前向传播
        good = 0
        optimizer.zero_grad()  # 清零梯度
        for i in range(data.shape[0]):
            cloud = data_gpu[i]
            outputs = net(cloud)
            loss = criterion(outputs, lable_gpu[i])
            # 比较本次预测的结果和真实值是否一致
            if labels[i][0] == outputs.max(1)[1].item():
                good += 1
            loss.backward()  # 计算梯度
        optimizer.step()  # 更新权重   
        # 打印准确率
        print('epoch:', epoch, 'loss:', loss.item(), 'accuracy:', good / data.shape[0])

        # 保存模型
        torch.save(net.state_dict(), 'model.pth')

