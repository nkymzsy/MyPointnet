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
#读取文件内的文件名到字符串列表中
file_list = []
with open(file_name) as f:
    for line in f:
        file_list.append(line.strip())

# 读取h5格式点云数据
file_name = file_list[0]
[data, labels] = read_h5_data(file_name)

#从data中提取点云数据


net = net.Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(10000):  # 遍历所有训练轮次
        # 前向传播
        good = 0
        for i in range(100):
            cloud = data[i]
            # cloud 转tensor
            cloud = torch.from_numpy(cloud).float()
            outputs = net(cloud)
            # 40个为0的数组
            lable = torch.tensor(labels[i],dtype=torch.long)
            loss = criterion(outputs, lable)

            # 比较本次预测的结果和真实值是否一致
            if labels[i][0] == outputs.max(1)[1].item():
                good += 1

            # 反向传播和优化
            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重
        # 打印准确率
        print('epoch:', epoch, 'loss:', loss.item(), 'accuracy:', good / 100)



