# 训练模板

## 简介

这是一个用pytorch训练的简单模板，实现了一些简单的功能。选用的是 torchvision 中预置的Resnet，并且开放了选择层数以及类别数量的接口，也内置了两个从 torchvision 预置的数据集。

文件结构如下。

    .
    ├── common
    │   ├── export.py
    ├── models
    │   └── Resnet.py
    ├── readme.md
    └── train.py

## 快速开始

在这个模板中，我们应该创建Dataset类(torch.utils.data.dataset)并且实例化，或者从 torchvision 导入数据集如下。

    torchvision.datasets.CIFAR10()

使用 [train.py](./train.py) 训练模型。

    python3 train.py --max_epoch 50 --device cuda --dataset cifar10

参数介绍：

    --num_classes
        # Number of classes
        default=10
    --layer
        # Layer for Resnet
        default=18
    --device
        # Device 'cpu' or 'cuda'
        default='cpu'
    --max_epoch     
        # Max epoch
        default=2
    --optim
        # Only 'SGD' now
        default='SGD'
    --input_size
        # Sample size
        default=32
    --batch_size
        # Batch size
        default=8
    --lr_init
        # Learn rate init
        default=1e-3
    --step_size
        # Step size for update learn rate
        default=20
    --gamma
        # Gamma in StepLR
        default=0.1
    --best_accuracy
        # Best accuracy on validation set
        default=0.98
    --print_iteration
        # The frequency of log printing print log
        default=100
    --svdir
        # The dir path for save model file
        default='./modelfile/'
    --svpath
        # The file path for save weight
        default='weight.pt'
    --dataset
        # Using dataset, support CIFAR10, Flower102
        default='cifar10'


## 使用自定义数据集

我们需要从 **torch.utils.data.dataset** 中继承出我们自己的Dataset类，其中必须覆盖 **__getitem__()** 和 **__len__()** 函数。

**__getitem__()** 需要根据传入的索引，返回数据和标签，而 **__len__()** 返回的是整个数据集的长度。

一个简单的例子：

    class TensorDataset(Dataset):
        def __init__(self, data_tensor, target_tensor):
            self.data_tensor = data_tensor
            self.target_tensor = target_tensor

        def __getitem__(self, index):
            return self.data_tensor[index], self.target_tensor[index]

        def __len__(self):
            return self.data_tensor.size(0)

通常，我们需要构建训练集和验证集，在这个模板里面也是如此。例如：

    training_set = TensorDataset(
        training_data,
        training_label)
    validation_set = TensorDataset(
        validation_data,
        validation_label)

封装成 Dataloader 的步骤已经包含在训练函数中，最后，我们只需要调用 **train()** 即可，在这里，我们使用了一个 **main()** 函数进行封装。

    def main(option):

        training_set = TensorDataset(
            training_data,
            training_label)
        validation_set = TensorDataset(
            validation_data,
            validation_label)
        
        train(num_classes=num_classes, # Number of classes in our dataset
            device='cuda',
            training_set=training_set,
            validation_set=validation_set,
            )

    return 0
