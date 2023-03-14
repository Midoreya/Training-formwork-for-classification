# Training formwork for classification

## Introduce

This is a simple formwork for training with pytorch, and only implemented some simple functions. Choose Resnet in torchvision and dataset also import from torchvision.

The file structure for this repository like.

    .
    ├── common
    │   ├── export.py
    ├── models
    │   └── Resnet.py
    ├── readme.md
    └── train.py

## Quick start

In this formwork, we should create class Dataset(torch.utils.data.dataset), or import dataset from torchvision. Example

    torchvision.datasets.CIFAR10()

Use [train.py](./train.py) to training a Resnet model on torchvision dataset.

    python3 train.py --max_epoch 50 --device cuda --dataset cifar10

Notes on some parameters

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


## Using custom dataset

All datasets that represent a map from keys to data samples should subclass it. All subclasses should overwrite **__getitem__()**, supporting fetching a data sample for a given key. Subclasses could also optionally overwrite **__len__()**, which is expected to return the size of the dataset by many Sampler implementations and the default options of DataLoader.

This is a simple example.

    class TensorDataset(Dataset):
        def __init__(self, data_tensor, target_tensor):
            self.data_tensor = data_tensor
            self.target_tensor = target_tensor

        def __getitem__(self, index):
            return self.data_tensor[index], self.target_tensor[index]

        def __len__(self):
            return self.data_tensor.size(0)

We need to build the training_set, validation_set. Example

    training_set = TensorDataset(
        training_data,
        training_label)
    validation_set = TensorDataset(
        validation_data,
        validation_label)

Finally, we should define main function to call function **train()**

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
