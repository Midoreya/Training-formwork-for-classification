import torch
import torchvision
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from models.Resnet import load_models
from common.export import export_onnx, export_jit

def train_one_epoch(model,
                    max_epoch,
                    epoch_index,
                    device,
                    print_iteration,
                    training_loader,
                    optimizer,
                    loss_fn):
    
    running_loss = 0.
    last_loss = 0.
    
    running_ture = 0
    total = 0
        
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        
        # Make predictions for this batch
        outputs = model(inputs)
            
        pred = outputs.argmax(1)
        total += labels.size(0)
        running_ture += (pred==labels).sum()
        
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # Adjust learning weights
        optimizer.step()
        
        # Gather data and report
        running_loss += loss.item()
        accuracy = float(running_ture) / float(total)
        
        if (i % print_iteration == print_iteration - 1) or i == len(training_loader) - 1:
            last_loss = running_loss / 200 # loss per batch
            print('===========================')
            print("Epoch: [{}/{}]".format(epoch_index, max_epoch - 1))
            print('Iteration:','[{}/{}]'.format(i + 1 ,len(training_loader)))
            print('Loss:', last_loss)
            print('Learn rate:', optimizer.param_groups[0]['lr'])
            print('Accuracy:','{:.2f}%'.format(accuracy * 100))
            print('===========================')
            print()
            # print('\n')
            running_loss = 0.
            running_ture = 0
            total = 0
          
    return last_loss, accuracy

def train(num_classes=10,
          layer=18,
          input_size=32,
          max_epoch=5,
          optim='SGD',
          loss_fn = torch.nn.CrossEntropyLoss(),
          lr_init=0.001,
          step_size=50,
          gamma=0.1,
          batch_size=8,
          best_accuracy_v = 0.98,
          device=None,
          training_set=None,
          validation_set=None,
          print_iteration=100,
          svdir='./modelfile/',
          svpath='weight.pt'
          ):
    
    # Report split sizes
    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    if device == 'cuda':
        print('Cuda is available:', torch.cuda.is_available())
        if torch.cuda.is_available(): 
            torch.set_default_tensor_type(torch. cuda. FloatTensor)
            print ("Using cuda:",torch.cuda.get_device_name(0))
            pass
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, generator=torch.Generator(device=device))
        
    elif device == 'cpu':
        device = torch.device('cpu')
        training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    
    else:
        print('Error: Device only Cpu or Cuda')
    
    model = load_models(num_classes=num_classes, layers=layer)
    model.to(device)

    if optim == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr_init, momentum=0.9)
        print()
        
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  
    
    for epoch in range(max_epoch):
        
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss, accuracy = train_one_epoch(model,
                                             max_epoch,
                                             epoch,
                                             device,
                                             print_iteration,
                                             training_loader,
                                             optimizer,
                                             loss_fn
                                             )
        
        model.train(False)
        torch.cuda.empty_cache()
        scheduler.step()
        
        running_vture = 0
        total = 0
        
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            
            voutputs = model(vinputs)
                    
            pred = voutputs.argmax(1)
            total += vlabels.size(0)
            running_vture += (pred==vlabels).sum()
        
        accuracy_v = float(running_vture) / float(total)
        print('===========================')
        print("Epoch: [{}/{}]".format(epoch, max_epoch - 1))
        print('Valid in validation set:')
        print('Accuracy:')
        print('Training:   {:.2f}%'.format(accuracy * 100))
        print('Validation: {:.2f}%'.format(accuracy_v * 100))
        print('===========================')
        print()
        
        if accuracy_v > best_accuracy_v or epoch==max_epoch - 1:
            best_accuracy_v = accuracy_v
            
            import os
            if not os.path.exists(svdir):
                os.makedirs(svdir)
                
            if not svdir.endswith('/'):
                svdir = svdir + '/'
            
            torch.save(model.state_dict(),svdir + svpath)
            print('Save weight as:\n', svdir + svpath + '\n')
            
            export_onnx(model, input_size=input_size, name='model', root=svdir)
            export_jit(model, input_size=input_size, name='model', root=svdir)
 
def main(option):
    
    # training_set = datasets.CIFAR10('./data_cifar10', train=True, transform=transforms.ToTensor(), download=True)
    # validation_set = datasets.CIFAR10('./data_cifar10', train=False, transform=transforms.ToTensor(), download=True)
    
    transform = transforms.Compose([transforms.Resize((option.input_size,option.input_size)),transforms.ToTensor()])
    
    training_set = datasets.Flowers102('./data_flowers102', split='test', transform=transform, target_transform=None, download=True)
    validation_set = datasets.Flowers102('./data_flowers102', split='val', transform=transform, target_transform=None, download=True)
    
    train(num_classes=option.num_classes,
          layer=option.layer,
          input_size=option.input_size,
          max_epoch=option.max_epoch,
          optim=option.optim,
          lr_init=option.lr_init,
          step_size=option.step_size,
          gamma=option.gamma,
          batch_size=option.batch_size,
          best_accuracy_v=option.best_accuracy,
          device=option.device,
          training_set=training_set,
          validation_set=validation_set,
          print_iteration = option.print_iteration,
          svdir=option.svdir,
          svpath=option.svpath,
          )

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--layer', type=int, default=18)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_epoch', type=int, default=2)
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--input_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr_init', type=float, default=1e-3)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--best_accuracy', type=float, default=0.98)
    parser.add_argument('--print_iteration', type=int, default=100)
    parser.add_argument('--svdir', type=str, default='./modelfile/')
    parser.add_argument('--svpath', type=str, default='weight.pt')

    option = parser.parse_args()

    for arg in vars(option):
        print(format(arg + ':', '<20'), format(str(getattr(option, arg)), '<'))
    
    main(option)