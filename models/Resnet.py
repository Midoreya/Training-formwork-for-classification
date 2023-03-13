import torch
import torch.nn as nn

import torchvision.models as models

def load_models(num_classes=10, layers=18):
    
    if layers == 18:
        model = models.resnet18(weights=None)
        del model.fc
        model.add_module('fc', nn.Linear(512, num_classes))

    if layers == 34:
        model = models.resnet34(weights=None)
        del model.fc
        model.add_module('fc', nn.Linear(512, num_classes))
        
    if layers == 50:
        model = models.resnet50(weights=None)
        del model.fc
        model.add_module('fc', nn.Linear(2048, num_classes))
        
    if layers == 101:
        model = models.resnet101(weights=None)
        del model.fc
        model.add_module('fc', nn.Linear(2048, num_classes))
        
    if layers == 152:
        model = models.resnet152(weights=None)
        del model.fc
        model.add_module('fc', nn.Linear(2048, num_classes))
                
    print('Load Resnet: layers {}, classes {}'.format(layers, num_classes))
    return model

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--layers', type=int, default=18)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--input_size', type=int, default=224)
    option = parser.parse_args()

    for arg in vars(option):
        print(format(arg + ':', '<20'), format(str(getattr(option, arg)), '<'))
    
    model = load_models(num_classes=option.num_classes, layers=option.layers)
    # print(model)