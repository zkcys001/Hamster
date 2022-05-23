from torch.autograd import  Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
# 3*32*32
class lenet(nn.Module):
    def __init__(self,class_num=10):
        super(lenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, class_num)
        self.layers_name = (
            'Layer1',
            'Layer2',
            'Layer3',
            'LayerE',
            'LayerC'
        )
        checkpoint = torch.load('./Lenet-cifar10.pth')

        self.load_state_dict(checkpoint['net'])
    def get_layer_labels(self):
        return self.layers_name
    def forward(self, x, extract = True):
        x1 = self.pool(F.relu(self.conv1(x)))
        x2= self.pool(F.relu(self.conv2(x1))).view(-1, 16 * 5 * 5)
        # fully connect
        x3 = F.relu(self.fc1(x2))
        x4 = F.relu(self.fc2(x3))
        x5 = self.fc3(x4)
        if not extract:
            return x5
        else:
            feats = []
            feats.append(x1)
            feats.append(x2)
            feats.append(x3)
            feats.append(x4)
            feats.append(x5)
            return tuple(feats)

class lenet_mlp(nn.Module):
    def __init__(self,class_num=10):
        super(lenet_mlp, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 32*32*3)
        self.fc2 = nn.Linear(32*32*3, 32*32*3)
        self.fc3 = nn.Linear(32*32*3, 32*32*3)
        self.fc4 = nn.Linear(32*32*3, 32*32*3)
        self.fc5 = nn.Linear(32*32*3, class_num)
        self.layers_name = (
            'Layer1',
            'Layer2',
            'Layer3',
            'LayerE',
            'LayerC'
        )
        checkpoint = torch.load('./lenet_mlp-cifar10_0epoch.pth')

        self.load_state_dict(checkpoint['net'])
    def get_layer_labels(self):
        return self.layers_name
    def forward(self, x, extract = True):
        x = x.view(-1, 32*32*3).contiguous()
        x1 = self.fc1(x)
        x2=  self.fc2(x1)
        x3 = self.fc3(x2)
        x4 = self.fc4(x3)
        # x1 = F.relu(self.fc1(x))
        # x2=  F.relu(self.fc2(x1))
        # x3 = F.relu(self.fc3(x2))
        # x4 = F.relu(self.fc4(x3))
        x5 = self.fc5(x4)
        if not extract:
            return x5
        else:
            feats = []
            feats.append(x1)
            feats.append(x2)
            feats.append(x3)
            feats.append(x4)
            feats.append(x5)
            return tuple(feats)

def LeNet(model_name, class_num=10):
    if 'mlp' in model_name:
        model = lenet_mlp(class_num=class_num)
    else:
        model = lenet(class_num=class_num)
    return model

if __name__ == '__main__':
    print(LeNet(10))