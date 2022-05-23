"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):
    layers = (
        'LayerP',
        'Layer1',
        'Layer2',
        'Layer3',
        'Layer4',
        'LayerE',
        'LayerC',
    )

    layer2feat = {
        'LayerP': 'xp',
        'Layer1': 'x1',
        'Layer2': 'x2',
        'Layer3': 'x3',
        'Layer4': 'x4',
        'LayerE': 'xf',
        'LayerC': 'y',
    }
    def __init__(self, block, num_block, num_classes = 100, 
                layers = None, args = None,  imagenet_dir =None, **kwargs):
        super().__init__()

        
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if layers is not None:
            if isinstance(layers, str):
                layers = (layers,)
            assert isinstance(layers, tuple) or isinstance(layers, list)
            self.layers = tuple(layers)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def get_layer_labels(self):
        return self.layers

    def forward(self, x, preprocess=None):
        if preprocess is not None:
            x = preprocess(x)
        xi = x
        if self.layers == ('Input',): return xi,
        xp = self.conv1(x)
        if self.layers == ('LayerP',): return xp,
        x1 = self.conv2_x(xp)
        if self.layers == ('Layer1',): return x1,
        x2 = self.conv3_x(x1)
        if self.layers == ('Layer2',): return x2,
        x3 = self.conv4_x(x2)
        if self.layers == ('Layer3',): return x3,
        x4 = self.conv5_x(x3)
        if self.layers == ('Layer4',): return x4,
        x = self.avg_pool(x4)
        xf = torch.flatten(x, 1)
        if self.layers == ('LayerF',): return xf,
        y = self.fc(xf)
        feats = []
        # import ipdb;ipdb.set_trace()
        for layer in self.layers:
            feats.append(eval(self.layer2feat[layer]))
        return tuple(feats)

def resnet_cifar(model_name, no_epoch = 200, **kwargs):
    """ return a ResNet 18 object
    """

    blocks = {
        'cifar_resnet18': BasicBlock,
        'cifar_resnet34': BasicBlock,
        'cifar_resnet50': BottleNeck,
        'cifar_resnet101': BottleNeck,
        'cifar_resnet152': BottleNeck,
    }
    num_blocks ={
        'cifar_resnet18': [2, 2, 2, 2],
        'cifar_resnet34': [3, 4, 6, 3],
        'cifar_resnet50': [3, 4, 6, 3],
        'cifar_resnet101': [3, 4, 23, 3],
        'cifar_resnet152': [3, 8, 36, 3],
    }
    block = blocks[model_name]
    num_block = num_blocks[model_name]

    model = ResNet(block, num_block, **kwargs)
    if no_epoch !=0:
        para_name = '/ossfs/workspace/models/{}-{}-regular.pth'.format(model_name.split('_')[1],str(no_epoch))
        import os
        if not os.path.exists(para_name):
            para_name = para_name.replace('regular','best')
        model_para = torch.load(para_name)
        print(para_name)
        
        para={k.replace('module.',''):v for k,v in model_para.items()}
        model.load_state_dict(model_para)
        print("loading weight from {}".format(model_name))
    return model
