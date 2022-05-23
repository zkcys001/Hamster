from .resnet import Backbone
from .vit import VIT
from .swim import SWIM
from .vgg import VGGs
from .LeNet import LeNet
from .mlp_mixer import MLP_backbone
# from .resnet_cifar import resnet_cifar


def build_model(model_name, method, use_cuda=True, use_eval=True, **kwargs):
    if 'cifar_resnet' in model_name:
        model = resnet_cifar(model_name, **kwargs)
    elif 'resnet' in model_name:
        model = Backbone(method, model_name, **kwargs)
    elif 'vit' in model_name or 'deit' in model_name:
        model = VIT(method, model_name, **kwargs)
    elif 'swim' in model_name:
        model = SWIM(method, model_name, **kwargs)
    elif 'vgg' in model_name:
        model = VGGs(method, model_name, **kwargs)
    elif 'Lenet' in model_name:
        model = LeNet(model_name)
    elif 'mlp' in model_name:
        model = MLP_backbone(model_name, **kwargs)
    else:
        assert 0, 'Invalid dataset name: {}'.format(model_name)
    if use_cuda:
        model = model.cuda()
    if use_eval:
        model = model.eval()
    return model
