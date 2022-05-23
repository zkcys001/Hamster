import os
import os.path as osp


def judge_platform(proj_name='CVPR2022'):
    if os.path.isdir('/data/kevinh'):
        platform = 'pai'
        data_root = '/data/kevinh/'
        model_root = '/model/kevinh/temp/output/{}/model/'.format(proj_name)
        torch_root = '/model/kevinh/torchvision/.torch/'
        output_root = '/output/'
    elif os.path.isdir('/ghome/huangyk'):
        platform = 'gpu'
        data_root = '/ghome/huangyk/dataset/'
        model_root = '/gdata1/huangyk/person-reid/{}/model/'.format(proj_name)
        torch_root = '/gdata/huangyk/.torch/'
        output_root = './'
        os.environ['MPLCONFIGDIR'] = './'
    elif os.path.isdir('/home/lab2/hyk'):
        platform = 'lab'
        data_root = '/home/lab2/hyk/dataset/'
        model_root = '/home/lab2/hyk/person-reid/{}/model/'.format(proj_name)
        torch_root = '/home/lab2/.torch/models/'
        output_root = './'
    elif os.path.isdir('F:/dataset'):
        platform = 'win'
        data_root = 'F:/dataset'
        model_root = 'F:/research/person-reid/{}/model/'.format(proj_name)
        torch_root = 'E:/56383/.cache/torch/'
        output_root = './'
    else:
        platform = 'local'
        data_root = '~/dataset'
        model_root = '~/research/person-reid/{}/model/'.format(proj_name)
        torch_root = '~/.cache/torch/'
        output_root = './'
    os.environ['TORCH_HOME'] = torch_root

    print("[INFO] Running on {}...".format(platform))
    return platform, data_root, model_root, torch_root, output_root


def get_reid_dataroot(platform, data_root):
    if platform == 'pai':
        reid_dir = osp.join(data_root, 'MSMT17_pt', 'fastreid')
    elif platform == 'gpu':
        reid_dir = osp.join(data_root, 'reid', 'fastreid')
    elif platform == 'lab':
        reid_dir = osp.join(data_root, 'reid', 'fastreid')
    else:
        reid_dir = osp.join(data_root, 'reid')
    return reid_dir


def get_imagenet_root(platform):
    if platform == 'pai':
        imagenet_train_dir = '/data/linshiqi047/imagenet/'
    elif platform == 'gpu':
        imagenet_train_dir = '/gpub/imagenet_raw/'
    elif platform == 'win':
        imagenet_train_dir = 'F:/dataset/classification/mini-imagenet/'
    else:
        if os.path.exists('/usr/zkc/data/ImageNet'):
            imagenet_train_dir = '/usr/zkc/data/ImageNet/'
        else:
            imagenet_train_dir = '/media/zkc/2D97AD940A9AD661/zkc/datasets/imagenet/'
    return imagenet_train_dir
