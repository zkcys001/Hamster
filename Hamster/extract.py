import os
import os.path as osp
import torch
from tqdm import tqdm

from configs import get_args
from core.head import judge_platform, get_imagenet_root
from core.data import build_data_loader
from core.models import build_model
from core.pca import PCA, flatten


if __name__ == '__main__':
    # Init
    args = get_args()
    use_cuda = True

    # Data
    platform, _, _, torch_root, _ = judge_platform()
    imagenet_dir = get_imagenet_root(platform)
    args.torch_root = torch_root
    data_loader = build_data_loader(args, args.data, imagenet_dir,  batch_size=256, num_workers=4)

    # Model
    net = build_model(args.model, args.method, layers=args.layers, no_epoch=args.epoch_num, use_cuda=use_cuda,
                      pretrained=not args.wo_pretrained, args=args, imagenet_dir=imagenet_dir)
    layers = net.get_layer_labels()
    with torch.no_grad():
        test_inputs = torch.rand(1, 3, 224, 224)
        num_layers = len(net(test_inputs.cuda() if use_cuda else test_inputs))
        assert num_layers == 2
        layer_labels = ['LayerE', 'LayerC']
        # if len(layers) != num_layers:
        #     print('[WARNING] num_layers ({}) != num_feats ({})'.format(len(layers), num_layers))
        #     layers = ['Layer{}'.format(i) for i in range(num_layers)]
    pca = PCA()

    with torch.no_grad():
        # Extract features and labels
        feat_list = [[] for _ in range(len(layer_labels))]
        label_list = []
        for images, labels in tqdm(data_loader):
            feats = net(images.cuda())
            for j, feat in enumerate(feats):
                feat_list[j].append(flatten(feat).detach().cpu())
            label_list.append(labels)
        # Save features
        os.makedirs(args.save, exist_ok=True)
        for feat, layer in zip(feat_list, layer_labels):
            feat = torch.cat(feat, dim=0)
            saved_name = 'feat_{}_{}_{}_{}.pt'.format(args.model, args.data, layer, args.method)
            saved_path = osp.join(args.save, saved_name)
            torch.save(feat.detach().cpu(), saved_path, pickle_protocol=4)
            print('Saved to {}!'.format(saved_path))

        # Save labels
        labels = torch.cat(label_list, dim=0)
        saved_name = 'label_{}_{}.pt'.format(args.model, args.data)
        saved_path = osp.join(args.save, saved_name)
        torch.save(labels.detach().cpu(), saved_path, pickle_protocol=4)
        print('Saved to {}!'.format(saved_path))

        # PCA
        for layer in layer_labels:
            saved_path = osp.join(args.save, 'feat_{}_{}_{}_{}.pt'.format(args.model, args.data, layer, args.method))
            feat = torch.load(saved_path, map_location=torch.device('cuda'))
            eigenvalues, eigenvectors = pca.fit(flatten(feat))

            saved_name = 'eig_value_{}_{}_{}_{}.pt'.format(args.model, args.data, layer, args.method)
            saved_path = osp.join(args.save, saved_name)
            torch.save(eigenvalues.detach().cpu(), saved_path, pickle_protocol=4)
            print('Saved to {}!'.format(saved_path))

            saved_name = 'eig_vector_{}_{}_{}_{}.pt'.format(args.model, args.data, layer, args.method)
            saved_path = osp.join(args.save, saved_name)
            torch.save(eigenvectors.detach().cpu(), saved_path, pickle_protocol=4)
            print('Saved to {}!'.format(saved_path))
