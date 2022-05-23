import os
import os.path as osp
import torch.nn as nn
from torch.nn import functional
import json
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import resize
from tqdm import tqdm
from collections import defaultdict
import torch
from functools import partial
import logging
import random
from PIL import Image
from math import ceil
import matplotlib.pyplot as plt
from matplotlib import ticker
from core.utils.logger import Logger

from configs import get_args
from core.head import judge_platform, get_imagenet_root
from core.models import build_model
from core.utils.classify import performance_metric, extract_slice
from core.utils.imagenet import load_folder_idx_to_label, load_class_idx_to_label,\
    calc_accuracy_each_class, get_an_instance
from core.utils.image import image_pre_process
from core.pca import feature_projection, classification_projection


def load_features(model_name, layer):
    suffix = '{}_{}_{}_{}'.format(model_name, args.data, layer, args.method)
    feats = torch_load(f=osp.join(args.save, f'feat_{suffix}.pt'))
    eig_values = torch_load(f=osp.join(args.save, f'eig_value_{suffix}.pt'))
    eig_vectors = torch_load(f=osp.join(args.save, f'eig_vector_{suffix}.pt'))
    return feats, eig_values, eig_vectors


def Exp1_Draw_PCA_Acc_Curve(save_path, model_names):
    plt.figure(num=0)
    acc_each_model, n_components = [], []
    which_layer = 'LayerE'
    split_indices = [1, 2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for model_name in model_names:
        net = build_model(model_name, args.method, layers=args.layers, no_epoch=args.epoch_num, use_cuda=True,
                          pretrained=not args.wo_pretrained, args=args)
        feats, eig_values, eig_vectors = load_features(model_name, which_layer)
        # Original Accuracy
        logits = classification_projection(feats, net, which_layer, model_name)
        acc_ori = calc_accuracy_each_class(logits, targets, each_class=False)
        print(f'[{model_name}] Classification Accuracy: {acc_ori}')
        # Accuracy of Principal Components
        acc_each_model.append([])
        n_components.append([])
        for split_idx in split_indices:
            # Set split_idx
            split_idx = int(0.01 * split_idx * feats.shape[-1])
            # Accuracy with principal components
            feat_proj_p = feature_projection(feats, eig_vectors, start_idx=0, end_idx=split_idx)
            logits = classification_projection(feat_proj_p, net, which_layer, model_name)
            acc_p = calc_accuracy_each_class(logits, targets, each_class=False)
            acc_each_model[-1].append(acc_p)
            n_components[-1].append(split_idx)
            print(f'[{model_name}] {split_idx}-Dim Classification Accuracy: {acc_p}')
        # Plot Curve
        plt.plot(np.array(acc_each_model[-1]), label=model_name)
    plt.xticks(np.arange(len(n_components[0])), labels=split_indices)
    plt.xlabel('Percent of Principal Components (%)')
    plt.ylabel('Classification Accuracy (%)')
    plt.legend()
    plt.savefig(save_path)
    np.save(osp.join(args.save, 'Exp1_curve-X.npy'), np.array(n_components))
    np.save(osp.join(args.save, 'Exp1_curve-Y.npy'), np.array(acc_each_model))


def Exp1_Effective_Dimension(model_names, ratio=0.95):
    # 计算达到95%准确率时的主成分数
    # resnet18 needs 149/512 components to reach 95% original classification accuracy!
    # resnet50 needs 131/2048 components to reach 95% original classification accuracy!
    # resmlp needs 196/384 components to reach 95% original classification accuracy!
    # gmlpmixer_t needs 199/384 components to reach 95% original classification accuracy!
    # vit_t needs 109/192 components to reach 95% original classification accuracy!
    # swim_t needs 344/768 components to reach 95% original classification accuracy!

    which_layer = 'LayerE'
    for model_name in model_names:
        net = build_model(model_name, args.method, layers=args.layers, no_epoch=args.epoch_num, use_cuda=True,
                          pretrained=not args.wo_pretrained, args=args)
        feats, eig_values, eig_vectors = load_features(model_name, which_layer)
        # Original Accuracy
        logits = classification_projection(feats, net, which_layer, model_name)
        acc_ori = calc_accuracy_each_class(logits, targets, each_class=False)
        # Accuracy of Principal Components
        n_components = feats.size(1)
        for end_idx in tqdm(range(n_components)):
            feat_proj_p = feature_projection(feats, eig_vectors, start_idx=0, end_idx=end_idx)
            logits = classification_projection(feat_proj_p, net, which_layer, model_name)
            acc_p = calc_accuracy_each_class(logits, targets, each_class=False)
            if acc_p >= ratio * acc_ori:
                print(f'{model_name} needs {end_idx}/{n_components} components'
                      f' to reach 95% original classification accuracy!')
                break


def Exp2_Train_Params(model_name, lr, n_iteration=5000, w_reg=20., n_logging=1000):
    def calc_loss(feat_n, feat_p, param):
        loss = functional.mse_loss(feat_n.matmul(param), feat_p.detach())
        reg = torch.norm(param, p=1)
        return loss + w_reg * reg
    # Load data
    feats, eig_values, eig_vectors = load_features(model_name, 'LayerC')
    # Init
    param_folder = osp.join(args.save, model_name, 'params')
    os.makedirs(param_folder, exist_ok=True)
    logging.basicConfig(filename=osp.join(param_folder, '../train_info.log'), level=logging.DEBUG,
                        format='%(asctime)s    %(message)s', datefmt='%m/%d/%Y %I:%M:%S',
                        filemode='w')
    # Estimation
    best_param, best_class_idx, min_err = None, -1, float('inf')
    for class_i in tqdm(sorted(target2index.keys())):
        feats_i, targets_i, feats_i_n, feats_i_p = extract_slice(feats, targets, class_i, target2index)
        param_i = nn.Parameter(torch.zeros(feats_i_n.shape[1], 1).to(feats_i.device))

        best_param_intra_class, min_err_intra_class = None, float('inf')

        # optimizer = torch.optim.Adam([param_i])
        # optimizer = torch.optim.SGD([param_i], lr)
        for j in range(n_iteration):
            loss_i = calc_loss(feats_i_n, feats_i_p, param_i)
            loss_i.backward()
            # Update by Torch
            # optimizer.step()
            # optimizer.zero_grad()
            # Update by Ours
            param_i.data.sub_(lr * param_i.grad)
            param_i.grad.zero_()
            # Logging
            if j % n_logging == 0:
                acc_fake, acc = performance_metric(feats_i_n, feats_i_p, param_i, targets_i, mode='acc')
                err_i = acc - acc_fake
                if err_i < min_err_intra_class:
                    min_err_intra_class = err_i
                    best_param_intra_class = torch.clone(param_i.data)
                    if err_i < min_err:
                        min_err = err_i
                        best_class_idx = class_i
                        best_param = torch.clone(param_i.data)
                logging.info(
                    f'[{class_i}, {j:06d} / {n_iteration:06d}]  loss: {loss_i.item():.3f},'
                    f'  err: {err_i:.3f},'
                    f'  min_err: {min_err:.3f}, best_class_idx: {best_class_idx}'
                )
        param_filename = f'param_{class_i:04d}_err={min_err_intra_class:+.3f}.pt'
        torch.save(best_param_intra_class.detach().cpu(), osp.join(param_folder, param_filename))
    logging.info(f'best_class_idx: {best_class_idx}, min_err: {min_err:.3f}')
    torch.save(
        best_param.detach().cpu(),
        osp.join(param_folder, f'param_{best_class_idx:04d}_best_err={min_err:+.3f}.pt')
    )


def Exp2_Visualize(param_folder, save_folder, model_name, n_instance=21, n_samples=50,
                   save_demo=True, auto_resize=True):

    logger = Logger(filename=osp.join(save_folder, '../visual.log'))
    os.makedirs(save_folder, exist_ok=True)

    feats, eig_values, eig_vectors = load_features(model_name, 'LayerC')

    param_paths, class_indices, errors = [], [], []
    for param_file in os.listdir(param_folder):
        if param_file.startswith('param_') and 'best' not in param_file and param_file.endswith('.pt'):
            param_paths.append(osp.join(param_folder, param_file))
            _, class_idx, error_str = osp.splitext(param_file)[0].split('_')
            class_indices.append(int(class_idx))
            errors.append(float(error_str.split('err=')[-1]))
    indices_sorted = np.argsort(np.abs(errors))
    # errors = np.array(errors)[indices_sorted]
    param_paths = np.array(param_paths)[indices_sorted]
    class_indices = np.array(class_indices)[indices_sorted]
    # Init
    for cnt_i, (class_idx, param_path) in enumerate(zip(class_indices, param_paths)):
        if cnt_i >= n_samples: break
        # Load Weight
        class_name = index2folder[class_idx]
        param_i = torch_load(param_path)
        values_sorted, indices_sorted = torch.sort(torch.abs(param_i), dim=0, descending=True)
        values_sorted_norm = values_sorted / torch.max(values_sorted)
        # indices_sorted = [elem.item() for elem in indices_sorted]
        # Extract feature slices
        feats_i, targets_i, feats_i_n, feats_i_p = extract_slice(feats, targets, class_idx, target2index)
        err_rel_logit, var_rel_logit = performance_metric(feats_i_n, feats_i_p, param_i, mode='logit', indices_sorted=indices_sorted)
        acc_fake, acc = performance_metric(feats_i_n, feats_i_p, param_i, target=targets_i, mode='acc', indices_sorted=indices_sorted)
        # Info for this sample
        logger.write(
            f'{class_idx:04d} {class_name} \"{index2label[class_idx]}\":'
            f'  err_rel_logit={err_rel_logit:.3f},'
            f'  var_rel_logit={var_rel_logit:.3f},'
            f'  acc={acc_fake:.3f},'
            f'  acc_gt={acc:.3f},'
        )
        if save_demo:
            src = image_pre_process(Image.open(get_an_instance(class_name)))
            saved_filename = f'{class_idx:04d}_{class_name}.jpg'
            src.save(osp.join(save_folder, saved_filename))
        # Print All Principle Category
        for cnt_j in range(n_instance):
            param_idx = indices_sorted[cnt_j].item()
            class_idx_of_instance = param_idx + 1 if param_idx >= class_idx else param_idx
            class_name_of_instance = index2folder[class_idx_of_instance]
            w_norm = values_sorted_norm[cnt_j].item()
            logger.write(
                f'[{cnt_j+1:03d}] {class_name_of_instance}'
                f' \"{index2label[class_idx_of_instance]}\":'
                f' w={param_i[param_idx].item():.3f},'
                f' w_norm={w_norm:.3f},'
            )
            if save_demo:
                src = image_pre_process(Image.open(get_an_instance(class_name_of_instance)))
                if auto_resize:
                    height, width = src.size
                    src = resize(src, [ceil(width * w_norm), ceil(height * w_norm)])
                saved_filename = f'{class_idx:04d}_{class_name}_{cnt_j+1:03d}_{class_name_of_instance}.jpg'
                src.save(osp.join(save_folder, saved_filename))

        logger.write('')


def Exp2_Draw_Param_Acc_Curve(class_idx, model_name, save_folder, param_folder):
    # Init
    os.makedirs(save_folder, exist_ok=True)
    feats, eig_values, eig_vectors = load_features(model_name, 'LayerC')
    # Get params and class idx
    param_path = None
    for param_file in os.listdir(param_folder):
        if param_file.startswith('param_') and 'best' not in param_file and param_file.endswith('.pt'):
            if f'_{class_idx:04d}_' in param_file:
                param_path = osp.join(param_folder, param_file)
                break
    param_i = torch_load(param_path)
    print(f'abs_max_weight: {torch.max(torch.abs(param_i)).item()}')
    # Get features
    feats_i, targets_i, feats_i_n, feats_i_p = extract_slice(feats, targets, class_idx, target2index)
    # Compute errors and vars
    values_sorted, indices_sorted = torch.sort(torch.abs(param_i), dim=0, descending=True)
    accs, accs_gt = [], []

    if model_name == 'swim_t':
        # split_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 400, 600, 800, 999]
        split_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 100, 500, 999]
    else:
        split_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 400, 600, 800, 999]
    # split_indices = [0, 1, 2, 3, 4, 5, 6, 7, 200, 600, 999]
    # split_indices = [0, 2, 4, 6, 8, 10, 100, 200, 400, 600, 800, 1000]
    # split_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 40, 60, 80, 100]
    # split_indices = [*range(max(param_i.size()) + 1)]
    for split_idx in split_indices:
        param_idx = split_idx
        # param_idx = int(0.01 * split_idx * max(param_i.size()))
        acc_fake, acc = performance_metric(feats_i_n, feats_i_p, param_i, mode='acc', target=targets_i,
                                           n_principle=param_idx, indices_sorted=indices_sorted)
        accs.append(acc_fake)
        # print(f'{param_idx} params: acc_predicted={acc_fake}, acc_ori={acc}.')
        accs_gt.append(acc)
    # accs_gt.append(accs_gt[-1])
    # Prepare
    colors = ['#946F5D', '#000000', '#FFE7D3', '#371515', '#FFE7D3']
    dat_x = [*range(len(accs))]
    # Reverse
    # dat_x = dat_x[::-1]
    # Draw figure
    width = 0.7
    # plt.plot(dat_x, accs, label='Predicted')
    plt.plot(accs_gt, '--', color=colors[1], label='Original')
    plt.plot(dat_x, accs, linestyle="-", linewidth=2, marker='d', color=colors[0], markersize=1, label='Predicted')
    plt.bar(dat_x, accs, width=width, linestyle="-", color=colors[0])
    # r1 = list(map(lambda x: x[0]-x[1], zip(y, stds))) #上方差
    # r2 = list(map(lambda x: x[0]+x[1], zip(y, stds))) #下方差
    # plt.fill_between(x, r1, r2, alpha=0.2)
    plt.ylim((0.0, 1.05))
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    # plt.xlabel('Number of Other Categories')
    # plt.ylabel('Classification Accuracy (%)')
    # plt.legend(loc='upper left')
    if split_indices[-1] == 1000: split_indices[-1] = '1K'
    plt.xticks(dat_x, labels=split_indices, fontsize=8, rotation=0)  # , size=8
    figure_path = osp.join(save_folder, '../Exp2', f'{model_name}_{class_idx:04d}.pdf')
    os.makedirs(osp.join(save_folder, '../Exp2'), exist_ok=True)
    plt.savefig(figure_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=600)
    try:
        plt.savefig(figure_path, bbox_inches='tight', dpi=600)
    except Exception as e:
        print(e)
    plt.close()
    # np.save(figure_path.replace('.pdf', '_acc-ori.npy'), np.array(accs_gt))
    # np.save(figure_path.replace('.pdf', '_acc-pred.npy'), np.array(accs))
    print(figure_path)


def Exp2_Accuracy_by_Param(class_idx, n_param, sample_source, model_name, param_folder):
    # Init
    feats, eig_values, eig_vectors = load_features(model_name, 'LayerC')
    # Get params and class idx
    param_path = None
    for param_file in os.listdir(param_folder):
        if param_file.startswith('param_') and 'best' not in param_file and param_file.endswith('.pt'):
            if f'_{class_idx:04d}_' in param_file:
                param_path = osp.join(param_folder, param_file)
                break
    param_i = torch_load(param_path)
    print(f'max_abs_weight: {torch.max(torch.abs(param_i)).item()}')
    # Get features
    feats_i, targets_i, feats_i_n, feats_i_p = extract_slice(feats, targets, class_idx, target2index,
                                                             source=sample_source)
    # Compute errors and vars
    values_sorted, indices_sorted = torch.sort(torch.abs(param_i), dim=0, descending=True)
    acc_fake, acc = performance_metric(feats_i_n, feats_i_p, param_i, mode='acc', target=targets_i,
                                       n_principle=n_param, indices_sorted=indices_sorted)
    print(f'{n_param} params: acc_predicted={acc_fake}, acc_ori={acc}')


if __name__ == '__main__':
    # Init
    random.seed(0)
    args = get_args()
    platform, _, _, torch_root, _ = judge_platform()
    args.torch_root = torch_root
    args.imagenet_dir = get_imagenet_root(platform)
    torch_load = partial(torch.load, map_location=torch.device('cuda'))
    all_model_names = ('resnet18', 'resnet50', 'resmlp', 'gmlpmixer_t', 'vit_t', 'swim_t')

    # Load Label
    resource_folder = './resource/'
    targets = torch_load(f=osp.join(resource_folder, 'label_{}.pt'.format(args.data)))
    target2index = defaultdict(list)
    for _i, _y in enumerate(targets):
        target2index[_y.item()].append(_i)
    index2label = load_class_idx_to_label()
    label2folder = load_folder_idx_to_label()
    index2folder = {}
    for _k, _v in index2label.items():
        index2folder[_k] = label2folder[_v]

    net = build_model(args.model, args.method, layers=args.layers, no_epoch=args.epoch_num, use_cuda=True,
                      pretrained=not args.wo_pretrained, args=args)

    # Exp 1.1 --- 主成分-分类准确率 曲线
    # with torch.no_grad():
    #     Exp1_Draw_PCA_Acc_Curve(
    #         save_path=osp.join(args.save, 'Exp1.pdf'),
    #         model_names=all_model_names,
    #     )

    # Exp 1.2 --- 测量当分类准确率达到95%时的主成分数目
    # with torch.no_grad():
    #     Exp1_Effective_Dimension(all_model_names, ratio=0.95)

    # Exp 2 --- Train
    # use_training = False
    # if use_training:
    #     for model_name in ('vit_t', 'swim_t'):   # 'resnet18', 'resnet50', 'resmlp', 'gmlpmixer_t',
    #         Exp2_Train_Params(model_name=model_name, lr=args.lr)

    # Exp 2.1 --- 样本依赖性可视化
    # with torch.no_grad():
    #     Exp2_Visualize(
    #         param_folder=osp.join(args.save, args.model, 'params'),
    #         save_folder=osp.join(args.save, args.model, 'visual'),
    #         model_name=args.model,
    #         n_samples=100,
    #     )

    # Exp 2.2 --- 权重-分类准确率 曲线
    # if args.model == 'resnet50': target_class_idx = int('0333')
    # if args.model == 'gmlpmixer_t': target_class_idx = int('0013')
    # if args.model == 'swim_t': target_class_idx = int('0132')
    # # if args.model == 'swim_t': target_class_idx = int('0251')
    # with torch.no_grad():
    #     Exp2_Draw_Param_Acc_Curve(
    #         class_idx=target_class_idx,
    #         model_name=args.model,
    #         param_folder=osp.join(args.save, args.model, 'params'),
    #         save_folder=osp.join(args.save, args.model),
    #     )

    # Exp 2.3
    if args.model == 'resnet50':
        target_class_idx = int('0333')
        target_n_param = 5
    if args.model == 'gmlpmixer_t':
        target_class_idx = int('0013')
        target_n_param = 1
    if args.model == 'swim_t':
        target_class_idx = int('0132')
        target_n_param = 3
    with torch.no_grad():
        Exp2_Accuracy_by_Param(
            class_idx=target_class_idx,
            n_param=target_n_param,
            model_name=args.model,
            sample_source='pos',
            param_folder=osp.join(args.save, args.model, 'params'),
        )
