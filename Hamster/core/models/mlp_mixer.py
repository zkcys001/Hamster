import os.path as osp
import math
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg, named_apply, load_checkpoint
from timm.models.layers import PatchEmbed, Mlp, GluMlp, GatedMlp, DropPath, lecun_normal_, to_2tuple
from timm.models.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 0.875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'stem.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = dict(
    mixer_s32_224=_cfg(),
    mixer_s16_224=_cfg(),
    mixer_b32_224=_cfg(),
    mixer_b16_224=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224-76587d61.pth',
    ),
    mixer_b16_224_in21k=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224_in21k-617b3de2.pth',
        num_classes=21843
    ),
    mixer_l32_224=_cfg(),
    mixer_l16_224=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224-92f9adc4.pth',
    ),
    mixer_l16_224_in21k=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224_in21k-846aa33c.pth',
        num_classes=21843
    ),

    # Mixer ImageNet-21K-P pretraining
    mixer_b16_224_miil_in21k=_cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/mixer_b16_224_miil_in21k.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear', num_classes=11221,
    ),
    mixer_b16_224_miil=_cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/mixer_b16_224_miil.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear',
    ),

    gmixer_12_224=_cfg(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    gmixer_24_224=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gmixer_24_224_raa-7daf7ae6.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    resmlp_12_224=_cfg(
        url='https://dl.fbaipublicfiles.com/deit/resmlp_12_no_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    resmlp_24_224=_cfg(
        url='https://dl.fbaipublicfiles.com/deit/resmlp_24_no_dist.pth',
        #url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resmlp_24_224_raa-a8256759.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    resmlp_36_224=_cfg(
        url='https://dl.fbaipublicfiles.com/deit/resmlp_36_no_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    resmlp_big_24_224=_cfg(
        url='https://dl.fbaipublicfiles.com/deit/resmlpB_24_no_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    resmlp_12_distilled_224=_cfg(
        url='https://dl.fbaipublicfiles.com/deit/resmlp_12_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    resmlp_24_distilled_224=_cfg(
        url='https://dl.fbaipublicfiles.com/deit/resmlp_24_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    resmlp_36_distilled_224=_cfg(
        url='https://dl.fbaipublicfiles.com/deit/resmlp_36_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    resmlp_big_24_distilled_224=_cfg(
        url='https://dl.fbaipublicfiles.com/deit/resmlpB_24_dist.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    resmlp_big_24_224_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/deit/resmlpB_24_22k.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    resmlp_12_224_dino=_cfg(
        url='https://dl.fbaipublicfiles.com/deit/resmlp_12_dino.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    resmlp_24_224_dino=_cfg(
        url='https://dl.fbaipublicfiles.com/deit/resmlp_24_dino.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    gmlp_ti16_224=_cfg(),
    gmlp_s16_224=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gmlp_s16_224_raa-10536d42.pth',
    ),
    gmlp_b16_224=_cfg(),
)


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0), mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))

    def forward(self, x):
        return torch.addcmul(self.beta, self.alpha, x)


class ResBlock(nn.Module):
    """ Residual MLP block w/ LayerScale and Affine 'norm'

    Based on: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=4, mlp_layer=Mlp, norm_layer=Affine,
            act_layer=nn.GELU, init_values=1e-4, drop=0., drop_path=0.):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.linear_tokens = nn.Linear(seq_len, seq_len)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, drop=drop)
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))
        # self.ls1 = nn.Parameter(torch.randn(dim, std=math.sqrt(2.0)))
        # self.ls2 = nn.Parameter(torch.randn(dim, std=math.sqrt(2.0)))
        # print(self.ls1.shape)
        # torch.nn.init.kaiming_normal_(self.ls1)
        # torch.nn.init.kaiming_normal_(self.ls2)
        torch.nn.init.kaiming_normal_(self.ls1.data.unsqueeze(-1))
        torch.nn.init.kaiming_normal_(self.ls2.data.unsqueeze(-1))

    def forward(self, x):
        x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        return x


class SpatialGatingUnit(nn.Module):
    """ Spatial Gating Unit

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(self, dim, seq_len, norm_layer=nn.LayerNorm):
        super().__init__()
        gate_dim = dim // 2
        self.norm = norm_layer(gate_dim)
        self.proj = nn.Linear(seq_len, seq_len)

    def init_weights(self):
        # special init for the projection gate, called as override by base model init
        nn.init.normal_(self.proj.weight, std=1e-6)
        nn.init.ones_(self.proj.bias)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.proj(v.transpose(-1, -2))
        return u * v.transpose(-1, -2)


class SpatialGatingBlock(nn.Module):
    """ Residual Block w/ Spatial Gating

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=4, mlp_layer=GatedMlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm = norm_layer(dim)
        sgu = partial(SpatialGatingUnit, seq_len=seq_len)
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, gate_layer=sgu, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        return x


class MlpMixer(nn.Module):

    def __init__(
            self,
            num_classes=1000,
            img_size=224,
            in_chans=3,
            patch_size=16,
            num_blocks=8,
            embed_dim=512,
            mlp_ratio=(0.5, 4.0),
            block_layer=MixerBlock,
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,
            stem_norm=False,
            **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.stem = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer=norm_layer if stem_norm else None)
        self.blocks = nn.Sequential(*[
            block_layer(
                embed_dim, self.stem.num_patches, mlp_ratio, mlp_layer=mlp_layer, norm_layer=norm_layer,
                act_layer=act_layer, drop=drop_rate, drop_path=drop_path_rate)
            for _ in range(num_blocks)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(nlhb=nlhb)
        self.layers =['Layer{}'.format(i) for i in range(25)]
        # for i, (k, v) in enumerate(self.named_parameters()):
        #     print(k)
        #     print(v)
        #     if i > 3:
        #         break

    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def get_layer_labels(self):
        return self.layers

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return x

    def forward(self, x, preprocess=None):
        if preprocess is not None:
            x = preprocess(x)
        x = self.stem(x)
        feats = []
        # feats.append(x.reshape(x.shape[0], -1))
        for i, block in enumerate(self.blocks):
            x = block(x)
            # feats.append(x.reshape(x.shape[0], -1))
            # if i == 23:
            #     feats.append(x.reshape(x.shape[0], -1))
            # # if i == 23:
            # #     feats.append(x.reshape(x.shape[0], -1))
            # #     break
            # if i <= 2 or i >= 22:
            #     feats.append(x.reshape(x.shape[0], -1))
        x = self.norm(x)
        x = x.mean(dim=1)
        feats.append(x.reshape(x.shape[0], -1))
        x = self.head(x)
        feats.append(x.reshape(x.shape[0], -1))
        return tuple(feats)


def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax=False):
    """ Mixer weight initialization (trying to match Flax defaults)
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            # nn.init.zeros_(module.weight)
            nn.init.kaiming_normal_(module.weight)  # my init
            nn.init.constant_(module.bias, head_bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # like MLP init in vit (my original init)
                # nn.init.normal_(module.weight, std=0.01)
                nn.init.kaiming_normal_(module.weight)  # my init
                # nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        # nn.init.normal_(module.bias)  # my init
                        # nn.init.normal_(module.bias, std=1e-6)
                        nn.init.zeros_(module.bias)
                    else:
                        # nn.init.normal_(module.bias)  # my init
                        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        # print(module)
        # NOTE if a parent module contains init_weights method, it can override the init of the
        # child modules as this will be called in depth-first order.
        module.init_weights()


def checkpoint_filter_fn(state_dict, model):
    """ Remap checkpoints if needed """
    if 'patch_embed.proj.weight' in state_dict:
        # Remap FB ResMlp models -> timm
        out_dict = {}
        for k, v in state_dict.items():
            k = k.replace('patch_embed.', 'stem.')
            k = k.replace('attn.', 'linear_tokens.')
            k = k.replace('mlp.', 'mlp_channels.')
            k = k.replace('gamma_', 'ls')
            if k.endswith('.alpha') or k.endswith('.beta'):
                v = v.reshape(1, 1, -1)
            out_dict[k] = v
        return out_dict
    return state_dict


def _create_mixer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for MLP-Mixer models.')

    model = build_model_with_cfg(
        MlpMixer, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model


@register_model
def mixer_s32_224(pretrained=False, **kwargs):
    """ Mixer-S/32 224x224
    Paper: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=32, num_blocks=8, embed_dim=512, **kwargs)
    model = _create_mixer('mixer_s32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_s16_224(pretrained=False, **kwargs):
    """ Mixer-S/16 224x224
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512, **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_b32_224(pretrained=False, **kwargs):
    """ Mixer-B/32 224x224
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=32, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer('mixer_b32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_b16_224(pretrained=False, **kwargs):
    """ Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer('mixer_b16_224', pretrained=False, **model_args)
    model.load_state_dict(torch.load('/ossfs/workspace/model/jx_mixer_b16_224-76587d61.pth'))
    return model


@register_model
def mixer_b16_224_in21k(pretrained=False, **kwargs):
    """ Mixer-B/16 224x224. ImageNet-21k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer('mixer_b16_224_in21k', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_l32_224(pretrained=False, **kwargs):
    """ Mixer-L/32 224x224.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=32, num_blocks=24, embed_dim=1024, **kwargs)
    model = _create_mixer('mixer_l32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_l16_224(pretrained=False, **kwargs):
    """ Mixer-L/16 224x224. ImageNet-1k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=24, embed_dim=1024, **kwargs)
    model = _create_mixer('mixer_l16_224', pretrained=False, **model_args)
    if pretrained:model.load_state_dict(torch.load())
    return model


@register_model
def mixer_l16_224_in21k(pretrained=False, **kwargs):
    """ Mixer-L/16 224x224. ImageNet-21k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=24, embed_dim=1024, **kwargs)
    model = _create_mixer('mixer_l16_224_in21k', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_b16_224_miil(pretrained=False, **kwargs):
    """ Mixer-B/16 224x224. ImageNet-21k pretrained weights.
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer('mixer_b16_224_miil', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_b16_224_miil_in21k(pretrained=False, **kwargs):
    """ Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer('mixer_b16_224_miil_in21k', pretrained=pretrained, **model_args)
    return model


@register_model
def gmixer_12_224(pretrained=False, **kwargs):
    """ Glu-Mixer-12 224x224
    Experiment by Ross Wightman, adding (Si)GLU to MLP-Mixer
    """
    model_args = dict(
        patch_size=16, num_blocks=12, embed_dim=384, mlp_ratio=(1.0, 4.0),
        mlp_layer=GluMlp, act_layer=nn.SiLU, **kwargs)
    model = _create_mixer('gmixer_12_224', pretrained=True, **model_args)
    return model


@register_model
def resmlp_12_224(pretrained=False, **kwargs):
    """ ResMLP-12
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=16, num_blocks=12, embed_dim=384, mlp_ratio=4, block_layer=ResBlock, norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_12_224', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_36_224(pretrained=False, **kwargs):
    """ ResMLP-36
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=16, num_blocks=36, embed_dim=384, mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-6), norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_36_224', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_big_24_224(pretrained=False, **kwargs):
    """ ResMLP-B-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=8, num_blocks=24, embed_dim=768, mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-6), norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_big_24_224', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_12_distilled_224(pretrained=False, **kwargs):
    """ ResMLP-12
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=16, num_blocks=12, embed_dim=384, mlp_ratio=4, block_layer=ResBlock, norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_12_distilled_224', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_24_distilled_224(pretrained=False, **kwargs):
    """ ResMLP-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=16, num_blocks=24, embed_dim=384, mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-5), norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_24_distilled_224', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_36_distilled_224(pretrained=False, **kwargs):
    """ ResMLP-36
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=16, num_blocks=36, embed_dim=384, mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-6), norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_36_distilled_224', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_big_24_distilled_224(pretrained=False, **kwargs):
    """ ResMLP-B-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=8, num_blocks=24, embed_dim=768, mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-6), norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_big_24_distilled_224', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_big_24_224_in22ft1k(pretrained=False, **kwargs):
    """ ResMLP-B-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=8, num_blocks=24, embed_dim=768, mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-6), norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_big_24_224_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_12_224_dino(pretrained=False, **kwargs):
    """ ResMLP-12
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404

    Model pretrained via DINO (self-supervised) - https://arxiv.org/abs/2104.14294
    """
    model_args = dict(
        patch_size=16, num_blocks=12, embed_dim=384, mlp_ratio=4, block_layer=ResBlock, norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_12_224_dino', pretrained=pretrained, **model_args)
    return model


@register_model
def resmlp_24_224_dino(pretrained=False, **kwargs):
    """ ResMLP-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404

    Model pretrained via DINO (self-supervised) - https://arxiv.org/abs/2104.14294
    """
    model_args = dict(
        patch_size=16, num_blocks=24, embed_dim=384, mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-5), norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_24_224_dino', pretrained=False, **model_args)
    load_checkpoint(model, "/ossfs/workspace/model/resmlp_24_dino.pth")
    return model


@register_model
def gmlp_ti16_224(pretrained=False, **kwargs):
    """ gMLP-Tiny
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=128, mlp_ratio=6, block_layer=SpatialGatingBlock,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_ti16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def gmlp_s16_224(pretrained=False, **kwargs):
    """ gMLP-Small
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=256, mlp_ratio=6, block_layer=SpatialGatingBlock,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_s16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def gmlp_b16_224(pretrained=False, **kwargs):
    """ gMLP-Base
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=512, mlp_ratio=6, block_layer=SpatialGatingBlock,
        mlp_layer=GatedMlp, **kwargs)
    model = _create_mixer('gmlp_b16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def gmixer_24_224(pretrained=False, **kwargs):
    """ Glu-Mixer-24 224x224
    Experiment by Ross Wightman, adding (Si)GLU to MLP-Mixer
    """
    model_args = dict(
        patch_size=16, num_blocks=24, embed_dim=384, mlp_ratio=(1.0, 4.0),
        mlp_layer=GluMlp, act_layer=nn.SiLU, **kwargs)
    model = _create_mixer('gmixer_24_224', pretrained=False, **model_args)
    if pretrained:
        weight_path = osp.join(kwargs['args'].torch_root, 'checkpoints', 'gmixer_24_224_raa-7daf7ae6.pth')
        load_checkpoint(model, weight_path)
        print(f'Load weights from {weight_path}')
    return model


@register_model
def resmlp_24_224(pretrained=False, **kwargs):
    """ ResMLP-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=16, num_blocks=24, embed_dim=384, mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-5), norm_layer=Affine, **kwargs)
    model = _create_mixer('resmlp_24_224', pretrained=False, **model_args)
    if pretrained:
        weight_path = osp.join(kwargs['args'].torch_root, 'checkpoints', 'resmlp_24_224_raa-a8256759.pth')
        load_checkpoint(model, weight_path)
        print(f'Load weights from {weight_path}')
    return model


__factory = {
    # 'mlpmixer_l': mixer_l16_224,
    'gmlpmixer_t': gmixer_24_224,
    'resmlp': resmlp_24_224
    # "swim_t" : 'swin_tiny_patch4_window7_224',
    # "swim_s" : 'swin_small_patch4_window7_224',
    # "swim_b" : 'swin_base_patch4_window7_224',
    # "swim_l" : 'swin_large_patch4_window7_224',
}


def MLP_backbone(model_name, **kwargs):
    model = __factory[model_name](**kwargs)
    return model
