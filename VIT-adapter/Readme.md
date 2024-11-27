```
import torch 
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from ops.modules import MSDeformAttn
from timm.layers import DropPath
import torch.utils.checkpoint as cp
```

# [VIT-adapter](https://zhuanlan.zhihu.com/p/608272954)

ViT-Adapter，源于一篇被ICLR 2023接受的论文，是一个面向密集型预测任务的轻量级适配器。不同于以往复杂的网络架构调整，ViT-Adapter为纯视觉Transformer（ViT）带来了革命性的转变，使其能够在无需预训练微调的情况下，直接应用于对象检测、实例分割、语义分割等任务，并达到与专门设计的模型相媲美的性能。这个项目在GitHub上提供了详尽的代码实现，以及一系列实验环境配置，助力开发者和研究者快速上手，探索视觉智能的新边疆。

***Paper***:https://arxiv.org/abs/2210.03453

***Github***:https://github.com/czczup/ViT-Adapter

在 EVA 和 DINOv2 的下游任务中均采用了该方法
[DINOv2 semantic segmentation example(use ViT-adapter)](https://github.com/facebookresearch/dinov2/blob/main/notebooks/semantic_segmentation.ipynb)

## ViT-Adapter架构

![image.png](https://raw.githubusercontent.com/Ttbyl/Pictures/main/pictures/vit-adapter.jpg)

在VIT架构中额外添加Adapter作为微调，主要包含模块如下：

## Spatial Prior Module

- ***空间先验模块(Spatial Prior Module)***: 卷积可以帮助Transformer更好地学习局部空间信息。受此启发，提出了空间先验模块（SPM），通过利用卷积Stem(参考ResNet)以及若干额外的卷积层，得到了一个具有三种分辨率（1/8、1/16、1/32）的特征金字塔。最后将这些特征图展平并拼接，得到最终的空间先验特征。

![Spatial Prior Module](https://raw.githubusercontent.com/Ttbyl/Pictures/main/pictures/Spatial_Prior_module.png)

In [8]:

```
class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim,
                             kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim,
                             kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim,
                             kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim,
                             kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        def _inner_forward(x):
            # x.shape: [batch_size, channels, height, width]: [2, 3, 384, 384]
            # [2,3,384,384] conv1-> [2,64,192,192] conv2-> [2,64,192,192] conv3-> [2,64,192,192] maxpool-> [2, 64, 96, 96]
            c1 = self.stem(x)  # c1.shape: [2, 64, 96, 96]
            c2 = self.conv2(c1)  # c2.shape: [2, 128, 48, 48]
            c3 = self.conv3(c2)  # c3.shape: [2, 256, 24, 24]
            c4 = self.conv4(c3)  # c4.shape: [2, 256, 12, 12]
            c1 = self.fc1(c1)  # c1.shape: [2, 768, 96, 96]
            c2 = self.fc2(c2)  # c2.shape: [2, 768, 48, 48]   384 / 8 = 48这里是8倍下采样
            c3 = self.fc3(c3)  # c3.shape: [2, 768, 24, 24]
            c4 = self.fc4(c4)  # c4.shape: [2, 768, 12, 12]

            bs, dim, _, _ = c1.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

            return c1, c2, c3, c4

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs
```

## Spatial Feature Injector

- ***空间特征注入器(Spatial Feature Injector)***: 将空间先验特征注入到Transformer的每个阶段中，将原先ViT原本的输入特征$F_{vit}^i$即$x$作为Query,$F_{sp}^i$即$c$作为Key和Value，输入到`Cross-Attention`中。在`Injector`中还设置了一个可学习的参数$\gamma$，用于控制空间先验特征与ViT特征融合的强度。在初始化的时候设置为0，保证在训练过程中，`ViT`特征能够先被学习。 

$$
\hat{F}_{vit}^i = F_{vit}^1 + {\gamma}^i Attention(norm(F_{vit}^i),norm(F_{sp}^i)
$$

![Spatial Feature Injector](https://raw.githubusercontent.com/Ttbyl/Pictures/main/pictures/Spatial_Feature_Injector.png)

In [9]:

```
class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(
            init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query
```

## Multi-scale Feature Extractor
- ***多尺度特征提取器(Multi-scale Feature Extractor)***: 主要由Cross-Attention和FFN组成，用于提取多尺度特征。这里的输入与Injector相反，将$F_{sp}^i$作为Query,$F_{vit}^i$作为Key和Value，输入到Cross-Attention中。这里就没有设置一个可学习的参数$\gamma$，因为空间先验特征已经通过`Injector`被注入到ViT中，所以这里只需要提取多尺度特征即可。
$$
 \hat{F}_{sp}^i = F_{sp}^1 + Attention(norm(F_{sp}^i),norm(F_{vit}^{i+1}) 
$$
​		这里的$F_{vit}^{i+1}$和`Injector`相比经过了多一个`ViT Block`，所以这里是*i+1*。
$$
F_{sp}^i = \hat{F}_{sp}^i + FFN(norm(\hat{F}_{sp}^i))
$$

![Multi-scale Feature Extractor](https://raw.githubusercontent.com/Ttbyl/Pictures/main/pictures/Multi-scale_Feature_Extractor.png)

In [10]:

```
class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(
                dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(
                drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn  # 这里和Injector少了gamma可学γ

            if self.with_cffn:  # 这里和Injector多了cffn
                query = query + \
                    self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query

# FFN
class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n,
               :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n,
               :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B,
                                                   C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x
```

## 其他注意点

这里的代码片段是ViTAdapter模型的整体，它继承自TIMMVisionTransformer。ViTAdapter模型在原始ViT模型的基础上增加了一些新的模块，如SpatialPriorModule、InteractionBlock和deform_inputs。这些模块用于实现模型的自适应能力，以适应不同的任务和数据集。
模型参数如下：

![Multi-scale Feature Extractor](https://raw.githubusercontent.com/Ttbyl/Pictures/main/pictures/config_vit-adapater.png)

这里的ViT-Adapter中的注意力模块是可以改变的，作者也在这上面做了一些消融实验。

![Multi-scale Feature Extractor](https://raw.githubusercontent.com/Ttbyl/Pictures/main/pictures/Ablation_different_attention.png)

![Multi-scale Feature Extractor](https://raw.githubusercontent.com/Ttbyl/Pictures/main/pictures/ViT_ViT-adapter_feature.png)

```python
# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmseg.models.builder import BACKBONES
from ops.modules import MSDeformAttn
from timm.layers import trunc_normal_
from torch.nn.init import normal_

from base.vit import TIMMVisionTransformer
from adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs

_logger = logging.getLogger(__name__)


# @BACKBONES.register_module()
class ViTAdapter(TIMMVisionTransformer):
    def __init__(self, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4,
                 deform_num_heads=6, init_values=0., interaction_indexes=None, with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=True, pretrained=None,
                 use_extra_extractor=True, with_cp=False, *args, **kwargs):

        super().__init__(num_heads=num_heads, pretrained=pretrained,
                         with_cp=with_cp, *args, **kwargs)

        # self.num_classes = 80
        self.cls_token = None
        # 获取block的个数
        self.num_block = len(self.blocks) 
        self.pretrain_size = (pretrain_size, pretrain_size) 
        # interaction_indexes: [[0, 2], [3, 5], [6, 8], [9, 11]]
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim # 768

        # level_embed: 3 * 768
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))

        # inplanes: 64 embed_dim: 768 
        self.spm = SpatialPriorModule(
            inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)  # [1, 576, 768] -> [1, 24, 24, 768] -> [1, 768, 24, 24]
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]  # 广播相加 [2, 2304, 768] + [768]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        # x: [2, 3, 384, 384]
        # len(deform_input1): 3 / len(deform_input2): 3
        # deform_inputs1[0].shape: [1, 576, 1, 2] / deform_inputs2[0].shape: [1, 3024, 1, 2]
        # deform_inputs1[1].shape: [3, 2]         / deform_inputs2[1].shape: [1, 2]
        # deform_inputs1[2].shape: [3]            / deform_inputs2[2].shape: [1]
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        # c1.shape: [2, 768, 96, 96] tips:96*96=9216 
        # c2.shape: [2, 2304, 768]   tips: 96*96=9216 9216/4=2304
        # c3.shape: [2, 576, 768]    tips:  2304/4=576
        # c4.shape: [2, 144, 768]
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        # 三种分辨率（1/8、1/16、1/32）的特征金字塔

        # c.shape: [2, 3024, 768]
        c = torch.cat([c2, c3, c4], dim=1)  # 拼接:引入局部空间信息的同时，可以避免改变ViT 的原始结构

        # Patch Embedding forward
        # x.shape: [2, 576, 768] H: 24 W: 24 tips: 384/16=24 24*24=576
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W) # 重采样保证一致，这里的重采样是插值
        x = self.pos_drop(x + pos_embed)

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            # interations: injector +  extractor
            # layer就是InterationBlock
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous() # [2, 2304, 768] -> [2, 768, 48, 48]
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous() # [2, 576, 768] -> [2, 768, 24, 24]
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous() # [2, 144, 768] -> [2, 768, 12, 12]
        c1 = self.up(c2) + c1 # 上采样求和

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4,
                               mode='bilinear', align_corners=False)  # 上采样4倍得到 [2, 768, 24, 24] -> [2, 768, 96, 96]
            x2 = F.interpolate(x2, scale_factor=2,
                               mode='bilinear', align_corners=False) # 上采样2倍得到 [2, 768, 24, 24] -> [2, 768, 48, 48]
            x4 = F.interpolate(x4, scale_factor=0.5,
                               mode='bilinear', align_corners=False) # 下采样2倍得到 [2, 768, 24, 24] -> [2, 768, 12, 12]
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


if __name__ == '__main__':
    model = ViTAdapter(img_size=384,
                       pretrain_size=384,
                       patch_size=16,
                       embed_dim=768,
                       depth=12, # ViT中block深度，这里和下面interaction_blocks须保持一致数量
                       num_heads=12, # ViT中多头注意力机制
                       mlp_ratio=4, # ViT中MLP隐藏层倍数
                       drop_path_rate=0.3, # ViT中drop_path
                       conv_inplane=64,  # SpatialPriorModule中间conv卷积层通道数（最后会映射回dim）
                       n_points=4,  # MSDeformAttn所需参数
                       deform_num_heads=12,  # MSDeformAttn所需参数
                       cffn_ratio=0.25,  # Extractor中隐藏层的通道倍数
                       deform_ratio=0.5,  # MSDeformAttn所需参数
                       interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]], # VIT中block的间隔
                       window_attn=[False] * 12, # VIT中block是否使用window
                       window_size=[None] * 12)
    print(model)
    input = torch.randn(2, 3, 384, 384)
    model = model.cuda()
    output = model(input.cuda())
    print(len(output)) # 4
    print(output[0].shape)  # torch.Size([2, 768, 96, 96])
    print(output[1].shape)  # torch.Size([2, 768, 48, 48])
    print(output[2].shape)  # torch.Size([2, 768, 48, 48])
    print(output[3].shape)  # torch.Size([2, 768, 48, 48])
```

### 输出结果

```python
ViTAdapter(
  (patch_embed): PatchEmbed(
    (proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    (norm): Identity()
  )
  (pos_drop): Dropout(p=0.0, inplace=False)
  (blocks): Sequential(
    (0): Block(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): Identity()
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (norm): Identity()
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (drop2): Dropout(p=0.0, inplace=False)
      )
    )
    (1): Block(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): DropPath(drop_prob=0.027)
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (norm): Identity()
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (drop2): Dropout(p=0.0, inplace=False)
      )
    )
    (2): Block(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): DropPath(drop_prob=0.055)
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (norm): Identity()
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (drop2): Dropout(p=0.0, inplace=False)
      )
    )
    (3): Block(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): DropPath(drop_prob=0.082)
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (norm): Identity()
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (drop2): Dropout(p=0.0, inplace=False)
      )
    )
    (4): Block(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): DropPath(drop_prob=0.109)
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (norm): Identity()
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (drop2): Dropout(p=0.0, inplace=False)
      )
    )
    (5): Block(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): DropPath(drop_prob=0.136)
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (norm): Identity()
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (drop2): Dropout(p=0.0, inplace=False)
      )
    )
    (6): Block(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): DropPath(drop_prob=0.164)
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (norm): Identity()
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (drop2): Dropout(p=0.0, inplace=False)
      )
    )
    (7): Block(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): DropPath(drop_prob=0.191)
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (norm): Identity()
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (drop2): Dropout(p=0.0, inplace=False)
      )
    )
    (8): Block(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): DropPath(drop_prob=0.218)
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (norm): Identity()
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (drop2): Dropout(p=0.0, inplace=False)
      )
    )
    (9): Block(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): DropPath(drop_prob=0.245)
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (norm): Identity()
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (drop2): Dropout(p=0.0, inplace=False)
      )
    )
    (10): Block(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): DropPath(drop_prob=0.273)
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (norm): Identity()
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (drop2): Dropout(p=0.0, inplace=False)
      )
    )
    (11): Block(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): DropPath(drop_prob=0.300)
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (norm): Identity()
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (drop2): Dropout(p=0.0, inplace=False)
      )
    )
  )
  (spm): SpatialPriorModule(
    (stem): Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): SyncBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (4): SyncBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
      (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (7): SyncBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): ReLU(inplace=True)
      (9): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )
    (conv2): Sequential(
      (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): SyncBatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (conv3): Sequential(
      (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (conv4): Sequential(
      (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (fc1): Conv2d(64, 768, kernel_size=(1, 1), stride=(1, 1))
    (fc2): Conv2d(128, 768, kernel_size=(1, 1), stride=(1, 1))
    (fc3): Conv2d(256, 768, kernel_size=(1, 1), stride=(1, 1))
    (fc4): Conv2d(256, 768, kernel_size=(1, 1), stride=(1, 1))
  )
  (interactions): Sequential(
    (0): InteractionBlock(
      (injector): Injector(
        (query_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (feat_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (attn): MSDeformAttn(
          (sampling_offsets): Linear(in_features=768, out_features=288, bias=True)
          (attention_weights): Linear(in_features=768, out_features=144, bias=True)
          (value_proj): Linear(in_features=768, out_features=384, bias=True)
          (output_proj): Linear(in_features=384, out_features=768, bias=True)
        )
      )
      (extractor): Extractor(
        (query_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (feat_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (attn): MSDeformAttn(
          (sampling_offsets): Linear(in_features=768, out_features=96, bias=True)
          (attention_weights): Linear(in_features=768, out_features=48, bias=True)
          (value_proj): Linear(in_features=768, out_features=384, bias=True)
          (output_proj): Linear(in_features=384, out_features=768, bias=True)
        )
        (ffn): ConvFFN(
          (fc1): Linear(in_features=768, out_features=192, bias=True)
          (dwconv): DWConv(
            (dwconv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
          )
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=192, out_features=768, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (ffn_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (drop_path): DropPath(drop_prob=0.300)
      )
    )
    (1): InteractionBlock(
      (injector): Injector(
        (query_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (feat_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (attn): MSDeformAttn(
          (sampling_offsets): Linear(in_features=768, out_features=288, bias=True)
          (attention_weights): Linear(in_features=768, out_features=144, bias=True)
          (value_proj): Linear(in_features=768, out_features=384, bias=True)
          (output_proj): Linear(in_features=384, out_features=768, bias=True)
        )
      )
      (extractor): Extractor(
        (query_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (feat_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (attn): MSDeformAttn(
          (sampling_offsets): Linear(in_features=768, out_features=96, bias=True)
          (attention_weights): Linear(in_features=768, out_features=48, bias=True)
          (value_proj): Linear(in_features=768, out_features=384, bias=True)
          (output_proj): Linear(in_features=384, out_features=768, bias=True)
        )
        (ffn): ConvFFN(
          (fc1): Linear(in_features=768, out_features=192, bias=True)
          (dwconv): DWConv(
            (dwconv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
          )
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=192, out_features=768, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (ffn_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (drop_path): DropPath(drop_prob=0.300)
      )
    )
    (2): InteractionBlock(
      (injector): Injector(
        (query_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (feat_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (attn): MSDeformAttn(
          (sampling_offsets): Linear(in_features=768, out_features=288, bias=True)
          (attention_weights): Linear(in_features=768, out_features=144, bias=True)
          (value_proj): Linear(in_features=768, out_features=384, bias=True)
          (output_proj): Linear(in_features=384, out_features=768, bias=True)
        )
      )
      (extractor): Extractor(
        (query_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (feat_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (attn): MSDeformAttn(
          (sampling_offsets): Linear(in_features=768, out_features=96, bias=True)
          (attention_weights): Linear(in_features=768, out_features=48, bias=True)
          (value_proj): Linear(in_features=768, out_features=384, bias=True)
          (output_proj): Linear(in_features=384, out_features=768, bias=True)
        )
        (ffn): ConvFFN(
          (fc1): Linear(in_features=768, out_features=192, bias=True)
          (dwconv): DWConv(
            (dwconv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
          )
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=192, out_features=768, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (ffn_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (drop_path): DropPath(drop_prob=0.300)
      )
    )
    (3): InteractionBlock(
      (injector): Injector(
        (query_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (feat_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (attn): MSDeformAttn(
          (sampling_offsets): Linear(in_features=768, out_features=288, bias=True)
          (attention_weights): Linear(in_features=768, out_features=144, bias=True)
          (value_proj): Linear(in_features=768, out_features=384, bias=True)
          (output_proj): Linear(in_features=384, out_features=768, bias=True)
        )
      )
      (extractor): Extractor(
        (query_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (feat_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (attn): MSDeformAttn(
          (sampling_offsets): Linear(in_features=768, out_features=96, bias=True)
          (attention_weights): Linear(in_features=768, out_features=48, bias=True)
          (value_proj): Linear(in_features=768, out_features=384, bias=True)
          (output_proj): Linear(in_features=384, out_features=768, bias=True)
        )
        (ffn): ConvFFN(
          (fc1): Linear(in_features=768, out_features=192, bias=True)
          (dwconv): DWConv(
            (dwconv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
          )
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=192, out_features=768, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (ffn_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (drop_path): DropPath(drop_prob=0.300)
      )
      (extra_extractors): Sequential(
        (0): Extractor(
          (query_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (feat_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (attn): MSDeformAttn(
            (sampling_offsets): Linear(in_features=768, out_features=96, bias=True)
            (attention_weights): Linear(in_features=768, out_features=48, bias=True)
            (value_proj): Linear(in_features=768, out_features=384, bias=True)
            (output_proj): Linear(in_features=384, out_features=768, bias=True)
          )
          (ffn): ConvFFN(
            (fc1): Linear(in_features=768, out_features=192, bias=True)
            (dwconv): DWConv(
              (dwconv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
            )
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=192, out_features=768, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
          (ffn_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (drop_path): DropPath(drop_prob=0.300)
        )
        (1): Extractor(
          (query_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (feat_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (attn): MSDeformAttn(
            (sampling_offsets): Linear(in_features=768, out_features=96, bias=True)
            (attention_weights): Linear(in_features=768, out_features=48, bias=True)
            (value_proj): Linear(in_features=768, out_features=384, bias=True)
            (output_proj): Linear(in_features=384, out_features=768, bias=True)
          )
          (ffn): ConvFFN(
            (fc1): Linear(in_features=768, out_features=192, bias=True)
            (dwconv): DWConv(
              (dwconv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
            )
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=192, out_features=768, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
          (ffn_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (drop_path): DropPath(drop_prob=0.300)
        )
      )
    )
  )
  (up): ConvTranspose2d(768, 768, kernel_size=(2, 2), stride=(2, 2))
  (norm1): SyncBatchNorm(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (norm2): SyncBatchNorm(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (norm3): SyncBatchNorm(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (norm4): SyncBatchNorm(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
4
torch.Size([2, 768, 96, 96])
torch.Size([2, 768, 48, 48])
torch.Size([2, 768, 24, 24])
torch.Size([2, 768, 12, 12])
```

