# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
from mmcv import ops
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d import rotation_3d_in_axis


@MODELS.register_module()
class Single3DRoIPointExtractor(nn.Module):
    """Point-wise roi-aware Extractor.

    Extract Point-wise roi features.

    Args:
        roi_layer (dict, optional): The config of roi layer.
    """

    def __init__(self, roi_layer: Optional[dict] = None) -> None:
        super(Single3DRoIPointExtractor, self).__init__()
        self.roi_layer = self.build_roi_layers(roi_layer)

    def build_roi_layers(self, layer_cfg: dict) -> nn.Module:
        """Build roi layers using `layer_cfg`"""
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = layer_cls(**cfg)
        return roi_layers

    def forward(self, feats: Tensor, coordinate: Tensor, batch_inds: Tensor,
                rois: Tensor) -> Tensor:
        """Extract point-wise roi features.

        Args:
            feats (torch.FloatTensor): Point-wise features with
                shape (batch, npoints, channels) for pooling.
            coordinate (torch.FloatTensor): Coordinate of each point.
            batch_inds (torch.LongTensor): Indicate the batch of each point.
            rois (torch.FloatTensor): Roi boxes with batch indices.

        Returns:
            torch.FloatTensor: Pooled features
        """
        rois = rois[..., 1:]
        rois = rois.view(batch_inds, -1, rois.shape[-1])
        with torch.no_grad():
            pooled_roi_feat, pooled_empty_flag = self.roi_layer(
                coordinate, feats, rois)

            # 加的
            # 维度检查与theta初始化
            rois_flat = rois.view(-1, rois.size(-1))  # 展平后形状: (N*B, D)
            theta = torch.zeros(
                rois_flat.size(0),
                dtype=rois.dtype,
                device=rois.device
            )  # 默认全零初始化

            if rois.size(-1) >= 7:  # 维度安全检查
                theta = rois_flat[:, 6]  # 仅当维度>=7时读取第6列

            # 无效ROI掩码处理
            theta[pooled_empty_flag.view(-1) > 0] = 0.0  # 应用空特征标志

            # canonical transformation
            roi_center = rois[:, :, 0:3]
            pooled_roi_feat[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
            pooled_roi_feat = pooled_roi_feat.view(-1,
                                                   pooled_roi_feat.shape[-2],
                                                   pooled_roi_feat.shape[-1])
            # 确保theta与pooled_roi_feat设备一致
            theta = theta.to(device=pooled_roi_feat.device, dtype=pooled_roi_feat.dtype)
            assert pooled_roi_feat.device == theta.device, "设备不一致！特征在%s而角度在%s" % \
                                                           (pooled_roi_feat.device, theta.device)

            # 提取旋转输入并做维度适配
            rot_input = pooled_roi_feat[..., 0:3]  # 原shape (N, K, 3)
            ori_shape = rot_input.shape  # 保存原始形状

            # 调整维度为旋转函数需要的格式 (可能需二维或三维)
            if rot_input.dim() != 3:  # 如果因view操作变成二维
                rot_input = rot_input.reshape(-1, 3)  # 强制展平 (N*K,3)
            # pooled_roi_feat[:, :, 0:3] = rotation_3d_in_axis(
            #     pooled_roi_feat[:, :, 0:3],
            #     -(rois.view(-1, rois.shape[-1])[:, 6]),
            #     axis=2)
            # pooled_roi_feat[pooled_empty_flag.view(-1) > 0] = 0
            # 修改
            # 执行旋转（替换原有旋转代码）
            rot_output = rotation_3d_in_axis(
                rot_input,
                -theta,  # 注意负号保持原逻辑
                axis=2
            )

            # ========== 形状恢复代码 ==========
            # 将旋转结果恢复为原特征形状
            pooled_roi_feat[..., 0:3] = rot_output.view(ori_shape)

            # 最终空特征清零
            pooled_roi_feat[pooled_empty_flag.view(-1) > 0] = 0

        return pooled_roi_feat
