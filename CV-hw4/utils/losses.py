#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothCrossEntropyLoss(nn.Module):
    """
    标签平滑交叉熵损失
    
    这种损失函数可以帮助减少过拟合，通过在标签中引入一定程度的平滑。
    而不是对正确类别给出1，其他类别给出0，标签平滑会给正确类别一个接近1但小于1的值，
    其他类别给予一个小的非零值。
    
    参数:
        smoothing (float): 平滑因子，通常设置为0.1或0.2
        ignore_index (int): 忽略的像素标签，通常为255
        weight (Tensor, optional): 每个类别的权重
    """
    def __init__(self, smoothing=0.1, ignore_index=255, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.weight = weight
        
    def forward(self, pred, target):
        """
        计算标签平滑交叉熵损失
        
        参数:
            pred (Tensor): 预测结果，形状为 [B, C, H, W]
            target (Tensor): 目标标签，形状为 [B, H, W]
            
        返回:
            Tensor: 计算的损失值
        """
        # 获取类别数
        n_classes = pred.size(1)
        
        # 创建平滑标签
        with torch.no_grad():
            # 将target展平以便处理
            target_flat = target.view(-1)
            
            # 创建有效像素的掩码
            valid_mask = (target_flat != self.ignore_index)
            
            # 只处理有效像素
            target_valid = target_flat[valid_mask]
            if target_valid.numel() == 0:
                return torch.tensor(0.0, device=pred.device)
            
            # 预测值只取有效像素
            pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, n_classes)
            pred_valid = pred_flat[valid_mask]
            
            # 创建one-hot编码
            target_one_hot = torch.zeros_like(pred_valid)
            target_one_hot.scatter_(1, target_valid.unsqueeze(1), 1.0)
            
            # 应用标签平滑
            target_one_hot = target_one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        # 计算损失
        log_prob = F.log_softmax(pred_valid, dim=1)
        
        # 应用类别权重（如果提供）
        if self.weight is not None:
            weight = self.weight.to(pred.device)
            log_prob = log_prob * weight.unsqueeze(0)
        
        # 计算最终损失
        loss = -(target_one_hot * log_prob).sum(dim=1).mean()
        
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation
    
    Dice系数是一种测量两个集合相似度的度量，对于分割任务非常有用
    1 - Dice系数可以用作损失函数
    
    参数:
        smooth (float): 添加到分子和分母的小值，避免除0错误
        ignore_index (int): 忽略的像素标签，通常为255
        weight (Tensor, optional): 每个类别的权重
    """
    def __init__(self, smooth=1.0, ignore_index=255, weight=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.weight = weight
    
    def forward(self, pred, target):
        """
        计算Dice Loss
        
        参数:
            pred (Tensor): 预测结果，形状为 [B, C, H, W]
            target (Tensor): 目标标签，形状为 [B, H, W]
            
        返回:
            Tensor: 计算的损失值
        """
        # 获取类别数
        n_classes = pred.size(1)
        
        # 将预测转换为概率
        pred = F.softmax(pred, dim=1)
        
        # 展平预测和目标
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, n_classes)
        target = target.view(-1)
        
        # 创建有效像素的掩码
        valid_mask = (target != self.ignore_index)
        
        # 只处理有效像素
        target_valid = target[valid_mask]
        pred_valid = pred[valid_mask]
        
        if target_valid.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # 转为one-hot编码
        target_one_hot = torch.zeros_like(pred_valid)
        target_one_hot.scatter_(1, target_valid.unsqueeze(1), 1.0)
        
        # 计算每个类别的Dice系数
        dice_per_class = torch.zeros(n_classes, device=pred.device)
        for i in range(n_classes):
            if (target_one_hot[:, i].sum() > 0): # 只计算存在的类别
                intersection = (pred_valid[:, i] * target_one_hot[:, i]).sum()
                union = pred_valid[:, i].sum() + target_one_hot[:, i].sum()
                dice_per_class[i] = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # 应用类别权重（如果提供）
        if self.weight is not None:
            weight = self.weight.to(pred.device)
            dice_per_class = dice_per_class * weight
        
        # 计算平均Dice系数，并转换为损失（1-Dice）
        dice_loss = 1.0 - dice_per_class.mean()
        
        return dice_loss


class CombinedLoss(nn.Module):
    """
    组合损失函数: 结合交叉熵损失和Dice损失
    
    参数:
        ce_weight (float): 交叉熵损失的权重
        dice_weight (float): Dice损失的权重
        smoothing (float): 标签平滑因子
        ignore_index (int): 忽略的像素标签
        weight (Tensor, optional): 每个类别的权重
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5, smoothing=0.1, ignore_index=255, weight=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = LabelSmoothCrossEntropyLoss(smoothing=smoothing, ignore_index=ignore_index, weight=weight)
        self.dice_loss = DiceLoss(ignore_index=ignore_index, weight=weight)
    
    def forward(self, pred, target):
        """
        计算组合损失
        
        参数:
            pred (Tensor): 预测结果
            target (Tensor): 目标标签
            
        返回:
            Tensor: 计算的损失值
        """
        ce_loss = self.ce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        
        # 组合两种损失
        combined_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        return combined_loss
