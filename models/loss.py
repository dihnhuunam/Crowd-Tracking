import torch
import torch.nn as nn

class CrowdCountingLoss(nn.Module):
    def __init__(self, weight_density=1.0, weight_count=1.0):
        super(CrowdCountingLoss, self).__init__()
        self.weight_density = weight_density
        self.weight_count = weight_count
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred_density, gt_density, pred_count, gt_count):
        density_loss = self.mse_loss(pred_density, gt_density)
        count_loss = self.l1_loss(pred_count, gt_count)
        
        total_loss = self.weight_density * density_loss + self.weight_count * count_loss
        return total_loss, density_loss, count_loss