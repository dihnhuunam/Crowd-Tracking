import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, weight_dmap=0.8, weight_sum_gt=0.2):
        super().__init__()
        self.weight_dmap = weight_dmap
        self.weight_sum_gt = weight_sum_gt
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, logits, batch_dmap, batch_gts):
        # Ensure batch_gts is in float format
        batch_gts = batch_gts.float()
        
        # Calculate density map loss using MSE
        img_loss = self.mse(logits, batch_dmap)
        
        # Sum logits over spatial dimensions once and squeeze for MAE calculation
        logits_sum = logits.sum(dim=(2, 3)).squeeze()
        
        # Calculate MAE between predicted sums and ground truth sums
        gt_loss_mae = self.mae(logits_sum, batch_gts)
        
        # Weighted combination of losses
        combined_loss = self.weight_dmap * img_loss + self.weight_sum_gt * gt_loss_mae
        
        # Return the combined loss and the MAE for tracking
        return combined_loss, gt_loss_mae