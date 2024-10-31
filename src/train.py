import torch
from torch.utils.data import DataLoader
from data.dataloader import ShanghaiTechDataset
from models.mcnn import MCNN
from models.loss import CrowdCountingLoss
from config import Config
import time
import numpy as np

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    device = torch.device(Config.DEVICE)
    model = model.to(device)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        for images, density_maps, gt_counts in train_loader:
            images = images.to(device)
            density_maps = density_maps.to(device)
            gt_counts = gt_counts.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            pred_counts = torch.sum(outputs.view(outputs.size(0), -1), dim=1)
            
            loss, density_loss, count_loss = criterion(outputs, density_maps, pred_counts, gt_counts)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for images, density_maps, gt_counts in val_loader:
                images = images.to(device)
                density_maps = density_maps.to(device)
                gt_counts = gt_counts.to(device)
                
                outputs = model(images)
                pred_counts = torch.sum(outputs.view(outputs.size(0), -1), dim=1)
                
                loss, density_loss, count_loss = criterion(outputs, density_maps, pred_counts, gt_counts)
                running_val_loss += loss.item()
        
        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        print(f'Train Loss: {epoch_train_loss:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f}')
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_model.pth')
            
    return model, train_losses, val_losses

def main():
    # Create datasets
    train_dataset = ShanghaiTechDataset(Config.TRAIN_PATH, gt_downsample=Config.GT_DOWNSAMPLE)
    test_dataset = ShanghaiTechDataset(Config.TEST_PATH, gt_downsample=Config.GT_DOWNSAMPLE)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
    
    # Initialize model, criterion and optimizer
    model = MCNN()
    criterion = CrowdCountingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Train model
    model, train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, Config.EPOCHS
    )
    
    return model, train_losses, val_losses

if __name__ == "__main__":
    model, train_losses, val_losses = main()