import torch
import cv2
import numpy as np
from models.mcnn import MCNN
from utils.density_map import visualize_data
from config import Config

def test_image(model, image_path):
    device = torch.device(Config.DEVICE)
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to match model input requirements
    height, width = image.shape[:2]
    new_height = int(height // Config.GT_DOWNSAMPLE) * Config.GT_DOWNSAMPLE
    new_width = int(width // Config.GT_DOWNSAMPLE) * Config.GT_DOWNSAMPLE
    image = cv2.resize(image, (new_width, new_height))
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        density_map = model(image_tensor)
    
    # Convert density map to numpy
    density_map = density_map.cpu().squeeze().numpy()
    
    # Calculate estimated count
    estimated_count = np.sum(density_map)
    
    return image, density_map, estimated_count

def main():
    # Load model
    model = MCNN()
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test image
    image_path = "../input/shanghaitech-with-people-density-map/ShanghaiTech/part_B/test_data/images/IMG_1.jpg"
    image, density_map, count = test_image(model, image_path)
    
    print(f"Estimated count: {count:.2f}")
    visualize_data(image, density_map)

if __name__ == "__main__":
    main()