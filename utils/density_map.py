import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def gen_density_map_gaussian(image, coords, sigma=5):
    """Generate density map given coordinates and sigma."""
    h, w = image.shape[0:2]
    density_map = np.zeros((h, w), dtype=np.float32)
    
    for coord in coords:
        x, y = int(coord[0]), int(coord[1])
        if x < w and y < h:
            density_map[y, x] = 1
            
    density_map = gaussian_filter(density_map, sigma=sigma)
    return density_map

def visualize_data(image, density_map=None, figsize=(10,5)):
    """Visualize image and its density map side by side."""
    plt.figure(figsize=figsize)
    
    plt.subplot(121)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    if density_map is not None:
        plt.subplot(122)
        plt.imshow(density_map, cmap='jet')
        plt.title(f'Density Map (Count: {np.sum(density_map):.2f})')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()