import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from model.model_CA import DeadEndDetectionModel
except ImportError:
    print("Warning: Could not import DeadEndDetectionModel. Make sure the model is in the correct location.")

class NoAnnotationDataset(Dataset):
    """Dataset for inference/visualization without ground truth annotations"""
    
    def __init__(self, data_root, transform=None, num_points=4096):
        """
        Initialize dataset for inference without annotations
        
        Args:
            data_root: Path to directory containing bag folders
            transform: Image transformations
            num_points: Number of points to sample from LiDAR data
        """
        self.data_root = data_root
        self.transform = transform if transform else self.get_default_transform()
        self.num_points = num_points
        
        # Find all bag directories
        self.bag_dirs = [d for d in glob.glob(os.path.join(data_root, "bag*"))]
        if not self.bag_dirs:
            self.bag_dirs = [data_root]  # Use data_root directly if no bag* subdirectories
        
        # Initialize list to store all sample paths
        self.samples = []
        
        # Load samples from each bag
        for bag_dir in self.bag_dirs:
            # Find all sample directories in this bag's images folder
            images_dir = os.path.join(bag_dir, "images")
            if not os.path.exists(images_dir):
                print(f"Warning: Images directory not found in {bag_dir}")
                continue
                
            sample_dirs = glob.glob(os.path.join(images_dir, "sample_id_*"))
            
            for sample_dir in sample_dirs:
                sample_id = os.path.basename(sample_dir)
                # Check if images exist
                front_img_path = os.path.join(sample_dir, "front.jpg")
                right_img_path = os.path.join(sample_dir, "side_right.jpg")
                left_img_path = os.path.join(sample_dir, "side_left.jpg")
                
                if all(os.path.exists(path) for path in [front_img_path, right_img_path, left_img_path]):
                    # Check if lidar data exists
                    lidar_dir = os.path.join(bag_dir, "lidar", sample_id)
                    front_lidar_path = os.path.join(lidar_dir, "front.bin")
                    right_lidar_path = os.path.join(lidar_dir, "side_right.bin")
                    left_lidar_path = os.path.join(lidar_dir, "side_left.bin")
                    
                    if all(os.path.exists(path) for path in [front_lidar_path, right_lidar_path, left_lidar_path]):
                        self.samples.append({
                            'sample_id': sample_id,
                            'bag_dir': bag_dir,
                            'img_paths': [front_img_path, right_img_path, left_img_path],
                            'lidar_paths': [front_lidar_path, right_lidar_path, left_lidar_path]
                        })
        
        print(f"Found {len(self.samples)} valid samples for visualization")
    
    def get_default_transform(self):
        """Default image transformations"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def read_lidar_bin(self, bin_path):
        """Read LiDAR point cloud from binary file"""
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity
        
        # Sample points if needed
        if points.shape[0] > self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[indices, :]
        
        return points[:, :3]  # Return only x, y, z
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Get sample info
        sample_info = self.samples[idx]
        sample_id = sample_info['sample_id']
        bag_dir = sample_info['bag_dir']
        
        # Load images
        front_img_path, right_img_path, left_img_path = sample_info['img_paths']
        
        front_img = Image.open(front_img_path).convert('RGB')
        right_img = Image.open(right_img_path).convert('RGB')
        left_img = Image.open(left_img_path).convert('RGB')
        
        if self.transform:
            front_img = self.transform(front_img)
            right_img = self.transform(right_img)
            left_img = self.transform(left_img)
        
        # Load LiDAR data
        front_lidar_path, right_lidar_path, left_lidar_path = sample_info['lidar_paths']
        
        front_lidar = torch.from_numpy(self.read_lidar_bin(front_lidar_path)).float()
        right_lidar = torch.from_numpy(self.read_lidar_bin(right_lidar_path)).float()
        left_lidar = torch.from_numpy(self.read_lidar_bin(left_lidar_path)).float()
        
        # Transpose to get shape [3, num_points] for model input
        front_lidar = front_lidar.transpose(0, 1)
        right_lidar = right_lidar.transpose(0, 1)
        left_lidar = left_lidar.transpose(0, 1)
        
        return {
            'front_img': front_img,
            'right_img': right_img,
            'left_img': left_img,
            'front_lidar': front_lidar, 
            'right_lidar': right_lidar,
            'left_lidar': left_lidar,
            'sample_id': sample_id,
            'bag_dir': bag_dir
        }

def create_dataloader(data_root, batch_size=1, num_workers=2):
    """Create dataloader for visualization"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = NoAnnotationDataset(data_root, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

def visualize_predictions(model_path, data_root, output_dir=None, num_samples=None, device='cuda'):
    """
    Visualize model predictions on new data without annotations
    
    Args:
        model_path: Path to saved model weights
        data_root: Path to dataset directory
        output_dir: Directory to save visualizations (defaults to 'visualizations' in model dir)
        num_samples: Number of samples to visualize (None = all)
        device: Device to run inference on
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(model_path), "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}")
    
    # Load model
    try:
        model = DeadEndDetectionModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create dataloader
    dataloader = create_dataloader(data_root, batch_size=1)
    
    # Set number of samples to visualize
    if num_samples is None or num_samples > len(dataloader):
        num_samples = len(dataloader)
    
    # Run inference and create visualizations
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            # Get data
            front_img = batch['front_img'].to(device)
            right_img = batch['right_img'].to(device)
            left_img = batch['left_img'].to(device)
            front_lidar = batch['front_lidar'].to(device)
            right_lidar = batch['right_lidar'].to(device)
            left_lidar = batch['left_lidar'].to(device)
            sample_id = batch['sample_id'][0]
            
            # Forward pass
            outputs = model(front_img, right_img, left_img, front_lidar, right_lidar, left_lidar)
            
            # Get predictions
            pred_path_status = (outputs['path_status'] > 0.5).float()
            pred_dead_end = (outputs['is_dead_end'] > 0.5).float()
            pred_directions = outputs['direction_vectors']
            pred_confidence = outputs['confidence_scores']
            
            # Convert images for visualization
            denorm = lambda x: x * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device) + \
                              torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
            
            front_img_np = denorm(front_img[0]).cpu().permute(1, 2, 0).numpy()
            right_img_np = denorm(right_img[0]).cpu().permute(1, 2, 0).numpy()
            left_img_np = denorm(left_img[0]).cpu().permute(1, 2, 0).numpy()
            
            # Create figure for visualization
            fig, axs = plt.subplots(3, 3, figsize=(15, 12))
            
            # Display images
            axs[0, 0].imshow(front_img_np)
            axs[0, 0].set_title("Front Camera")
            axs[0, 1].imshow(left_img_np)
            axs[0, 1].set_title("Left Camera")
            axs[0, 2].imshow(right_img_np)
            axs[0, 2].set_title("Right Camera")
            
            # Display lidar points (top view)
            front_lidar_np = front_lidar.cpu().numpy()[0].T  # Shape: [num_points, 3]
            right_lidar_np = right_lidar.cpu().numpy()[0].T
            left_lidar_np = left_lidar.cpu().numpy()[0].T
            
            axs[1, 0].scatter(front_lidar_np[:, 0], front_lidar_np[:, 2], s=1, c='blue', alpha=0.5)
            axs[1, 0].set_title("Front LiDAR")
            axs[1, 0].set_xlim([-10, 10])
            axs[1, 0].set_ylim([0, 20])
            
            axs[1, 1].scatter(left_lidar_np[:, 0], left_lidar_np[:, 2], s=1, c='green', alpha=0.5)
            axs[1, 1].set_title("Left LiDAR")
            axs[1, 1].set_xlim([-10, 10])
            axs[1, 1].set_ylim([0, 20])
            
            axs[1, 2].scatter(right_lidar_np[:, 0], right_lidar_np[:, 2], s=1, c='red', alpha=0.5)
            axs[1, 2].set_title("Right LiDAR")
            axs[1, 2].set_xlim([-10, 10])
            axs[1, 2].set_ylim([0, 20])
            
            # Display predictions
            path_labels = ['Front', 'Left', 'Right']
            pred_status_np = pred_path_status[0].cpu().numpy()
            
            # Bar chart for path status
            axs[2, 0].bar(path_labels, pred_status_np)
            axs[2, 0].set_title("Path Status Predictions (1=Open, 0=Blocked)")
            axs[2, 0].set_ylim([0, 1.1])
            
            # Show dead end prediction
            is_dead_end_pred = pred_dead_end[0].item()
            axs[2, 1].bar(['Dead End'], [is_dead_end_pred])
            axs[2, 1].set_title(f"Dead End Prediction: {is_dead_end_pred:.2f}")
            axs[2, 1].set_ylim([0, 1.1])
            
            # Show confidence scores
            pred_conf_np = pred_confidence[0].cpu().numpy()
            axs[2, 2].bar(path_labels, pred_conf_np)
            axs[2, 2].set_title("Confidence Scores")
            axs[2, 2].set_ylim([0, 1.1])
            
            # Add direction vectors to the LiDAR plots
            # Only draw vectors for paths predicted as open
            for j, (ax, is_open, direction) in enumerate(zip(
                [axs[1, 0], axs[1, 1], axs[1, 2]],
                pred_status_np,
                pred_directions[0].cpu().numpy()
            )):
                if is_open > 0.5:  # If path is predicted as open
                    # Normalize for visualization
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm * 10  # Scale to make it visible
                    
                    # Draw arrow for direction
                    ax.arrow(0, 0, direction[0], direction[2], 
                             head_width=0.8, head_length=1.0, fc='yellow', ec='black', width=0.2)
                    
                    ax.text(direction[0]/2, direction[2]/2, 
                            f"Conf: {pred_conf_np[j]:.2f}", 
                            color='black', fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.7))
            
            # Add overall info to figure
            is_dead_end_text = "Dead End" if is_dead_end_pred > 0.5 else "Not Dead End"
            fig.suptitle(f"Sample: {sample_id} - Prediction: {is_dead_end_text}", fontsize=16)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sample_{sample_id}.png"))
            plt.close()
            
            print(f"Processed sample {i+1}/{num_samples}: {sample_id}")
    
    print(f"Visualization completed. {num_samples} samples processed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize model predictions on new data")
    parser.add_argument("--model_path", type=str, default='/home/vicky/IROS2025/DRaM/codes/data/saved_models/model_final.pth', help="Path to saved model weights")
    parser.add_argument("--data_root", type=str, default='/home/vicky/IROS2025/test_bags/bag3', help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default='/home/vicky/IROS2025/DRaM/codes/data/predictions/bag3', help="Directory to save visualizations")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to visualize")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda/cpu)")
    
    args = parser.parse_args()
    
    visualize_predictions(
        model_path=args.model_path,
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device
    ) 