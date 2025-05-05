import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# import cv2
import os
# import open3d
from concurrent.futures import ThreadPoolExecutor
import time
# from model.cross_multi import CrossModalAttention
import matplotlib.pyplot as plt
from numba import cuda
# import cupy as cp
import traceback
# import re
import shutil

@cuda.jit
def process_points_cuda(points, ground_mask, front_mask, min_distances, path_width):
    """CUDA kernel for processing point cloud data"""
    idx = cuda.grid(1)
    if idx < points.shape[0]:
        # Extract coordinates (points are in vehicle frame)
        x = points[idx, 0]  # Forward
        y = points[idx, 1]  # Left
        z = points[idx, 2]  # Up
        
        # Ground point detection (points below vehicle height)
        if z < -0.5 and z > -2.0:  # Typical ground height range
            ground_mask[idx] = True
            
            # Update path width
            cuda.atomic.min(path_width, 1, y)  # Min y
            cuda.atomic.max(path_width, 0, y)  # Max y
        
        # Front point detection (obstacles at vehicle height)
        if z > -0.5 and z < 2.0 and x > 0.0 and x < 15.0 and abs(y) < 8.0:
            front_mask[idx] = True
            
            # Calculate distance to robot
            distance = cuda.libdevice.sqrt(x * x + y * y)
            cuda.atomic.min(min_distances, idx, distance)

# class CrossModalAttention(nn.Module):
#     """Cross-modal attention module for feature fusion"""
#     def __init__(self, dim, num_heads=8, dropout=0.1):
#         super().__init__()
#         self.num_heads = num_heads
#         self.dim = dim
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
        
#         # Linear projections for query, key, value
#         self.q_proj = nn.Linear(dim, dim)
#         self.k_proj = nn.Linear(dim, dim)
#         self.v_proj = nn.Linear(dim, dim)
        
#         # Output projection
#         self.out_proj = nn.Linear(dim, dim)
        
#         # Layer normalization
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
        
#         # Dropout
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x1, x2):
#         # Ensure both inputs have the same batch size
#         if x1.size(0) != x2.size(0):
#             if x1.size(0) < x2.size(0):
#                 x1 = x1.expand(x2.size(0), -1)
#             else:
#                 x2 = x2.expand(x1.size(0), -1)
        
#         # Project queries from x1, keys and values from x2
#         q = self.q_proj(x1).view(-1, self.num_heads, self.head_dim)
#         k = self.k_proj(x2).view(-1, self.num_heads, self.head_dim)
#         v = self.v_proj(x2).view(-1, self.num_heads, self.head_dim)
        
#         # Compute attention scores
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = F.softmax(attn, dim=-1)
#         attn = self.dropout(attn)
        
#         # Apply attention to values
#         out = (attn @ v).reshape(-1, self.dim)
#         out = self.out_proj(out)
        
#         # Add residual connection and normalize
#         out = self.norm1(x1 + out)
#         out = self.norm2(out + self.dropout(out))
        
#         return out

class EfficientPointEncoder(nn.Module):
    """Efficient point cloud encoder using 1D convolutions"""
    def __init__(self, output_dim=128):
        super().__init__()
        
        # 1D convolution layers for point processing
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)  # Changed input channels to 3
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Global max pooling and MLP for feature aggregation
        self.fc1 = nn.Linear(256, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    @torch.cuda.amp.autocast()
    def forward(self, x):
        """Forward pass with automatic mixed precision"""
        # Ensure input is in the correct format (B, C, N)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        if x.shape[1] != 3:
            x = x.transpose(1, 2)  # Transpose to (B, C, N) format
        
        # Handle single sample case
        is_single_sample = x.size(0) == 1
        if is_single_sample:
            self.eval()  # Set to eval mode to use running statistics
        
        # Point feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global feature aggregation
        x = torch.max(x, 2)[0]  # Global max pooling
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        if is_single_sample:
            self.train()  # Set back to train mode
        
        return x

class CLIPVisionEncoder(nn.Module):
    """Vision encoder using CLIP model for features and zero-shot classification."""
    def __init__(self, output_dim=128, model_id="openai/clip-vit-base-patch32"):
        super().__init__()
        self.output_dim = output_dim
        self.model_id = model_id
        print(f"Initializing CLIPVisionEncoder with model: {self.model_id}")
        
        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(self.model_id)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        
        # Instead of relying on config, get the embedding dimension dynamically
        with torch.no_grad():
            # Create a dummy input to get the actual output size
            dummy_pixel_values = torch.zeros(1, 3, 224, 224)
            dummy_outputs = self.model.get_image_features(pixel_values=dummy_pixel_values)
            actual_dim = dummy_outputs.shape[-1]
        
        print(f"CLIP actual output dimension: {actual_dim}")
        # Define projector to map CLIP features to output dimension using actual size
        self.projector = nn.Linear(actual_dim, output_dim)
        
        # Define path descriptions for zero-shot classification
        self.path_descriptions = [
            "An open path ahead that is clear for travel",
            "A blocked path ahead with obstacles"
        ]

    @torch.cuda.amp.autocast(enabled=False)
    def get_features(self, image):
        """Extracts image features using CLIP's vision encoder."""
        # Process image using CLIP processor
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            # Extract features from vision encoder
            outputs = self.model.get_image_features(**inputs)
            
            # Project features to output dimension
            projected_features = self.projector(outputs)
        
        return projected_features

    @torch.cuda.amp.autocast(enabled=False)
    def analyze_path(self, image):
        """Performs zero-shot classification for path analysis using CLIP."""
        # Process image using CLIP processor
        image_inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        
        # Process text descriptions
        text_inputs = self.processor(text=self.path_descriptions, padding=True, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            # Get image features
            image_features = self.model.get_image_features(**image_inputs)
            # Get text features
            text_features = self.model.get_text_features(**text_inputs)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity scores
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Extract probabilities
        prob_open = similarity[0, 0].item()
        prob_blocked = similarity[0, 1].item()
        
        # Determine path status and confidence
        is_open = prob_open > prob_blocked
        confidence = max(prob_open, prob_blocked)
        
        # Create response text
        response_text = f"{'Open' if is_open else 'Blocked'} path. (Confidence: {confidence:.2f})"
        
        return response_text, is_open, confidence

class MultiCameraLiDARFusion(nn.Module):
    """Fusion model using CLIP and CUDA-accelerated point processing"""
    def __init__(self, num_cameras=3, feature_dim=128):
        super().__init__()
        
        # Use CLIP Vision Encoders
        self.camera_encoders = nn.ModuleList([
            CLIPVisionEncoder(output_dim=feature_dim) 
            for _ in range(num_cameras)
        ])
        
        # LiDAR encoder
        self.lidar_encoder = EfficientPointEncoder(output_dim=feature_dim)
        
        # Cross-modal attention
        # self.cross_attention = CrossModalAttention(feature_dim)
        
        # Final classifier
        self.classifier = nn.Linear(feature_dim, 2)

    def analyze_lidar_geometry(self, lidar_points):
        """CUDA-accelerated LiDAR point cloud analysis"""
        # Convert to numpy for CUDA processing
        points = lidar_points.detach().cpu().numpy()
        batch_results = []
        
        # Process each batch
        for batch_idx in range(points.shape[0]):
            batch_points = points[batch_idx]
            
            # Allocate CUDA memory
            d_points = cuda.to_device(batch_points)
            d_ground_mask = cuda.to_device(np.zeros(len(batch_points), dtype=np.bool_))
            d_front_mask = cuda.to_device(np.zeros(len(batch_points), dtype=np.bool_))
            d_min_distances = cuda.to_device(np.full(len(batch_points), np.inf, dtype=np.float32))
            d_path_width = cuda.to_device(np.array([float('-inf'), float('inf')], dtype=np.float32))
            
            # Configure CUDA kernel
            threadsperblock = 256
            blockspergrid = (batch_points.shape[0] + (threadsperblock - 1)) // threadsperblock
            
            # Launch CUDA kernel
            process_points_cuda[blockspergrid, threadsperblock](
                d_points, d_ground_mask, d_front_mask, d_min_distances, d_path_width
            )
            
            # Copy results back to host
            ground_mask = d_ground_mask.copy_to_host()
            front_mask = d_front_mask.copy_to_host()
            min_distances = d_min_distances.copy_to_host()
            path_width_result = d_path_width.copy_to_host()
            
            # Process results
            ground_points = batch_points[ground_mask]
            front_points = batch_points[front_mask]
            valid_distances = min_distances[front_mask]
            
            # Analysis (handle potential empty arrays)
            is_navigable = len(front_points) < 500 # Example threshold
            avg_min_distance = np.mean(valid_distances) if len(valid_distances) > 0 else float('inf')
            path_width = path_width_result[0] - path_width_result[1] if np.isfinite(path_width_result).all() else 0.0
            
            batch_results.append({
                'is_navigable': is_navigable,
                'avg_min_distance': avg_min_distance,
                'path_width': path_width,
                'num_ground_points': len(ground_points),
                'num_front_points': len(front_points)
            })
            
        return batch_results

    @torch.cuda.amp.autocast()
    def forward(self, images, lidar_points):
        """Forward pass using CLIP features and zero-shot classification."""
        
        # 1. Process camera images for features
        camera_features_list = []
        # Only use the first camera for CLIP analysis for now to simplify
        with torch.cuda.amp.autocast(enabled=False):
            # Assuming batch size is 1 for images input typically
            front_image = images[0] 
            # Use the first encoder instance to get features
            camera_features = self.camera_encoders[0].get_features(front_image) 
            # For simplicity now, let's just use the front camera features for fusion

        # 2. Process LiDAR points
        lidar_features = self.lidar_encoder(lidar_points)
        
        # Ensure features are on the same device and dtype for attention
        device = lidar_features.device
        camera_features = camera_features.to(device, dtype=torch.float16)
        lidar_features = lidar_features.to(device, dtype=torch.float16)

        # 3. Apply cross-modal attention
        # Ensure dimensions match (Batch size might differ if lidar batch > 1)
        if camera_features.shape[0] != lidar_features.shape[0]:
             camera_features = camera_features.expand(lidar_features.shape[0], -1)
        
        # fused_features = self.cross_attention(camera_features, lidar_features)
        
        # 4. Classify based on fused features
        # logits = self.classifier(fused_features.to(torch.float32)).to(torch.float16)

        # 5. CLIP Path Analysis (using the front camera image)
        with torch.cuda.amp.autocast(enabled=False):
            clip_response, clip_is_open, clip_confidence = self.camera_encoders[0].analyze_path(front_image)

        # 6. Geometric analysis from LiDAR
        geometric_analysis = self.analyze_lidar_geometry(lidar_points)
        lidar_score = 1.0 if geometric_analysis[0]['is_navigable'] else 0.0
        
        # 7. Combine results
        # Using CLIP's analysis directly
        final_is_open = clip_is_open
        final_confidence = clip_confidence
        
        # 8. Return results
        return {
            # "logits": logits,
            # "fused_features": fused_features,
            "geometric_analysis": geometric_analysis,
            "is_open": final_is_open,
            "confidence": final_confidence,
            "clip_response": clip_response,
            "lidar_score": lidar_score
        }

def load_image(image_path_or_array):
    """Load image from path or use provided array with error handling."""
    try:
        if isinstance(image_path_or_array, str):
            if not os.path.exists(image_path_or_array):
                print(f"[Error] Image file not found: {image_path_or_array}")
                return None
            img = Image.open(image_path_or_array).convert("RGB")
            print(f"[Success] Loaded image from path: {image_path_or_array}")
            return img
        elif isinstance(image_path_or_array, np.ndarray):
            img = Image.fromarray(image_path_or_array).convert("RGB")
            print(f"[Success] Loaded image from numpy array.")
            return img
        elif isinstance(image_path_or_array, Image.Image):
            img = image_path_or_array.convert("RGB")
            print(f"[Success] Used provided PIL Image.")
            return img
        else:
            print(f"[Error] Invalid input type for image loading: {type(image_path_or_array)}")
            return None
    except Exception as e:
        print(f"[Error] Failed to load or process image ({image_path_or_array}): {e}")
        return None

def load_lidar_data(lidar_path_or_array, max_points=20000):
    """
    Load and preprocess LiDAR point cloud data
    
    Args:
        lidar_path_or_array: Can be either:
                            - String path to LiDAR file (.bin, .pcd, .npy)
                            - Numpy array of points [N, 4] (x, y, z, intensity)
        max_points: Maximum number of points to keep
    
    Returns:
        Tensor of shape [1, max_points, 4]
    """
    # Case 1: String path
    if isinstance(lidar_path_or_array, str):
        if not os.path.exists(lidar_path_or_array):
            raise FileNotFoundError(f"LiDAR file not found: {lidar_path_or_array}")
        
        # Load based on file extension
        ext = os.path.splitext(lidar_path_or_array)[1].lower()
        
        if ext == '.bin':  # KITTI format binary
            lidar_data = np.fromfile(lidar_path_or_array, dtype=np.float32)
            lidar_data = lidar_data.reshape(-1, 4)  # x, y, z, intensity
        
        elif ext == '.npy':  # Numpy format
            lidar_data = np.load(lidar_path_or_array)
        
        elif ext == '.pcd':  # Point Cloud Data format
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(lidar_path_or_array)
                points = np.asarray(pcd.points)
                
                # Add intensity if not present (default to 1.0)
                if points.shape[1] == 3:
                    intensities = np.ones((points.shape[0], 1))
                    lidar_data = np.hstack([points, intensities])
                else:
                    lidar_data = points
            except ImportError:
                raise ImportError("Please install open3d: pip install open3d")
        
        else:
            raise ValueError(f"Unsupported LiDAR file format: {ext}")
    
    # Case 2: Already a numpy array
    elif isinstance(lidar_path_or_array, np.ndarray):
        lidar_data = lidar_path_or_array
    
    # Invalid input
    else:
        raise TypeError("Input must be a LiDAR file path or numpy array")
    
    # Ensure data has correct shape
    if lidar_data.ndim != 2 or lidar_data.shape[1] < 3:
        raise ValueError("LiDAR data must have shape [N, 4] or [N, 3+]")
    
    # Ensure we have at least 4 columns (x, y, z, intensity)
    if lidar_data.shape[1] == 3:
        # Add default intensity value
        intensities = np.ones((lidar_data.shape[0], 1))
        lidar_data = np.hstack([lidar_data, intensities])
    elif lidar_data.shape[1] > 4:
        # Keep only the first 4 columns
        lidar_data = lidar_data[:, :4]
    
    # Normalize point cloud (center and scale)
    if lidar_data.shape[0] > 0:
        # Center the point cloud
        centroid = np.mean(lidar_data[:, :3], axis=0)
        lidar_data[:, :3] = lidar_data[:, :3] - centroid
        
        # Scale to unit sphere
        scale = np.max(np.sqrt(np.sum(lidar_data[:, :3]**2, axis=1)))
        if scale > 0:
            lidar_data[:, :3] = lidar_data[:, :3] / scale
    
    # Sample or pad to max_points
    if lidar_data.shape[0] > max_points:
        # Random sampling without replacement
        indices = np.random.choice(lidar_data.shape[0], max_points, replace=False)
        lidar_data = lidar_data[indices]
    elif lidar_data.shape[0] < max_points:
        # Pad with zeros
        padding = np.zeros((max_points - lidar_data.shape[0], 4))
        lidar_data = np.vstack([lidar_data, padding])
    
    # Convert to tensor and add batch dimension
    lidar_tensor = torch.tensor(lidar_data, dtype=torch.float32).unsqueeze(0)
    return lidar_tensor

# def detect_path_navigability(model, images, lidar_data, device):
#     """Detect path navigability from camera images and LiDAR data"""
#     model.eval()
#     with torch.no_grad():
#         # Forward pass
#         outputs = model(images, lidar_data)
        
#         # Get combined analysis results
#         results = outputs['combined_analysis']
        
#         # Print timing information
#         timing = outputs['timing']
#         print("\nTiming Information:")
#         for key, value in timing.items():
#             print(f"{key}: {value:.3f} seconds")
        
#         return results

def load_binary_lidar(file_path, num_features=4):
    """Load LiDAR data from a binary file (adjust based on actual format)"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"LiDAR file not found: {file_path}")
        
    # Assuming binary file contains float32 points (x, y, z, intensity)
    point_cloud = np.fromfile(file_path, dtype=np.float32)
    
    # Try different reshapes to find the right format
    try:
        # Try to infer the number of features
        total_elements = point_cloud.shape[0]
        
        # Check if divisible by 4 (x, y, z, intensity)
        if total_elements % 4 == 0:
            point_cloud = point_cloud.reshape(-1, 4)
            # print(f"Reshaped LiDAR data as [N, 4] (x, y, z, intensity): {point_cloud.shape}")
        # Check if divisible by 3 (x, y, z)
        elif total_elements % 3 == 0:
            point_cloud = point_cloud.reshape(-1, 3)
            # print(f"Reshaped LiDAR data as [N, 3] (x, y, z): {point_cloud.shape}")
        # Check if divisible by 5 (x, y, z, intensity, ring)
        elif total_elements % 5 == 0:
            point_cloud = point_cloud.reshape(-1, 5)
            # print(f"Reshaped LiDAR data as [N, 5] (x, y, z, intensity, ring): {point_cloud.shape}")
        else:
            # Try the specified number of features as a fallback
            point_cloud = point_cloud.reshape(-1, num_features)
            # print(f"Reshaped LiDAR data as [N, {num_features}]: {point_cloud.shape}")
            
    except ValueError as e:
        # If reshaping fails, try to get the size divisors to help diagnose
        divisors = [i for i in range(2, 10) if total_elements % i == 0]
        print(f"Could not reshape LiDAR data. Total elements: {total_elements}. Possible divisors: {divisors}")
        raise ValueError(f"Error reshaping LiDAR data from {file_path}. Total elements: {total_elements}. Error: {e}")
        
    # Return only x, y, z coordinates for the encoder
    return point_cloud[:, :3] 

# def visualize_features(features, save_path, title="Feature Visualization"):
#     """Visualize feature embeddings as a 2D heatmap"""
#     # Convert features to numpy array
#     if torch.is_tensor(features):
#         features = features.detach().cpu().numpy()
    
#     # Reshape if needed
#     if len(features.shape) == 3:
#         features = features.squeeze(0)  # Remove batch dimension
    
#     # Create heatmap
#     plt.figure(figsize=(10, 6))
#     plt.imshow(features, aspect='auto', cmap='viridis')
#     plt.colorbar(label='Feature Value')
#     plt.title(title)
#     plt.xlabel('Feature Dimension')
#     plt.ylabel('Sample')
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

def create_lidar_occupancy_map(points, resolution=0.1, size=20, colorize_by_height=True):
    """
    Create 2D occupancy map from LiDAR points with enhanced visualization
    
    Args:
        points: Tensor or numpy array of points (x, y, z, ...)
        resolution: Grid resolution in meters per pixel
        size: Size of the map in meters (half-width)
        colorize_by_height: Whether to color points by height
        
    Returns:
        occupancy_map: 2D occupancy map as numpy array
        colored_map: (optional) 3-channel colored occupancy map if colorize_by_height=True
    """
    # Convert points to numpy if needed
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    if len(points.shape) == 3:
        points = points.squeeze(0)
    
    # Create grid
    grid_size = int(size * 2 / resolution)
    occupancy_map = np.zeros((grid_size, grid_size))
    
    # Create colored map if requested
    if colorize_by_height:
        colored_map = np.zeros((grid_size, grid_size, 3))
    
    # Make sure we have points
    if points.shape[0] == 0:
        if colorize_by_height:
            return occupancy_map, colored_map
        else:
            return occupancy_map
    
    # Convert points to grid coordinates
    x_idx = ((points[:, 0] + size) / resolution).astype(int)
    y_idx = ((points[:, 1] + size) / resolution).astype(int)
    
    # Filter valid indices
    valid = (x_idx >= 0) & (x_idx < grid_size) & (y_idx >= 0) & (y_idx < grid_size)
    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    
    # If we have z coordinates, use them for coloring
    if colorize_by_height and points.shape[1] > 2:
        z_values = points[valid, 2]
        
        # Normalize z values for coloring
        z_min, z_max = z_values.min(), z_values.max()
        if z_max > z_min:
            z_normalized = (z_values - z_min) / (z_max - z_min)
        else:
            z_normalized = np.zeros_like(z_values)
        
        # Create colored map using height values
        for i, (x, y, z_norm) in enumerate(zip(x_idx, y_idx, z_normalized)):
            occupancy_map[y, x] = 1
            
            # Use a color map: blue for low points, red for high points
            colored_map[y, x, 0] = z_norm  # Red channel (high points)
            colored_map[y, x, 2] = 1 - z_norm  # Blue channel (low points)
    else:
        # Just create binary occupancy map
        occupancy_map[y_idx, x_idx] = 1
    
    # Return the appropriate maps
    if colorize_by_height:
        return occupancy_map, colored_map
    else:
        return occupancy_map

# def create_feature_occupancy_map(features, resolution=0.1, size=20):
#     """Create 2D occupancy map from feature embeddings"""
#     # Convert features to numpy if needed
#     if torch.is_tensor(features):
#         features = features.detach().cpu().numpy()
#     if len(features.shape) == 3:
#         features = features.squeeze(0)
    
#     # Calculate feature magnitude as occupancy value
#     feature_magnitude = np.linalg.norm(features, axis=1)
    
#     # Normalize feature magnitudes to [0, 1]
#     feature_magnitude = (feature_magnitude - feature_magnitude.min()) / (feature_magnitude.max() - feature_magnitude.min())
    
#     # Create grid
#     grid_size = int(size * 2 / resolution)
#     occupancy_map = np.zeros((grid_size, grid_size))
    
#     # Project features onto 2D grid using their magnitudes
#     x_coords = np.linspace(-size, size, grid_size)
#     y_coords = np.linspace(-size, size, grid_size)
#     X, Y = np.meshgrid(x_coords, y_coords)
    
#     # Create Gaussian mixture model from features
#     for i, magnitude in enumerate(feature_magnitude):
#         # Create Gaussian centered at grid points
#         gaussian = magnitude * np.exp(-((X)**2 + (Y)**2) / (2 * (resolution * 5)**2))
#         occupancy_map += gaussian
    
#     # Normalize final map
#     occupancy_map = (occupancy_map - occupancy_map.min()) / (occupancy_map.max() - occupancy_map.min())
    
#     return occupancy_map

def visualize_occupancy_maps(lidar_map, feature_map, save_path):
    """Visualize LiDAR and feature occupancy maps side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot LiDAR occupancy map
    im1 = ax1.imshow(lidar_map, cmap='viridis', origin='lower')
    ax1.set_title('LiDAR Occupancy Map')
    plt.colorbar(im1, ax=ax1, label='Occupancy')
    
    # Plot feature occupancy map
    im2 = ax2.imshow(feature_map, cmap='plasma', origin='lower')
    ax2.set_title('Feature Occupancy Map')
    plt.colorbar(im2, ax=ax2, label='Feature Magnitude')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def filter_lidar_by_camera_fov(points, camera_fov_params):
    """
    Filter LiDAR points to match the field of view of a specific camera
    
    Args:
        points: Numpy array of shape [N, 3+] containing point cloud data (x, y, z, ...)
        camera_fov_params: Dictionary with camera FOV parameters:
            - 'horizontal_fov': horizontal field of view in degrees
            - 'vertical_fov': vertical field of view in degrees (optional)
            - 'orientation': camera orientation in degrees (0 = front, 90 = left, -90 = right)
            - 'position': camera position offset from LiDAR [x, y, z] (optional)
    
    Returns:
        Filtered points within the camera's FOV
    """
    # Extract camera parameters
    h_fov = camera_fov_params.get('horizontal_fov', 180)
    v_fov = camera_fov_params.get('vertical_fov', 90)
    orientation = camera_fov_params.get('orientation', 0)  # 0 = front, 90 = left, -90 = right
    position = camera_fov_params.get('position', [0, 0, 0])  # Camera position relative to LiDAR
    
    # Adjust points for camera position if needed
    adjusted_points = points.copy()
    adjusted_points[:, :3] -= np.array(position)
    
    # Calculate angles for each point
    x, y, z = adjusted_points[:, 0], adjusted_points[:, 1], adjusted_points[:, 2]
    
    # Calculate horizontal angle (azimuth) in degrees
    # atan2 gives angle in range [-π, π] where 0 is forward
    horizontal_angles = np.degrees(np.arctan2(y, x))
    
    # Adjust angles based on camera orientation
    adjusted_angles = horizontal_angles - orientation
    
    # Normalize angles to range [-180, 180]
    adjusted_angles = np.mod(adjusted_angles + 180, 360) - 180
    
    # Calculate vertical angle (elevation) in degrees if needed
    if v_fov < 180:
        # Distance in horizontal plane
        distance_xy = np.sqrt(x**2 + y**2)
        # Elevation angle in degrees (0 = horizontal, 90 = up, -90 = down)
        vertical_angles = np.degrees(np.arctan2(z, distance_xy))
    
    # Create masks for horizontal and vertical FOV
    h_mask = np.abs(adjusted_angles) <= (h_fov / 2)
    
    # Add vertical mask if needed
    if v_fov < 180:
        v_mask = np.abs(vertical_angles) <= (v_fov / 2)
        combined_mask = h_mask & v_mask
    else:
        combined_mask = h_mask
    
    # Return filtered points
    return points[combined_mask]

def split_lidar_by_cameras(lidar_points):
    """
    Split LiDAR point cloud into segments corresponding to different camera FOVs
    
    Args:
        lidar_points: Numpy array of shape [N, 3+] containing point cloud data
        
    Returns:
        Dictionary with filtered point clouds for each camera view
    """
    # Define camera FOV parameters
    camera_fovs = {
        'front': {
            'horizontal_fov': 180,
            'vertical_fov': 90,
            'orientation': 0,  # Front-facing
        },
        'side_left': {
            'horizontal_fov': 90,
            'vertical_fov': 90,
            'orientation': 90,  # Left-facing
        },
        'side_right': {
            'horizontal_fov': 90,
            'vertical_fov': 90,
            'orientation': -90,  # Right-facing
        }
    }
    
    # Filter points for each camera
    filtered_points = {}
    for camera_name, fov_params in camera_fovs.items():
        filtered_points[camera_name] = filter_lidar_by_camera_fov(lidar_points, fov_params)
        
    return filtered_points

def load_and_split_lidar(lidar_path, max_points_per_segment=20000):
    """
    Load LiDAR data and split it into camera FOV segments
    
    Args:
        lidar_path: Path to the LiDAR data file
        max_points_per_segment: Maximum number of points to keep per segment
        
    Returns:
        Dictionary with filtered point cloud tensors for each camera view
    """
    # Load the full point cloud
    if isinstance(lidar_path, str):
        try:
            # Try to determine the file format and load accordingly
            lidar_points = load_binary_lidar(lidar_path)
            if lidar_points is None:
                print(f"Failed to load LiDAR data from {lidar_path}")
                # Return empty dictionary as fallback
                return {'front': torch.zeros((max_points_per_segment, 3)),
                        'side_left': torch.zeros((max_points_per_segment, 3)),
                        'side_right': torch.zeros((max_points_per_segment, 3))}
        except Exception as e:
            print(f"Error loading LiDAR data: {e}")
            # Return empty dictionary as fallback
            return {'front': torch.zeros((max_points_per_segment, 3)),
                    'side_left': torch.zeros((max_points_per_segment, 3)),
                    'side_right': torch.zeros((max_points_per_segment, 3))}
    else:
        # Assume it's already a numpy array
        lidar_points = lidar_path
        
    # Split the point cloud into camera FOV segments
    split_points = split_lidar_by_cameras(lidar_points)
    
    # Convert to tensors and sample/pad if needed
    lidar_tensors = {}
    for camera_name, points in split_points.items():
        # Sample or pad to max_points
        if points.shape[0] > max_points_per_segment:
            # Random sampling without replacement
            indices = np.random.choice(points.shape[0], max_points_per_segment, replace=False)
            points = points[indices]
        elif points.shape[0] < max_points_per_segment and points.shape[0] > 0:
            # Pad with zeros
            padding = np.zeros((max_points_per_segment - points.shape[0], points.shape[1]))
            points = np.vstack([points, padding])
        elif points.shape[0] == 0:
            # Empty point cloud, create dummy data
            points = np.zeros((max_points_per_segment, lidar_points.shape[1]))
        
        # Convert to tensor
        lidar_tensors[camera_name] = torch.tensor(points, dtype=torch.float32)
    
    return lidar_tensors

def save_occupancy_maps(lidar_segments, output_dir, resolution=0.1, size=20):
    """
    Save detailed occupancy maps for each camera LiDAR segment
    
    Args:
        lidar_segments: Dictionary of LiDAR point clouds for each camera
        output_dir: Directory to save the maps
        resolution: Grid resolution in meters per pixel
        size: Size of the map in meters (half-width)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each camera segment
    for camera_name, points in lidar_segments.items():
        # Convert to numpy if needed
        if torch.is_tensor(points):
            points = points.detach().cpu().numpy()
        
        # Create both standard and colored occupancy maps
        binary_map, colored_map = create_lidar_occupancy_map(
            points, resolution=resolution, size=size, colorize_by_height=True
        )
        
        # Save binary occupancy map
        plt.figure(figsize=(10, 10))
        plt.imshow(binary_map, cmap='viridis', origin='lower')
        plt.title(f'LiDAR Points - {camera_name} FOV')
        plt.colorbar(label='Occupancy')
        plt.savefig(f'{output_dir}/{camera_name}_binary_map.png')
        plt.close()
        
        # Save height-colored occupancy map
        plt.figure(figsize=(10, 10))
        plt.imshow(colored_map, origin='lower')
        plt.title(f'LiDAR Points (Height Colored) - {camera_name} FOV')
        plt.savefig(f'{output_dir}/{camera_name}_height_map.png')
        plt.close()
        
        # Save top-view with points
        plt.figure(figsize=(12, 12))
        plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=1, cmap='jet')
        plt.colorbar(label='Height (m)')
        plt.xlim(-size, size)
        plt.ylim(-size, size)
        plt.grid(True)
        plt.title(f'Top View - {camera_name} LiDAR Points')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.savefig(f'{output_dir}/{camera_name}_top_view.png')
        plt.close()
        
        # Save a 3D visualization if we have enough points
        if points.shape[0] > 100:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                         c=points[:, 2], s=1, cmap='jet')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'3D Point Cloud - {camera_name} FOV')
            plt.colorbar(scatter, label='Height (m)')
            plt.savefig(f'{output_dir}/{camera_name}_3d_view.png')
            plt.close()
            
        print(f"Saved occupancy maps for camera {camera_name} with {points.shape[0]} points")

def analyze_lidar_navigability(points, view='front'):
    if points is None or len(points) == 0:
        return "Unknown", 0.5
    
    # Convert to numpy if tensor
    if hasattr(points, 'numpy'):
        points = points.numpy()
    
    # Remove batch dimension if present
    if len(points.shape) == 3:
        points = points[0]
    
    # Define view-specific parameters
    if view == 'front':
        x_range = (-2, 2)  # Wider range for front view
        y_range = (0, 4)   # Look up to 4m ahead
        height_range = (-0.5, 1.5)  # Consider points from ground to above robot height
    else:  # side views
        x_range = (-1, 1)
        y_range = (-2, 2)
        height_range = (-0.5, 1.5)
    
    # Filter points within the region of interest
    mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) & \
           (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) & \
           (points[:, 2] >= height_range[0]) & (points[:, 2] <= height_range[1])
    
    filtered_points = points[mask]
    
    if len(filtered_points) == 0:
        return "Open", 0.8  # High confidence if no obstacles in region
    
    # Calculate metrics
    volume = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0]) * (height_range[1] - height_range[0])
    point_density = len(filtered_points) / volume
    
    # Calculate path width (gap analysis)
    if view == 'front':
        sorted_x = np.sort(filtered_points[:, 0])
        gaps = sorted_x[1:] - sorted_x[:-1]
        if len(gaps) > 0:
            max_gap = np.max(gaps)
        else:
            max_gap = x_range[1] - x_range[0]
    else:
        max_gap = 2.0  # Default for side views
    
    # Define thresholds
    density_threshold = 30.0  # points/m³
    width_threshold = 0.8    # meters
    
    # Calculate confidence scores
    density_score = min(1.0, point_density / density_threshold)
    width_score = 1.0 - min(1.0, max_gap / width_threshold)
    
    # Combine scores with weights
    confidence = 0.7 * density_score + 0.3 * width_score
    
    # Debug output
    # print(f"  LiDAR Debug ({view}):")
    # print(f"    Points in region: {len(filtered_points)}")
    # print(f"    Point density: {point_density:.2f} pts/m³")
    # print(f"    Path width: {max_gap:.2f}m")
    # print(f"    Density score: {density_score:.2f}")
    # print(f"    Width score: {width_score:.2f}")
    # print(f"    Confidence: {confidence:.2f}")
    
    if confidence > 0.6:
        return "Blocked", confidence
    elif confidence < 0.3:
        return "Open", 1.0 - confidence
    else:
        return "Unknown", 0.5

def combine_predictions(clip_predictions, lidar_predictions):
    """Combine CLIP and LiDAR predictions for all views."""
    # For CLIP: If any view predicts open, consider it open
    clip_is_open = any(pred[0] for pred in clip_predictions.values())
    # Use the highest confidence among all views
    clip_confidence = max(pred[1] for pred in clip_predictions.values())
    
    # For LiDAR: Only use actual measurements, no default values
    lidar_is_open = any(pred[0] for pred in lidar_predictions.values())
    # Use the highest confidence among all views
    lidar_confidence = max(pred[1] for pred in lidar_predictions.values())
    
    # If CLIP predicts open (from any view), consider it open
    if clip_is_open:
        return True, clip_confidence
        
    # If CLIP predicts blocked but LiDAR predicts open with high confidence
    if lidar_is_open and lidar_confidence > 0.5:
        return True, lidar_confidence
        
    # Otherwise consider it blocked
    return False, max(clip_confidence, lidar_confidence)

def process_from_files(data_dir):
    """Process images and LiDAR data from files"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")
    
    # Initialize CLIP model
    clip_model = CLIPVisionEncoder()
    clip_model.to(device)
    
    # Create output directories for classified images
    output_dir = os.path.join(data_dir, "classified_images")
    open_dir = os.path.join(output_dir, "open_paths")
    deadend_dir = os.path.join(output_dir, "dead_ends")
    os.makedirs(open_dir, exist_ok=True)
    os.makedirs(deadend_dir, exist_ok=True)
    
    # Get list of timestamps from front camera directory
    front_camera_dir = os.path.join(data_dir, 'camera_front')
    if not os.path.exists(front_camera_dir):
        print(f"Error: Front camera directory not found at {front_camera_dir}")
        return
        
    # Get all timestamps from the front camera directory
    timestamps = [os.path.splitext(f)[0] for f in os.listdir(front_camera_dir) if f.endswith('.jpg')]
    timestamps.sort()
    
    if not timestamps:
        print("No frames found in the front camera directory")
        return
        
    print(f"Found {len(timestamps)} frames to process")
    
    # Process each frame
    for timestamp in timestamps:
        print(f"\nProcessing frame with timestamp {timestamp}")
        try:
            # Initialize predictions for this frame
            clip_predictions = {}
            lidar_predictions = {}
            
            # Process each view
            for view in ['front', 'side_left', 'side_right']:
                # Load camera image
                img_path = os.path.join(data_dir, f'camera_{view}', f'{timestamp}.jpg')
                if not os.path.exists(img_path):
                    print(f"Warning: {view} image not found at {img_path}")
                    continue
                image = Image.open(img_path).convert("RGB")
                
                # Get CLIP analysis
                with torch.no_grad():
                    clip_response, is_open, confidence = clip_model.analyze_path(image)
                    clip_predictions[view] = (is_open, confidence)
                    print(f"CLIP Analysis ({view}): {clip_response}")
                
                # Load and analyze LiDAR data
                lidar_path = os.path.join(data_dir, f'lidar_{view}', f'{timestamp}.bin')
                if not os.path.exists(lidar_path):
                    print(f"Warning: {view} LiDAR data not found at {lidar_path}")
                    continue
                    
                try:
                    points = load_lidar_data(lidar_path)
                    if points is not None and points.shape[0] > 0:
                        is_navigable, confidence = analyze_lidar_navigability(points, view)
                        lidar_predictions[view] = (is_navigable, confidence)
                        print(f"LiDAR Analysis ({view}): {'Navigable' if is_navigable else 'Blocked'} (Confidence: {confidence:.2f})")
                    else:
                        print(f"Warning: No valid points found in LiDAR data for {view}")
                        # Don't add default values for LiDAR
                except Exception as e:
                    print(f"Error processing LiDAR data for {view}: {str(e)}")
                    # Don't add default values for LiDAR
            
            # Only combine predictions if we have at least one view with predictions
            if clip_predictions and lidar_predictions:
                is_navigable, confidence = combine_predictions(clip_predictions, lidar_predictions)
                print(f"\nFinal Prediction: {'Open' if is_navigable else 'Closed'} path (Confidence: {confidence:.2f})")
                
                # Save the front camera image in the appropriate folder
                front_img_path = os.path.join(data_dir, 'camera_front', f'{timestamp}.jpg')
                if os.path.exists(front_img_path):
                    # Create new filename with confidence score
                    new_filename = f"{timestamp}_conf_{confidence:.2f}.jpg"
                    
                    # Copy image to appropriate folder
                    if is_navigable:
                        output_path = os.path.join(open_dir, new_filename)
                    else:
                        output_path = os.path.join(deadend_dir, new_filename)
                        
                    # Open the image and add confidence text
                    img = Image.open(front_img_path)
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Create a copy of the image to draw on
                    img_with_text = img.copy()
                    draw = ImageDraw.Draw(img_with_text)
                    
                    # Get image dimensions
                    width, height = img.size
                    
                    # Add confidence text
                    text = f"Confidence: {confidence:.2f}"
                    # Use a larger font size
                    font_size = int(height * 0.05)  # 5% of image height
                    try:
                        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
                    
                    # Calculate text position (bottom right corner)
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    x = width - text_width - 20  # 20 pixels from right edge
                    y = height - text_height - 20  # 20 pixels from bottom
                    
                    # Draw white background for text
                    draw.rectangle([x-10, y-10, x+text_width+10, y+text_height+10], fill='white')
                    
                    # Draw text
                    draw.text((x, y), text, fill='black', font=font)
                    
                    # Save the image with text
                    img_with_text.save(output_path)
                    print(f"Saved image to: {output_path}")
            
        except Exception as e:
            print(f"Error processing frame at timestamp {timestamp}: {str(e)}")
            traceback.print_exc()
    
    print("\nFinished processing all frames")
    print(f"Images saved in: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="/home/vicky/IROS2025/processed_images", type=str)
    args = parser.parse_args()
    process_from_files(args.data_dir)