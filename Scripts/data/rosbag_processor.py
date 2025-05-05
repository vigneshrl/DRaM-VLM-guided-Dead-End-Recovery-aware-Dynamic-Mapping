#!/usr/bin/env python3

import os
import rclpy
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
import sensor_msgs.msg as sensor_msgs
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py import point_cloud2
import numpy as np
import torch
from PIL import Image as PILImage
from datetime import datetime
import argparse
import gc
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm

counter = 0

# Constants for optimization
CHUNK_SIZE = 100  # Process messages in chunks
MAX_WORKERS = multiprocessing.cpu_count() - 1  # Leave one core free
MEMORY_THRESHOLD = 0.9  # 90% memory usage threshold

def ensure_directory(directory_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def get_rosbag_options(path, storage_id, serialization_format='cdr'):
    """Get rosbag reader options."""
    storage_options = rosbag2_py.StorageOptions(
        uri=path,
        storage_id=storage_id
    )
    
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format
    )
    
    return storage_options, converter_options

def process_camera_messages(camera_messages, output_dir):
    """Process and save camera images with synchronized timestamps."""
    global counter
    
    # Create images directory
    images_dir = os.path.join(output_dir, "images")
    ensure_directory(images_dir)
    
    # Group messages by approximate timestamp
    timestamp_groups = {}
    time_threshold = 0.1  # 100ms threshold for grouping messages
    
    # First pass: group messages by timestamp
    for camera_type, messages in camera_messages.items():
        for timestamp_ns, msg in messages:
            # Convert to seconds for easier comparison
            timestamp_sec = timestamp_ns / 1e9
            
            # Find closest existing group or create new one
            matched_group = None
            for group_time in timestamp_groups:
                if abs(group_time - timestamp_sec) < time_threshold:
                    matched_group = group_time
                    break
            
            if matched_group is None:
                matched_group = timestamp_sec
                timestamp_groups[matched_group] = {}
            
            if camera_type not in timestamp_groups[matched_group]:
                timestamp_groups[matched_group][camera_type] = (timestamp_ns, msg)
    
    # Second pass: process grouped messages
    for group_time, group_messages in timestamp_groups.items():
        # Only process if we have messages from all cameras
        if len(group_messages) == len(camera_messages):
            # Create sample directory
            sample_dir = os.path.join(images_dir, f"sample_id_{counter}")
            ensure_directory(sample_dir)
            
            # Process each camera view
            for camera_type, (timestamp_ns, msg) in group_messages.items():
                try:
                    # Convert ROS Image message to numpy array
                    img_data = np.array(msg.data, dtype=np.uint8)
                    img_data = img_data.reshape((msg.height, msg.width, -1))
                    
                    # Convert from BGR to RGB
                    img_data = img_data[..., ::-1]
                    
                    # Convert to PIL Image
                    pil_image = PILImage.fromarray(img_data)
                    
                    # Save image with view name
                    image_path = os.path.join(sample_dir, f"{camera_type}.jpg")
                    pil_image.save(image_path, 'JPEG')
                    
                except Exception as e:
                    print(f"Error processing message at time {timestamp_ns}: {e}")
                    continue
            
            counter += 1
            
            # Print progress every 10 samples
            if counter % 10 == 0:
                print(f"Processed {counter} samples")

"""This function is used to filter the LIDAR points based on horizontal and vertical angle calculations.
The CAMERA FOV'S are defined as follows:
front: 180 x 90
side_left: 90 x 90
side_right: 90 x -90

Cloud filtering are done by calculating angles for each point , adjusts angles based on camera orientation and mask points based on horizontal and vertical FOV's
"""

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
            print(f"Reshaped LiDAR data as [N, 4] (x, y, z, intensity): {point_cloud.shape}")
        # Check if divisible by 3 (x, y, z)
        elif total_elements % 3 == 0:
            point_cloud = point_cloud.reshape(-1, 3)
            print(f"Reshaped LiDAR data as [N, 3] (x, y, z): {point_cloud.shape}")
        # Check if divisible by 5 (x, y, z, intensity, ring)
        elif total_elements % 5 == 0:
            point_cloud = point_cloud.reshape(-1, 5)
            print(f"Reshaped LiDAR data as [N, 5] (x, y, z, intensity, ring): {point_cloud.shape}")
        else:
            # Try the specified number of features as a fallback
            point_cloud = point_cloud.reshape(-1, num_features)
            print(f"Reshaped LiDAR data as [N, {num_features}]: {point_cloud.shape}")
            
    except ValueError as e:
        # If reshaping fails, try to get the size divisors to help diagnose
        divisors = [i for i in range(2, 10) if total_elements % i == 0]
        print(f"Could not reshape LiDAR data. Total elements: {total_elements}. Possible divisors: {divisors}")
        raise ValueError(f"Error reshaping LiDAR data from {file_path}. Total elements: {total_elements}. Error: {e}")
        
    # Return only x, y, z coordinates for the encoder
    return point_cloud[:, :3] 

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

def get_memory_usage():
    """Get current memory usage as a percentage."""
    import psutil
    return psutil.Process().memory_percent()

def process_lidar_chunk(chunk_data):
    """Process a chunk of LiDAR messages."""
    lidar_messages, output_dir, camera_timestamps, time_threshold = chunk_data
    processed_samples = set()
    
    for lidar_timestamp_ns, msg in lidar_messages:
        try:
            # Find closest camera sample
            best_sample_id = None
            min_time_diff = float('inf')
            
            for current_sample_id, cam_timestamp_ns in camera_timestamps:
                time_diff = abs((lidar_timestamp_ns - cam_timestamp_ns) / 1e9)
                if time_diff < min_time_diff and time_diff < time_threshold:
                    min_time_diff = time_diff
                    best_sample_id = current_sample_id
            
            if best_sample_id is not None and best_sample_id not in processed_samples:
                # Create LiDAR sample directory
                lidar_sample_dir = os.path.join(output_dir, "lidar", f"sample_id_{best_sample_id}")
                os.makedirs(lidar_sample_dir, exist_ok=True)
                
                # Extract points from LiDAR message
                points_list = []
                for data in point_cloud2.read_points(
                    msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
                ):
                    points_list.append(data)
                
                points_array = np.array(points_list, dtype=np.float32)
                
                # Process each view
                for view in ['front', 'side_left', 'side_right']:
                    # Filter points for this view
                    filtered_points = filter_lidar_by_camera_fov(points_array, {
                        'front': {'orientation': 0},
                        'side_left': {'orientation': 90},
                        'side_right': {'orientation': -90}
                    }[view])
                    
                    # Save filtered points
                    lidar_path = os.path.join(lidar_sample_dir, f"{view}.bin")
                    filtered_points.tofile(lidar_path)
                
                processed_samples.add(best_sample_id)

                image_path = os.path.join(output_dir, "images", f"sample_id_{best_sample_id}")
                if not os.path.exists(lidar_sample_dir) or len(os.listdir(lidar_sample_dir)) == 0:
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        print(f"Removed orphaned image sample: {image_path}")
                
        except Exception as e:
            print(f"Error processing LiDAR message: {e}")
            continue
    
    return processed_samples

def process_lidar_messages(lidar_messages, output_dir, camera_messages, time_threshold):
    """Process and save LiDAR data with synchronized timestamps."""
    print(f"Processing {len(lidar_messages)} LiDAR messages")
    
    # Create lidar directory
    lidar_dir = os.path.join(output_dir, "lidar")
    os.makedirs(lidar_dir, exist_ok=True)
    
    # First, collect all camera timestamps from the grouped messages
    camera_timestamps = []
    timestamp_groups = {}
    
    # Group camera messages by timestamp
    for camera_type, messages in camera_messages.items():
        for timestamp_ns, msg in messages:
            timestamp_sec = timestamp_ns / 1e9
            
            matched_group = None
            for group_time in timestamp_groups:
                if abs(group_time - timestamp_sec) < time_threshold:
                    matched_group = group_time
                    break
            
            if matched_group is None:
                matched_group = timestamp_sec
                timestamp_groups[matched_group] = {}
            
            if camera_type not in timestamp_groups[matched_group]:
                timestamp_groups[matched_group][camera_type] = (timestamp_ns, msg)
    
    # Create a list of camera timestamps from the groups
    for group_time, group_messages in timestamp_groups.items():
        if len(group_messages) == len(camera_messages):
            first_camera_type = list(group_messages.keys())[0]
            timestamp_ns, _ = group_messages[first_camera_type]
            camera_timestamps.append((len(camera_timestamps), timestamp_ns))
    
    print(f"Found {len(camera_timestamps)} camera samples to match with LiDAR")
    
    # Process LiDAR messages in chunks
    all_processed_samples = set()
    chunks = [lidar_messages[i:i + CHUNK_SIZE] for i in range(0, len(lidar_messages), CHUNK_SIZE)]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for chunk in chunks:
            # Check memory usage
            if get_memory_usage() > MEMORY_THRESHOLD:
                gc.collect()
            
            chunk_data = (chunk, output_dir, camera_timestamps, time_threshold)
            futures.append(executor.submit(process_lidar_chunk, chunk_data))
        
        # Process results as they complete
        for future in tqdm(futures, desc="Processing LiDAR chunks"):
            processed_samples = future.result()
            all_processed_samples.update(processed_samples)

    # Clean up unmatched images
    image_dir = os.path.join(output_dir, "images")
    if os.path.exists(image_dir):
        print(f"Cleaning up unmatched images in {image_dir}")
        for sample_id_dir in os.listdir(image_dir):
            if sample_id_dir.startswith("sample_id_"):
                # Extract sample ID number
                try:
                    sample_id = int(sample_id_dir.split("_")[-1])
                    lidar_sample_dir = os.path.join(lidar_dir, f"sample_id_{sample_id}")
                    
                    # Check if corresponding LiDAR data exists
                    if not os.path.exists(lidar_sample_dir) or len(os.listdir(lidar_sample_dir)) == 0:
                        sample_path = os.path.join(image_dir, sample_id_dir)
                        print(f"Removing unmatched image sample: {sample_path}")
                        # Remove directory with all images
                        import shutil
                        shutil.rmtree(sample_path)
                except ValueError:
                    continue
    
    print(f"Successfully processed {len(all_processed_samples)} LiDAR samples")
    return all_processed_samples

def process_rosbag(bag_path, output_dir, process_camera=True, process_lidar=False, time_threshold=0.1):
    """
    Process ROS bag file and extract camera images and/or LiDAR data.
    
    Args:
        bag_path (str): Path to the ROS bag file
        output_dir (str): Output directory for saving data
        process_camera (bool): Whether to process camera data
        process_lidar (bool): Whether to process LiDAR data
        time_threshold (float): Time threshold for matching camera and LiDAR data (seconds)
    """
    if not rclpy.ok():
        rclpy.init()
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Opening bag at: {bag_path}")
    
    is_sqlite = bag_path.endswith('.db3')
    storage_id = 'sqlite3' if is_sqlite else 'mcap'
    
    storage_options, converter_options = get_rosbag_options(bag_path, storage_id)
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    # Get topic information
    topic_types = {}
    for topic_metadata in reader.get_all_topics_and_types():
        topic_types[topic_metadata.name] = topic_metadata.type
    
    # Define topics
    camera_topics = {
        'front': '/argus/ar0234_front_left/image_raw',
        'side_left': '/argus/ar0234_side_left/image_raw',
        'side_right': '/argus/ar0234_side_right/image_raw'
    }
    
    lidar_topic = '/os_cloud_node/points'
    
    # Make sure our topics exist
    all_topics = list(topic_types.keys())
    available_camera_topics = {}
    
    for camera_type, topic in camera_topics.items():
        if topic in all_topics:
            available_camera_topics[camera_type] = topic
        else:
            print(f"Warning: Camera topic {topic} not found in bag")
    
    if process_lidar and lidar_topic not in all_topics:
        print(f"Error: LiDAR topic {lidar_topic} not found in bag")
        process_lidar = False
    
    if not available_camera_topics:
        print("Error: No requested camera topics found in bag")
        return
    
    print(f"Processing bag with camera topics: {available_camera_topics}")
    if process_lidar:
        print(f"LiDAR topic: {lidar_topic}")
    
    # Storage for messages
    camera_messages = {camera_type: [] for camera_type in available_camera_topics}
    lidar_messages = []
    
    # Read all messages
    print("Reading messages from bag...")
    
    while reader.has_next():
        try:
            topic_name, data, timestamp_ns = reader.read_next()
            msg_type = get_message(topic_types[topic_name])
            msg = deserialize_message(data, msg_type)
            
            # Store the message based on topic
            for camera_type, topic in available_camera_topics.items():
                if topic_name == topic:
                    camera_messages[camera_type].append((timestamp_ns, msg))
            if process_lidar and topic_name == lidar_topic:
                lidar_messages.append((timestamp_ns, msg))
                
        except Exception as e:
            print(f"Error reading message: {e}")
            continue
    
    print(f"Found {sum(len(msgs) for msgs in camera_messages.values())} camera messages")
    # if process_lidar:
    #     print(f"Found {len(lidar_messages)} LiDAR messages")
    
    # Process messages
    if process_camera:
        process_camera_messages(camera_messages, output_dir)
    
    if process_lidar:
        process_lidar_messages(lidar_messages, output_dir, camera_messages, time_threshold)
    
    # Create metadata file
    metadata_file = os.path.join(output_dir, "metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write(f"Extraction completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"ROS Bag path: {bag_path}\n")
        f.write(f"Camera topics:\n")
        for camera_type, topic in available_camera_topics.items():
            count = len(camera_messages[camera_type])
            f.write(f"  - {topic}: {count} messages\n")
        if process_lidar:
            f.write(f"LiDAR topic: {lidar_topic}, {len(lidar_messages)} messages\n")
        f.write(f"Time threshold for matching: {time_threshold} seconds\n")
    
    print(f"Processing complete. Results saved to {output_dir}")
    print(f"Metadata saved to {metadata_file}")
    
    if rclpy.ok():
        rclpy.shutdown()

def main():
    parser = argparse.ArgumentParser(description='Process ROS bag file and extract data for annotation')
    parser.add_argument('--bag_path', default='/home/vicky/IROS2025/rosbags/deadend_recovery_Apr_21_bag_2_0-001.db3', help='Path to the ROS bag file')
    parser.add_argument('--output_dir', default='/home/vicky/IROS2025/processed_bags/rosbags_21*/rosbag_21_2', help='Output directory for saving data')
    parser.add_argument('--process_camera', action='store_true', help='Process camera data')
    parser.add_argument('--process_lidar', action='store_true', help='Process LiDAR data')
    parser.add_argument('--time_threshold', type=float, default=0.1,
                        help='Time threshold for matching camera and LiDAR data (seconds)')
    
    args = parser.parse_args()
    
    # If neither camera nor LiDAR is specified, process both
    if not args.process_camera and not args.process_lidar:
        args.process_camera = True
        args.process_lidar = True
    
    process_rosbag(args.bag_path, args.output_dir, args.process_camera, args.process_lidar, args.time_threshold)

if __name__ == '__main__':
    main() 