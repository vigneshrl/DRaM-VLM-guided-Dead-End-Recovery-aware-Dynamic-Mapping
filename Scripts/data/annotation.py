import os
import sys
import json
import numpy as np
import argparse
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from functools import partial

class LidarViewer(ttk.Frame):
    """Widget for viewing and interacting with LiDAR point clouds"""
    def __init__(self, parent=None, width=5, height=4, dpi=100, view_name="LiDAR View", on_direction_select_callback=None):
        super().__init__(parent)
        self.view_name = view_name
        self.on_direction_select_callback = on_direction_select_callback
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111, projection='3d')
        self.axes.set_title(self.view_name) # Add title here
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize point cloud data
        self.points = None
        self.scatter = None
        self.direction_arrow = None
        self.selected_direction = None
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
    def load_bin_file(self, bin_path):
        """Load and display LiDAR point cloud from binary file"""
        if not os.path.exists(bin_path):
            self.clear_plot()
            return False
            
        try:
            points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
            self.points = points[:, :3]
            
            if self.points.shape[0] > 15000: # Display slightly more points
                indices = np.random.choice(self.points.shape[0], 15000, replace=False)
                display_points = self.points[indices]
            else:
                display_points = self.points
                
            self.update_plot(display_points)
            return True
        except Exception as e:
            print(f"Error loading LiDAR data: {e}")
            self.clear_plot()
            return False
            
    def update_plot(self, points=None):
        """Update the point cloud visualization"""
        self.axes.clear()
        self.axes.set_title(self.view_name)
        
        if points is None and self.points is None:
            self.canvas.draw()
            return
            
        current_points = points if points is not None else self.points
        if current_points is None or current_points.size == 0:
            self.canvas.draw()
            return
        
        # Use fixed coordinate system
        FIXED_RANGE = 10.0  # meters
        self.axes.set_xlim([-FIXED_RANGE, FIXED_RANGE])
        self.axes.set_ylim([-FIXED_RANGE, FIXED_RANGE])
        self.axes.set_zlim([-FIXED_RANGE/2, FIXED_RANGE/2])
        
        # Plot points with fixed size
        self.scatter = self.axes.scatter(current_points[:, 0], current_points[:, 1], current_points[:, 2],
                                       c=current_points[:, 2], cmap='viridis', s=2, alpha=0.6)
        
        self.axes.set_xlabel('X')
        self.axes.set_ylabel('Y')
        self.axes.set_zlabel('Z')
        
        # Draw the selected direction arrow if it exists
        if self.selected_direction is not None:
            origin = np.array([0, 0, 0])
            self.axes.quiver(origin[0], origin[1], origin[2], 
                         self.selected_direction[0], self.selected_direction[1], self.selected_direction[2],
                         color='red', arrow_length_ratio=0.1, linewidth=3) # Smaller ratio
        
        self.canvas.draw()
        
    def clear_plot(self):
        """Clear the plot area"""
        self.axes.clear()
        self.axes.set_title(self.view_name)
        self.points = None
        self.selected_direction = None
        self.canvas.draw()
        
    def on_click(self, event):
        """Handle mouse click - select direction vector and trigger callback"""
        if event.inaxes != self.axes or event.button != 1: # Only left click
            return
            
        # Get the nearest point to the click in 3D space
        if self.points is not None:
            # Convert click coordinates to 3D
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            
            # Find nearest point in the point cloud
            distances = np.sqrt((self.points[:, 0] - x)**2 + (self.points[:, 1] - y)**2)
            nearest_idx = np.argmin(distances)
            nearest_point = self.points[nearest_idx]
            
            # Create direction vector from origin to nearest point
            direction = nearest_point - np.array([0, 0, 0])
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
            else:
                direction = np.array([1.0, 0.0, 0.0])
                
            new_direction_list = direction.tolist()
        else:
            # If no points available, use the click position in XY plane
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
                
            # Simple direction from origin towards the click point in XY plane
            direction = np.array([x, y, 0.0])
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
            else:
                direction = np.array([1.0, 0.0, 0.0]) # Default to X-axis
                
            new_direction_list = direction.tolist()
            
        # Only update and trigger callback if the direction actually changed
        # Avoids flooding with updates on static clicks
        current_dir_list = self.selected_direction.tolist() if self.selected_direction is not None else None
        if new_direction_list != current_dir_list:
            self.selected_direction = direction
            self.update_plot() # Update plot to show the new arrow
            
            # Trigger the callback if it's set
            if self.on_direction_select_callback:
                self.on_direction_select_callback(new_direction_list)
        
    def get_selected_direction(self):
        """Return the currently selected direction vector"""
        return self.selected_direction.tolist() if self.selected_direction is not None else None
        
    def set_selected_direction(self, direction):
        """Set the direction vector and update the plot"""
        if direction and len(direction) == 3:
            new_dir_array = np.array(direction)
            # Only update if different from current
            current_dir_list = self.selected_direction.tolist() if self.selected_direction is not None else None
            if direction != current_dir_list:
                self.selected_direction = new_dir_array
                self.update_plot()
        # Don't clear selection here, let AnnotationTool manage clearing if needed
        # else:
        #     self.clear_selection()
        
    def clear_selection(self):
        """Clear the selected direction"""
        if self.selected_direction is not None:
            self.selected_direction = None
            self.update_plot()

    def screen_to_world_coords(self, x, y):
        """Convert screen coordinates to world coordinates"""
        # Get current view parameters
        view_angles = self.axes.view_init()
        elevation, azimuth = view_angles
        
        # Convert to radians
        elevation = np.radians(elevation)
        azimuth = np.radians(azimuth)
        
        # Calculate rotation matrices
        R_elev = np.array([
            [1, 0, 0],
            [0, np.cos(elevation), -np.sin(elevation)],
            [0, np.sin(elevation), np.cos(elevation)]
        ])
        
        R_azim = np.array([
            [np.cos(azimuth), -np.sin(azimuth), 0],
            [np.sin(azimuth), np.cos(azimuth), 0],
            [0, 0, 1]
        ])
        
        # Transform screen coordinates to world coordinates
        screen_vec = np.array([x, y, 0])
        world_vec = R_azim @ R_elev @ screen_vec
        
        return world_vec

class ImageViewer(ttk.Frame):
    """Widget for viewing camera images"""
    def __init__(self, parent=None, view_name="Image View"):
        super().__init__(parent)
        self.view_name = view_name
        
        # Add title label
        self.title_label = ttk.Label(self, text=self.view_name, anchor=tk.CENTER)
        self.title_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 2)) # Add padding below title
        
        self.image_label = ttk.Label(self)
        self.image_label.pack(expand=True, fill=tk.BOTH)
        
        self.current_image = None
        self.original_image = None # Ensure it's initialized
        self._after_id = None # For debouncing resize
        
        # Bind configure event for resizing
        self.bind("<Configure>", self.on_resize)
        
    def load_image(self, image_path):
        """Load and display an image from file"""
        if not os.path.exists(image_path):
            self.image_label.config(text="Image not found", image='')
            self.original_image = None
            self.current_image = None # Clear PhotoImage ref
            return False
            
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
            self.original_image = img
            
            # Trigger resize to display the image
            self.display_resized_image()
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            self.image_label.config(text=f"Error: {e}", image='')
            self.original_image = None
            self.current_image = None # Clear PhotoImage ref
            return False

    def display_resized_image(self):
        """Resize and display the current original image"""
        if not hasattr(self, 'original_image') or self.original_image is None:
            self.image_label.config(text="No Image", image='')
            self.current_image = None # Clear PhotoImage ref
            return

        parent_width = self.winfo_width()
        # Estimate available height (subtract title)
        title_height = self.title_label.winfo_reqheight()
        parent_height = max(1, self.winfo_height() - title_height)

        if parent_width <= 1 or parent_height <= 1:
            # Widget not yet properly sized, defer resizing
            if self._after_id: self.after_cancel(self._after_id)
            self._after_id = self.after(50, self.display_resized_image)
            return

        img_width, img_height = self.original_image.size

        # Calculate aspect ratio
        ratio = min(parent_width / img_width, parent_height / img_height)
        new_size = (max(1, int(img_width * ratio)), max(1, int(img_height * ratio)))

        if new_size[0] <=0 or new_size[1] <=0: # Prevent zero size image
             return

        # Resize using LANCZOS for quality
        img_resized = self.original_image.resize(new_size, Image.Resampling.LANCZOS)

        # Keep a reference to the PhotoImage
        self.current_image = ImageTk.PhotoImage(img_resized)
        self.image_label.config(image=self.current_image, text='') # Clear any error text

    def on_resize(self, event=None):
        """Handle resize events by redisplaying the image"""
        # Debounce resize events
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(150, self.display_resized_image) # Increased delay

class AnnotationTool(tk.Tk):
    """Main annotation tool GUI"""
    def __init__(self):
        super().__init__()
        
        self.title("Dead End Detection Annotation Tool")
        
        # Make window fullscreen
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height}")
        
        # Initialize data structures
        self.data_root = ""
        self.sample_ids = []
        self.current_sample_id = ""
        self.annotations = {}
        self.annotation_file = ""
        self.current_idx = -1
        
        # Create status bar first
        self.statusbar = ttk.Label(self, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.create_widgets()
        
        # Bind resize event to image viewers
        self.bind('<Configure>', self.on_window_resize)
        
    def create_widgets(self):
        """Create and layout all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Top bar
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)
        
        # Data directory selection
        self.data_dir_btn = ttk.Button(top_frame, text="Select Data Directory", command=self.select_data_directory)
        self.data_dir_btn.pack(side=tk.LEFT, padx=5)
        
        self.data_dir_label = ttk.Label(top_frame, text="No directory selected")
        self.data_dir_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.save_btn = ttk.Button(top_frame, text="Save Annotations", command=self.save_annotations)
        self.save_btn.pack(side=tk.RIGHT, padx=5)
        self.save_btn['state'] = 'disabled'
        
        # Content area with paned window for resizable sections
        content_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        content_paned.pack(expand=True, fill=tk.BOTH, pady=5)
        
        # Sample browser
        sample_frame = ttk.LabelFrame(content_paned, text="Samples")
        content_paned.add(sample_frame, weight=1)
        
        self.sample_list = tk.Listbox(sample_frame, width=30)
        self.sample_list.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.sample_list.bind('<<ListboxSelect>>', self.on_sample_select)
        
        # Navigation buttons
        nav_frame = ttk.Frame(sample_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.prev_btn = ttk.Button(nav_frame, text="Previous", command=self.prev_sample)
        self.prev_btn.pack(side=tk.LEFT, expand=True, padx=2)
        
        self.next_btn = ttk.Button(nav_frame, text="Next", command=self.next_sample)
        self.next_btn.pack(side=tk.LEFT, expand=True, padx=2)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(sample_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Filter options
        filter_frame = ttk.LabelFrame(sample_frame, text="Filter")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.filter_var = tk.StringVar(value="all")
        ttk.Radiobutton(filter_frame, text="All", variable=self.filter_var, 
                       value="all", command=self.apply_filter).pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(filter_frame, text="Annotated", variable=self.filter_var,
                       value="annotated", command=self.apply_filter).pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(filter_frame, text="Unannotated", variable=self.filter_var,
                       value="unannotated", command=self.apply_filter).pack(anchor=tk.W, padx=5)
        
        # Main annotation area
        annotation_frame = ttk.Frame(content_paned)
        content_paned.add(annotation_frame, weight=4) # Give more weight to annotation area
        
        # Sample info
        self.sample_info = ttk.Label(annotation_frame, text="No sample loaded")
        self.sample_info.pack(pady=5, fill=tk.X)
        
        # Split annotation area vertically
        view_paned = ttk.PanedWindow(annotation_frame, orient=tk.VERTICAL)
        view_paned.pack(expand=True, fill=tk.BOTH, pady=5)
        
        # Camera images frame (no fixed height)
        camera_frame = ttk.LabelFrame(view_paned, text="Camera Views")
        view_paned.add(camera_frame, weight=1)
        
        camera_container = ttk.Frame(camera_frame)
        camera_container.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Create Image Viewers with titles
        self.front_img = ImageViewer(camera_container, view_name="Front Camera")
        self.right_img = ImageViewer(camera_container, view_name="Right Camera")
        self.left_img = ImageViewer(camera_container, view_name="Left Camera")
        
        self.front_img.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        self.right_img.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        self.left_img.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        
        # LiDAR views frame (no fixed height)
        lidar_frame = ttk.LabelFrame(view_paned, text="LiDAR Views")
        view_paned.add(lidar_frame, weight=1)
        
        lidar_container = ttk.Frame(lidar_frame)
        lidar_container.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Create Lidar Viewers with titles and callbacks
        # Use functools.partial to pass the direction type to the handler
        self.front_lidar = LidarViewer(lidar_container, width=4, height=3, view_name="Front LiDAR",
                                       on_direction_select_callback=partial(self._handle_direction_selected, 'front'))
        self.right_lidar = LidarViewer(lidar_container, width=4, height=3, view_name="Right LiDAR",
                                       on_direction_select_callback=partial(self._handle_direction_selected, 'right'))
        self.left_lidar = LidarViewer(lidar_container, width=4, height=3, view_name="Left LiDAR",
                                      on_direction_select_callback=partial(self._handle_direction_selected, 'left'))
        
        self.front_lidar.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        self.right_lidar.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        self.left_lidar.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        
        # Annotation controls (bottom section)
        controls_frame = ttk.LabelFrame(annotation_frame, text="Annotation Controls")
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Path status
        path_frame = ttk.Frame(controls_frame)
        path_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(path_frame, text="Path Status:").pack(side=tk.LEFT)
        
        self.front_open = tk.BooleanVar()
        self.left_open = tk.BooleanVar()
        self.right_open = tk.BooleanVar()
        
        ttk.Checkbutton(path_frame, text="Front Open", variable=self.front_open, 
                       command=self.update_dead_end_status).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(path_frame, text="Left Open", variable=self.left_open,
                       command=self.update_dead_end_status).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(path_frame, text="Right Open", variable=self.right_open,
                       command=self.update_dead_end_status).pack(side=tk.LEFT, padx=5)
        
        # Dead end status
        dead_end_frame = ttk.Frame(controls_frame)
        dead_end_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(dead_end_frame, text="Dead End Status:").pack(side=tk.LEFT)
        self.dead_end_label = ttk.Label(dead_end_frame, text="Not Set")
        self.dead_end_label.pack(side=tk.LEFT, padx=5)
        
        # Direction display (no capture buttons needed)
        direction_frame = ttk.LabelFrame(controls_frame, text="Direction Vectors (Select by clicking LiDAR plot)")
        direction_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Direction display frame remains
        display_frame = ttk.Frame(direction_frame)
        display_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(display_frame, text="Front:").pack(side=tk.LEFT, padx=(0,2))
        self.front_dir_label = ttk.Label(display_frame, text="[N/A]", width=20)
        self.front_dir_label.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(display_frame, text="Left:").pack(side=tk.LEFT, padx=(0,2))
        self.left_dir_label = ttk.Label(display_frame, text="[N/A]", width=20)
        self.left_dir_label.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(display_frame, text="Right:").pack(side=tk.LEFT, padx=(0,2))
        self.right_dir_label = ttk.Label(display_frame, text="[N/A]", width=20)
        self.right_dir_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Quick tools
        quick_tools_frame = ttk.Frame(controls_frame)
        quick_tools_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.quick_dead_end = ttk.Button(quick_tools_frame, text="Mark as Dead End", command=self.mark_as_dead_end)
        self.quick_dead_end.pack(side=tk.LEFT, padx=5)
        
        self.quick_front_only = ttk.Button(quick_tools_frame, text="Front Path Only", command=self.mark_front_only)
        self.quick_front_only.pack(side=tk.LEFT, padx=5)

        self.quick_side_only = ttk.Button(quick_tools_frame, text="Side Path Only", command=self.mark_side_only)
        self.quick_side_only.pack(side=tk.LEFT, padx=5)

        self.quick_all_paths_open = ttk.Button(quick_tools_frame, text="All Paths Open", command=self.mark_all_paths_open)
        self.quick_all_paths_open.pack(side=tk.LEFT, padx=5)

        self.quick_front_right_only = ttk.Button(quick_tools_frame, text="Front Right Path Only", command=self.mark_front_right_only)
        self.quick_front_right_only.pack(side=tk.LEFT, padx=5)

        self.quick_front_left_only = ttk.Button(quick_tools_frame, text="Front Left Path Only", command=self.mark_front_left_only)
        self.quick_front_left_only.pack(side=tk.LEFT, padx=5)

        self.quick_side_left_only = ttk.Button(quick_tools_frame, text="Side Left Path Only", command=self.mark_side_left_only)
        self.quick_side_left_only.pack(side=tk.LEFT, padx=5)    

        self.quick_side_right_only = ttk.Button(quick_tools_frame, text="Side Right Path Only", command=self.mark_side_right_only)
        self.quick_side_right_only.pack(side=tk.LEFT, padx=5)
        
        
        self.quick_copy_prev = ttk.Button(quick_tools_frame, text="Copy From Previous", command=self.copy_from_previous)
        self.quick_copy_prev.pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        action_frame = ttk.Frame(controls_frame)
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.clear_btn = ttk.Button(action_frame, text="Clear", command=self.clear_annotation)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_sample_btn = ttk.Button(action_frame, text="Save Sample", command=self.save_current_sample)
        self.save_sample_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_next_btn = ttk.Button(action_frame, text="Save & Next", command=self.save_and_next)
        self.save_next_btn.pack(side=tk.LEFT, padx=5)
        
        # Initially disable controls
        self.disable_annotation_controls()
    
    def select_data_directory(self):
        """Browse for and select the dataset root directory"""
        data_dir = filedialog.askdirectory(title="Select Dataset Root Directory")
        if data_dir:
            self.data_root = data_dir
            self.data_dir_label.config(text=f"Data dir: {data_dir}")
            self.annotation_file = os.path.join(data_dir, "annotations.json")
            
            # Load annotations if they exist
            if os.path.exists(self.annotation_file):
                try:
                    with open(self.annotation_file, 'r') as f:
                        self.annotations = json.load(f)
                    self.statusbar.config(text=f"Loaded {len(self.annotations)} existing annotations")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load annotations: {e}")
                    self.annotations = {}
            else:
                self.annotations = {}
            
            # Find all sample directories
            images_dir = os.path.join(data_dir, "images")
            if os.path.exists(images_dir):
                self.sample_ids = []
                for d in os.listdir(images_dir):
                    sample_path = os.path.join(images_dir, d)
                    if os.path.isdir(sample_path):
                        self.sample_ids.append(d)
                
                self.sample_ids.sort()
                self.update_sample_list()
                self.save_btn.config(state=tk.NORMAL)
                self.statusbar.config(text=f"Found {len(self.sample_ids)} samples")
            else:
                messagebox.showerror("Error", f"Images directory not found at {images_dir}")
    
    def update_sample_list(self):
        """Update the sample list widget"""
        self.sample_list.delete(0, tk.END)
        for sample_id in self.sample_ids:
            status = "✓" if sample_id in self.annotations else "○"
            self.sample_list.insert(tk.END, f"{status} {sample_id}")
        
        # Update progress bar
        if self.sample_ids:
            annotated = sum(1 for s in self.sample_ids if s in self.annotations)
            self.progress_bar["value"] = int(annotated / len(self.sample_ids) * 100)
    
    def apply_filter(self):
        """Apply the selected filter to the sample list"""
        if not self.sample_ids:
            return
            
        filtered_ids = []
        if self.filter_var.get() == "all":
            filtered_ids = self.sample_ids
        elif self.filter_var.get() == "annotated":
            filtered_ids = [s for s in self.sample_ids if s in self.annotations]
        elif self.filter_var.get() == "unannotated":
            filtered_ids = [s for s in self.sample_ids if s not in self.annotations]
        
        self.sample_list.delete(0, tk.END)
        for sample_id in filtered_ids:
            status = "✓" if sample_id in self.annotations else "○"
            self.sample_list.insert(tk.END, f"{status} {sample_id}")
    
    def on_sample_select(self, event):
        """Handle sample selection from the list"""
        selection = self.sample_list.curselection()
        if not selection:
            return
            
        row = selection[0]
        item_text = self.sample_list.get(row)
        sample_id = item_text[2:]  # Remove status indicator
        
        self.current_idx = row
        self.current_sample_id = sample_id
        self.load_sample(row)
    
    def load_sample(self, idx):
        """Load sample at given index"""
        if not (0 <= idx < len(self.sample_ids)):
             return

        self.current_idx = idx
        self.current_sample_id = self.sample_ids[idx]

        # Update status
        self.statusbar.config(text=f"Sample {idx + 1} of {len(self.sample_ids)}: {self.current_sample_id}")
        
        # Load images
        image_dir = os.path.join(self.data_root, "images", self.current_sample_id)
        self.front_img.load_image(os.path.join(image_dir, "front.jpg"))
        self.right_img.load_image(os.path.join(image_dir, "side_right.jpg"))
        self.left_img.load_image(os.path.join(image_dir, "side_left.jpg"))
        
        # Load LiDAR data
        lidar_dir = os.path.join(self.data_root, "lidar", self.current_sample_id)
        self.front_lidar.load_bin_file(os.path.join(lidar_dir, "front.bin"))
        self.right_lidar.load_bin_file(os.path.join(lidar_dir, "side_right.bin"))
        self.left_lidar.load_bin_file(os.path.join(lidar_dir, "side_left.bin"))

        # Ensure annotation entry exists before accessing it
        self._ensure_annotation_exists()
        ann = self.annotations[self.current_sample_id]

        # Load existing annotation values into UI
        self.front_open.set(ann.get("front_open", 0) == 1)
        self.left_open.set(ann.get("side_left_open", 0) == 1)
        self.right_open.set(ann.get("side_right_open", 0) == 1)

        # Update direction labels and set directions in LiDAR viewers
        front_dir = ann.get('front_direction', [0.0, 0.0, 0.0])
        left_dir = ann.get('left_direction', [0.0, 0.0, 0.0])
        right_dir = ann.get('right_direction', [0.0, 0.0, 0.0])

        self.front_dir_label.config(text=f"[{front_dir[0]:.2f}, {front_dir[1]:.2f}, {front_dir[2]:.2f}]")
        self.left_dir_label.config(text=f"[{left_dir[0]:.2f}, {left_dir[1]:.2f}, {left_dir[2]:.2f}]")
        self.right_dir_label.config(text=f"[{right_dir[0]:.2f}, {right_dir[1]:.2f}, {right_dir[2]:.2f}]")

        # Set the arrows in the viewers
        self.front_lidar.set_selected_direction(front_dir)
        self.left_lidar.set_selected_direction(left_dir)
        self.right_lidar.set_selected_direction(right_dir)

        # Update dead end status label (doesn't need saving here, handled by its own update method)
        self.update_dead_end_status(save_changes=False) # Avoid double save on load
        
        # Enable controls
        self.enable_annotation_controls()
    
        # Print debug info
        print(f"Loading images from: {image_dir}")
        print(f"Loading LiDAR data from: {lidar_dir}")

    def update_dead_end_status(self, save_changes=True):
        """Update the dead-end status based on path openness"""
        is_dead_end = not (self.front_open.get() or
                           self.left_open.get() or
                           self.right_open.get())
        
        if is_dead_end:
            self.dead_end_label.config(text="DEAD END", foreground="red")
        else:
            self.dead_end_label.config(text="NOT A DEAD END", foreground="green")

        # Update the annotation immediately
        if self.current_sample_id:
            self._ensure_annotation_exists()
            ann = self.annotations[self.current_sample_id]
            ann['front_open'] = 1 if self.front_open.get() else 0
            ann['side_left_open'] = 1 if self.left_open.get() else 0
            ann['side_right_open'] = 1 if self.right_open.get() else 0
            ann['is_dead_end'] = 1 if is_dead_end else 0

            # Save the changes if requested
            if save_changes:
                 self.save_current_sample()

    def _handle_direction_selected(self, direction_type, direction_vector):
        """Callback function when a direction is selected in a LidarViewer."""
        if not self.current_sample_id:
            return
        
        if direction_vector is None:
            direction_vector = [0.0, 0.0, 0.0]
        
        # Update direction label
        label_text = f"[{direction_vector[0]:.2f}, {direction_vector[1]:.2f}, {direction_vector[2]:.2f}]"
        if direction_type == 'front':
            self.front_dir_label.config(text=label_text)
            self.front_open.set(True)  # Automatically mark path as open
        elif direction_type == 'left':
            self.left_dir_label.config(text=label_text)
            self.left_open.set(True)
        elif direction_type == 'right':
            self.right_dir_label.config(text=label_text)
            self.right_open.set(True)
        
        # Update annotation data
        self._ensure_annotation_exists()
        self.annotations[self.current_sample_id][f"{direction_type}_direction"] = direction_vector
        self.annotations[self.current_sample_id][f"{direction_type}_open"] = 1
        
        # Save and update status
        self.save_current_sample()
        self.update_dead_end_status()

    def _ensured_annotation_exists(self):
        """Helper to create a default annotation if it doesn't exist."""
        if self.current_sample_id and self.current_sample_id not in self.annotations:
             self.annotations[self.current_sample_id] = {
                "front_open": 1 if self.front_open.get() else 0,
                "side_left_open": 1 if self.left_open.get() else 0,
                "side_right_open": 1 if self.right_open.get() else 0,
                "is_dead_end": 1 if not (self.front_open.get() or self.left_open.get() or self.right_open.get()) else 0,
                "front_direction": [0.0, 0.0, 0.0],
                "left_direction": [0.0, 0.0, 0.0],
                "right_direction": [0.0, 0.0, 0.0]
        }
    
    def save_current_sample(self):
        """Save the current sample annotation (data is already updated)"""
        if not self.current_sample_id:
            return
            
        # Ensure the annotation entry exists (might be redundant but safe)
        self._ensure_annotation_exists()
        
        # The self.annotations[self.current_sample_id] dictionary
        # should already be up-to-date due to direct updates in callbacks.
        
        # Update sample status in list
        # Check if the index is valid for the current list view
        try:
             list_item = self.sample_list.get(self.current_idx)
             expected_id = list_item[2:] # Remove status indicator
             if self.current_sample_id == expected_id:
                 # Only update if the index corresponds to the current sample
                 current_status = "✓"
                 self.sample_list.delete(self.current_idx)
                 self.sample_list.insert(self.current_idx, f"{current_status} {self.current_sample_id}")
                 self.sample_list.itemconfig(self.current_idx, {'fg': 'black'}) # Ensure default color
             else:
                 # Index mismatch (e.g., due to filtering), find the item to update
                 self.update_sample_list() # Just refresh the whole list for simplicity
        except tk.TclError:
            # Index out of bounds, likely due to filtering. Refresh list.
             print(f"Info: Index {self.current_idx} out of bounds for list update. Refreshing list.")
             self.update_sample_list()

        # Save all annotations to disk
        self.save_annotations()
        
        self.statusbar.config(text=f"Saved annotation for {self.current_sample_id}")
        
    def save_annotations(self):
        """Save annotations to JSON file"""
        try:
            with open(self.annotation_file, 'w') as f:
                json.dump(self.annotations, f, indent=4)
            self.statusbar.config(text="Annotations saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {e}")
            
    def save_and_next(self):
        """Save current sample and load next sample"""
        self.save_current_sample()
        if self.current_idx < self.sample_list.size() - 1:
            self.load_sample(self.current_idx + 1)
    
    def prev_sample(self):
        """Load the previous sample"""
        if self.current_idx > 0:
            self.load_sample(self.current_idx - 1)
    
    def next_sample(self):
        """Load the next sample"""
        if self.current_idx < self.sample_list.size() - 1:
            self.load_sample(self.current_idx + 1)

    def mark_as_dead_end(self):
        """Quick tool to mark the current sample as a dead end"""
        self.front_open.set(False)
        self.left_open.set(False)
        self.right_open.set(False)
        
        # Clear all direction vectors
        self.front_lidar.clear_selection()
        self.right_lidar.clear_selection()
        self.left_lidar.clear_selection()
        
        # Update direction labels
        self.front_dir_label.config(text="[N/A]")
        self.left_dir_label.config(text="[N/A]")
        self.right_dir_label.config(text="[N/A]")
        
        # Update annotation data
        self._ensure_annotation_exists()
        self.annotations[self.current_sample_id]["front_direction"] = [0.0, 0.0, 0.0]
        self.annotations[self.current_sample_id]["left_direction"] = [0.0, 0.0, 0.0]
        self.annotations[self.current_sample_id]["right_direction"] = [0.0, 0.0, 0.0]
        
        self.update_dead_end_status()
        self.statusbar.config(text="Marked as dead end")
    
    def mark_front_only(self):
        """Quick tool to mark only the front path as open"""
        self.front_open.set(True)
        self.left_open.set(False)
        self.right_open.set(False)
        
        # Set default direction vector for front (straight ahead)
        default_front_dir = [1.0, 0.0, 0.0]
        self.front_lidar.set_selected_direction(default_front_dir)
        self.front_dir_label.config(text=f"[{default_front_dir[0]:.2f}, {default_front_dir[1]:.2f}, {default_front_dir[2]:.2f}]")
        
        # Clear other directions
        self.left_lidar.clear_selection()
        self.right_lidar.clear_selection()
        self.left_dir_label.config(text="[N/A]")
        self.right_dir_label.config(text="[N/A]")
        
        # Update annotation data
        self._ensure_annotation_exists()
        self.annotations[self.current_sample_id]["front_direction"] = default_front_dir
        self.annotations[self.current_sample_id]["left_direction"] = [0.0, 0.0, 0.0]
        self.annotations[self.current_sample_id]["right_direction"] = [0.0, 0.0, 0.0]
        
        self.update_dead_end_status()
        self.statusbar.config(text="Marked front path only")
    
    def mark_front_right_only(self):
        """Quick tool to mark only the front and right paths as open"""
        self.front_open.set(True)
        self.left_open.set(False)
        self.right_open.set(True)   
        
        # Set default direction vectors for front and right
        default_front_dir = [1.0, 0.0, 0.0]
        default_right_dir = [0.0, -1.0, 0.0]
        
        # Update LiDAR viewers
        self.front_lidar.set_selected_direction(default_front_dir)
        self.right_lidar.set_selected_direction(default_right_dir)
        self.left_lidar.clear_selection()
        
        # Update direction labels
        self.front_dir_label.config(text=f"[{default_front_dir[0]:.2f}, {default_front_dir[1]:.2f}, {default_front_dir[2]:.2f}]")
        self.right_dir_label.config(text=f"[{default_right_dir[0]:.2f}, {default_right_dir[1]:.2f}, {default_right_dir[2]:.2f}]")
        self.left_dir_label.config(text="[N/A]")
        
        # Update annotation data
        self._ensure_annotation_exists()
        self.annotations[self.current_sample_id]["front_direction"] = default_front_dir 
        self.annotations[self.current_sample_id]["right_direction"] = default_right_dir
        self.annotations[self.current_sample_id]["left_direction"] = [0.0, 0.0, 0.0]
        
        self.update_dead_end_status()
        self.statusbar.config(text="Marked front and right paths only")     
        
    def mark_front_left_only(self):
        """Quick tool to mark only the front and left paths as open"""
        self.front_open.set(True)
        self.left_open.set(True)
        self.right_open.set(False)
        
        # Set default direction vectors for front and left
        default_front_dir = [1.0, 0.0, 0.0]
        default_left_dir = [0.0, 1.0, 0.0]
        
        # Update LiDAR viewers
        self.front_lidar.set_selected_direction(default_front_dir)  
        self.left_lidar.set_selected_direction(default_left_dir)    
        self.right_lidar.clear_selection()
        
        # Update direction labels
        self.front_dir_label.config(text=f"[{default_front_dir[0]:.2f}, {default_front_dir[1]:.2f}, {default_front_dir[2]:.2f}]")
        self.left_dir_label.config(text=f"[{default_left_dir[0]:.2f}, {default_left_dir[1]:.2f}, {default_left_dir[2]:.2f}]")
        self.right_dir_label.config(text="[N/A]")   
        
        # Update annotation data
        self._ensure_annotation_exists()
        self.annotations[self.current_sample_id]["front_direction"] = default_front_dir
        self.annotations[self.current_sample_id]["left_direction"] = default_left_dir
        self.annotations[self.current_sample_id]["right_direction"] = [0.0, 0.0, 0.0]
        
        self.update_dead_end_status()
        self.statusbar.config(text="Marked front and left paths only")
        
    def mark_all_paths_open(self):
        """Quick tool to mark all paths as open"""
        self.front_open.set(True)
        self.left_open.set(True)
        self.right_open.set(True)   
        
        # Set default direction vectors for all paths
        default_front_dir = [1.0, 0.0, 0.0]
        default_left_dir = [0.0, 1.0, 0.0]
        default_right_dir = [0.0, -1.0, 0.0]
        
        # Update LiDAR viewers
        self.front_lidar.set_selected_direction(default_front_dir)
        self.left_lidar.set_selected_direction(default_left_dir)
        self.right_lidar.set_selected_direction(default_right_dir)
        
        # Update direction labels
        self.front_dir_label.config(text=f"[{default_front_dir[0]:.2f}, {default_front_dir[1]:.2f}, {default_front_dir[2]:.2f}]")
        self.left_dir_label.config(text=f"[{default_left_dir[0]:.2f}, {default_left_dir[1]:.2f}, {default_left_dir[2]:.2f}]")
        self.right_dir_label.config(text=f"[{default_right_dir[0]:.2f}, {default_right_dir[1]:.2f}, {default_right_dir[2]:.2f}]")
        
        # Update annotation data
        self._ensure_annotation_exists()    
        self.annotations[self.current_sample_id]["front_direction"] = default_front_dir
        self.annotations[self.current_sample_id]["left_direction"] = default_left_dir
        self.annotations[self.current_sample_id]["right_direction"] = default_right_dir
        
        self.update_dead_end_status()
        self.statusbar.config(text="Marked all paths open")         
        
        
    def mark_side_only(self):
        """Quick tool to mark only the side paths as open"""
        self.front_open.set(False)
        self.left_open.set(True)
        self.right_open.set(True)
        
        # Set default direction vectors for sides
        default_left_dir = [0.0, 1.0, 0.0]  # Left direction
        default_right_dir = [0.0, -1.0, 0.0] # Right direction
        
        # Update LiDAR viewers
        self.left_lidar.set_selected_direction(default_left_dir)
        self.right_lidar.set_selected_direction(default_right_dir)
        self.front_lidar.clear_selection()
        
        # Update direction labels
        self.left_dir_label.config(text=f"[{default_left_dir[0]:.2f}, {default_left_dir[1]:.2f}, {default_left_dir[2]:.2f}]")
        self.right_dir_label.config(text=f"[{default_right_dir[0]:.2f}, {default_right_dir[1]:.2f}, {default_right_dir[2]:.2f}]")
        self.front_dir_label.config(text="[N/A]")
        
        # Update annotation data
        self._ensure_annotation_exists()
        self.annotations[self.current_sample_id]["front_direction"] = [0.0, 0.0, 0.0]
        self.annotations[self.current_sample_id]["left_direction"] = default_left_dir
        self.annotations[self.current_sample_id]["right_direction"] = default_right_dir
        
        self.update_dead_end_status()
        self.statusbar.config(text="Marked side paths only")
    
    def mark_side_left_only(self):
        """Quick tool to mark only the side left path as open"""
        self.front_open.set(False)
        self.left_open.set(True)
        self.right_open.set(False)
        
        # Set default direction vector for left side
        default_left_dir = [0.0, 1.0, 0.0]  # Left direction
        
        # Update LiDAR viewers
        self.left_lidar.set_selected_direction(default_left_dir)
        self.front_lidar.clear_selection()
        self.right_lidar.clear_selection()
        
        # Update direction labels
        self.left_dir_label.config(text=f"[{default_left_dir[0]:.2f}, {default_left_dir[1]:.2f}, {default_left_dir[2]:.2f}]")
        self.front_dir_label.config(text="[N/A]")
        self.right_dir_label.config(text="[N/A]")
        
        # Update annotation data
        self._ensure_annotation_exists()
        self.annotations[self.current_sample_id]["front_direction"] = [0.0, 0.0, 0.0]
        self.annotations[self.current_sample_id]["left_direction"] = default_left_dir
        self.annotations[self.current_sample_id]["right_direction"] = [0.0, 0.0, 0.0]
        
        self.update_dead_end_status()
        self.statusbar.config(text="Marked side left path only")
    
    def mark_side_right_only(self):
        """Quick tool to mark only the side right path as open"""
        self.front_open.set(False)
        self.left_open.set(False)
        self.right_open.set(True)
        
        # Set default direction vector for right side
        default_right_dir = [0.0, -1.0, 0.0] # Right direction
        
        # Update LiDAR viewers
        self.right_lidar.set_selected_direction(default_right_dir)
        self.front_lidar.clear_selection()
        self.left_lidar.clear_selection()
        
        # Update direction labels
        self.right_dir_label.config(text=f"[{default_right_dir[0]:.2f}, {default_right_dir[1]:.2f}, {default_right_dir[2]:.2f}]")
        self.front_dir_label.config(text="[N/A]")
        self.left_dir_label.config(text="[N/A]")
        
        # Update annotation data
        self._ensure_annotation_exists()
        self.annotations[self.current_sample_id]["front_direction"] = [0.0, 0.0, 0.0]
        self.annotations[self.current_sample_id]["left_direction"] = [0.0, 0.0, 0.0]
        self.annotations[self.current_sample_id]["right_direction"] = default_right_dir
        
        self.update_dead_end_status()
        self.statusbar.config(text="Marked side right path only")

    def copy_from_previous(self):
        """Copy annotations from the previous sample"""
        if self.current_idx <= 0 or len(self.sample_ids) <= 1:
            messagebox.showwarning("Error", "No previous sample available")
            return
            
        prev_item = self.sample_list.get(self.current_idx - 1)
        prev_sample_id = prev_item[2:]  # Remove status indicator
        
        if prev_sample_id not in self.annotations:
            messagebox.showwarning("Error", "No annotation for previous sample")
            return
            
        self.load_sample(self.current_idx - 1)
        self.statusbar.config(text=f"Copied annotation from {prev_sample_id}")
    
    def enable_annotation_controls(self):
        """Enable the annotation controls"""
        # Checkbuttons are always enabled or handled by load_sample
        # No capture buttons to enable
        self.clear_btn.config(state=tk.NORMAL)
        self.save_sample_btn.config(state=tk.NORMAL)
        self.save_next_btn.config(state=tk.NORMAL)
        self.quick_dead_end.config(state=tk.NORMAL)
        self.quick_front_only.config(state=tk.NORMAL)
        self.quick_copy_prev.config(state=tk.NORMAL)

    def disable_annotation_controls(self):
        """Disable the annotation controls"""
        # Checkbuttons state set by load/clear
        # No capture buttons to disable
        self.clear_btn.config(state=tk.DISABLED)
        self.save_sample_btn.config(state=tk.DISABLED)
        self.save_next_btn.config(state=tk.DISABLED)
        self.quick_dead_end.config(state=tk.DISABLED)
        self.quick_front_only.config(state=tk.DISABLED)
        self.quick_copy_prev.config(state=tk.DISABLED)
    
    def on_window_resize(self, event):
        """Handle window resize events"""
        # Update images if they exist
        if hasattr(self, 'front_img'):
            self.front_img.on_resize(event)
        if hasattr(self, 'right_img'):
            self.right_img.on_resize(event)
        if hasattr(self, 'left_img'):
            self.left_img.on_resize(event)
    
    def clear_annotation(self):
        """Clear the current annotation UI and data"""
        if not self.current_sample_id:
            return

        # Reset path status checkboxes
        self.front_open.set(False)
        self.left_open.set(False)
        self.right_open.set(False)

        # Reset direction labels
        self.front_dir_label.config(text="[N/A]")
        self.left_dir_label.config(text="[N/A]")
        self.right_dir_label.config(text="[N/A]")

        # Clear LiDAR selections/arrows
        self.front_lidar.clear_selection()
        self.right_lidar.clear_selection()
        self.left_lidar.clear_selection()

        # Reset annotation data for this sample
        self._ensure_annotation_exists()
        self.annotations[self.current_sample_id]["front_open"] = 0
        self.annotations[self.current_sample_id]["side_left_open"] = 0
        self.annotations[self.current_sample_id]["side_right_open"] = 0
        self.annotations[self.current_sample_id]["is_dead_end"] = 1 # Default to dead end on clear
        self.annotations[self.current_sample_id]["front_direction"] = [0.0, 0.0, 0.0]
        self.annotations[self.current_sample_id]["left_direction"] = [0.0, 0.0, 0.0]
        self.annotations[self.current_sample_id]["right_direction"] = [0.0, 0.0, 0.0]

        # Update dead end status label (without saving again)
        self.update_dead_end_status(save_changes=False)

        # Save the cleared state
        self.save_current_sample()
        self.statusbar.config(text="Annotation cleared")

def main(data_root=None):
    """Main application entry point"""
    app = AnnotationTool()
    
    # If data_root is provided, initialize with it
    if data_root and os.path.exists(data_root):
        app.data_root = data_root
        app.data_dir_label.config(text=f"Data dir: {data_root}")
        app.annotation_file = os.path.join(data_root, "annotations.json")
        
        # Load annotations if they exist
        if os.path.exists(app.annotation_file):
            try:
                with open(app.annotation_file, 'r') as f:
                    app.annotations = json.load(f)
                app.statusbar.config(text=f"Loaded {len(app.annotations)} existing annotations")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load annotations: {e}")
                app.annotations = {}
        else:
            app.annotations = {}
        
        # Find all sample directories
        images_dir = os.path.join(data_root, "images")
        if os.path.exists(images_dir):
            app.sample_ids = []
            for d in os.listdir(images_dir):
                sample_path = os.path.join(images_dir, d)
                if os.path.isdir(sample_path):
                    app.sample_ids.append(d)
            
            app.sample_ids.sort()
            app.update_sample_list()
            app.save_btn.config(state=tk.NORMAL)
            app.statusbar.config(text=f"Found {len(app.sample_ids)} samples")
    
    app.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dead End Detection Annotation Tool")
    parser.add_argument("--data_root", type=str, help="Path to the dataset directory (optional)")
    args = parser.parse_args()
    
    main(args.data_root if hasattr(args, 'data_root') else None)
        
    