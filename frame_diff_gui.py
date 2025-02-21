import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import PIL.Image, PIL.ImageTk
import numpy as np
from frame_diff_detector import FrameDiffDetector
from datetime import datetime
import threading
import queue
import os
import json
from pathlib import Path
import time

class SliderWithValue(ttk.Frame):
    """Custom slider widget with value display and tooltip"""
    def __init__(self, parent, text, variable, from_, to, tooltip="", unit="", **kwargs):
        super().__init__(parent)
        
        # Create main label frame
        self.label_frame = ttk.LabelFrame(self, text=text)
        self.label_frame.pack(fill=tk.X, padx=2, pady=2)
        
        # Create slider frame
        slider_frame = ttk.Frame(self.label_frame)
        slider_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Create and pack the slider
        self.slider = ttk.Scale(slider_frame, from_=from_, to=to,
                              variable=variable, orient=tk.HORIZONTAL, **kwargs)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Create and pack the value label
        self.value_label = ttk.Label(slider_frame, width=8)
        self.value_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Create description label
        if tooltip:
            desc_label = ttk.Label(self.label_frame, text=tooltip, wraplength=250,
                                 font=('TkDefaultFont', 9, 'italic'))
            desc_label.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # Store unit for display
        self.unit = unit
        
        # Bind value updates
        variable.trace_add('write', self.update_value_label)
        self.update_value_label()
        
    def update_value_label(self, *args):
        """Update the value label when the slider changes"""
        try:
            value = self.slider.get()
            self.value_label.config(text=f"{value:.1f}{self.unit}")
        except:
            self.value_label.config(text="N/A")

class FrameDiffGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Frame Difference Detector")
        
        # Thread safety
        self.gui_update_lock = threading.Lock()
        self.frame_update_lock = threading.Lock()
        
        # Load or create default settings
        self.settings = self.load_settings()
        
        # Initialize variables
        self.threshold = tk.DoubleVar(value=self.settings['threshold'])
        self.min_area = tk.DoubleVar(value=self.settings['min_area'])
        self.frame_width = tk.IntVar(value=self.settings['frame_width'])
        self.frame_height = tk.IntVar(value=self.settings['frame_height'])
        self.save_frames = tk.BooleanVar(value=self.settings['save_frames'])
        self.save_original = tk.BooleanVar(value=self.settings['save_original'])
        self.save_diff = tk.BooleanVar(value=self.settings['save_diff'])
        self.create_date_folders = tk.BooleanVar(value=self.settings['create_date_folders'])
        self.show_diff_overlay = tk.BooleanVar(value=self.settings['show_diff_overlay'])
        self.camera_source = tk.StringVar(value=self.settings['camera_source'])
        self.save_directory = tk.StringVar(value=self.settings['save_directory'])
        self.update_interval = tk.IntVar(value=self.settings['update_interval'])
        
        self.is_running = False
        self.queue = queue.Queue(maxsize=2)  # Limit queue size for better performance
        self.session_start_time = None
        
        # Add new variables
        self.noise_reduction = tk.IntVar(value=self.settings.get('noise_reduction', 0))
        self.light_compensation = tk.BooleanVar(value=self.settings.get('light_compensation', False))
        self.detection_zones = []
        self.is_drawing_zone = False
        self.start_point = None
        self.current_zone = None
        self.recording = False
        self.video_writer = None
        
        # Error tracking
        self.error_count = 0
        self.last_error_time = None
        
        # Performance monitoring
        self.last_frame_time = datetime.now()
        self.fps_update_counter = 0
        
        # Frame counter
        self.frames_saved = 0
        
        # Create GUI elements
        self.create_widgets()
        self.create_menu()
        
        # Initialize video capture and detector
        self.cap = None
        self.detector = None
        
        # Bind window close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_menu(self):
        menubar = tk.Menu(self.window)
        self.window.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Select Save Directory", command=self.select_save_directory)
        file_menu.add_command(label="Save Settings", command=self.save_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Show Difference Overlay", variable=self.show_diff_overlay)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def create_widgets(self):
        # Set window minimum size and style
        self.window.minsize(1200, 800)
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        style.configure('Info.TLabel', font=('Helvetica', 10, 'italic'))
        
        # Create main container with padding
        container = ttk.Frame(self.window, padding="10")
        container.pack(fill=tk.BOTH, expand=True)
        
        # Create title
        title_frame = ttk.Frame(container)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(title_frame, text="Frame Difference Detector", style='Title.TLabel').pack(side=tk.LEFT)
        
        # Create main content frame
        main_frame = ttk.Frame(container)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create video frame (left side)
        video_container = ttk.LabelFrame(main_frame, text="Video Feed", padding="5")
        video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video display
        self.video_label = ttk.Label(video_container)
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Bind mouse events for zone drawing
        self.video_label.bind('<Button-1>', self.on_mouse_down)
        self.video_label.bind('<B1-Motion>', self.on_mouse_move)
        self.video_label.bind('<ButtonRelease-1>', self.on_mouse_up)
        
        # Create control panel (right side)
        control_container = ttk.Frame(main_frame)
        control_container.pack(side=tk.LEFT, fill=tk.Y)
        
        # Add scrollbar to control panel
        control_scroll = ttk.Scrollbar(control_container, orient=tk.VERTICAL)
        control_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create canvas for scrolling
        canvas = tk.Canvas(control_container, yscrollcommand=control_scroll.set, width=400)
        canvas.pack(side=tk.LEFT, fill=tk.Y)
        
        # Configure scrollbar
        control_scroll.config(command=canvas.yview)
        
        # Create frame for controls inside canvas
        self.control_frame = ttk.Frame(canvas)
        canvas_frame = canvas.create_window((0, 0), window=self.control_frame, anchor=tk.NW, width=380)
        
        # Configure canvas scrolling
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        self.control_frame.bind('<Configure>', configure_scroll_region)
        
        # Enable mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Quick Actions Section
        quick_frame = ttk.LabelFrame(self.control_frame, text="Quick Actions", padding="5")
        quick_frame.pack(fill=tk.X, pady=(0, 10))
        
        quick_buttons = ttk.Frame(quick_frame)
        quick_buttons.pack(fill=tk.X)
        
        self.start_button = ttk.Button(quick_buttons, text="Start Camera",
                                     command=self.toggle_camera, width=15)
        self.start_button.pack(side=tk.LEFT, padx=2)
        
        self.record_button = ttk.Button(quick_buttons, text="Start Recording",
                                      command=self.toggle_recording, width=15)
        self.record_button.pack(side=tk.LEFT, padx=2)
        
        # Camera Settings Section
        camera_frame = ttk.LabelFrame(self.control_frame, text="Camera Settings", padding="5")
        camera_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Camera source with refresh button
        source_frame = ttk.Frame(camera_frame)
        source_frame.pack(fill=tk.X, pady=2)
        ttk.Label(source_frame, text="Camera Source:", style='Header.TLabel').pack(side=tk.LEFT)
        self.camera_combo = ttk.Combobox(source_frame, textvariable=self.camera_source, width=15)
        self.camera_combo['values'] = self.get_available_cameras()
        self.camera_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(source_frame, text="⟳", width=3,
                  command=self.refresh_cameras).pack(side=tk.LEFT)
        
        # Resolution settings
        res_frame = ttk.Frame(camera_frame)
        res_frame.pack(fill=tk.X, pady=5)
        ttk.Label(res_frame, text="Resolution:", style='Header.TLabel').pack(side=tk.LEFT)
        resolutions = ["Custom", "640x480 (VGA)", "1280x720 (HD)", "1920x1080 (FHD)"]
        self.resolution_var = tk.StringVar(value=resolutions[0])
        res_combo = ttk.Combobox(res_frame, textvariable=self.resolution_var,
                                values=resolutions, width=20)
        res_combo.pack(side=tk.LEFT, padx=5)
        res_combo.bind('<<ComboboxSelected>>', self.on_resolution_change)
        
        # Custom resolution
        custom_res_frame = ttk.Frame(camera_frame)
        custom_res_frame.pack(fill=tk.X, pady=2)
        vcmd = (self.window.register(self.validate_number), '%P')
        
        width_frame = ttk.Frame(custom_res_frame)
        width_frame.pack(side=tk.LEFT)
        ttk.Label(width_frame, text="Width:").pack(side=tk.LEFT)
        ttk.Entry(width_frame, textvariable=self.frame_width,
                 validate='key', validatecommand=vcmd, width=6).pack(side=tk.LEFT, padx=2)
        
        height_frame = ttk.Frame(custom_res_frame)
        height_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(height_frame, text="Height:").pack(side=tk.LEFT)
        ttk.Entry(height_frame, textvariable=self.frame_height,
                 validate='key', validatecommand=vcmd, width=6).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(camera_frame, text="Note: Higher resolutions may impact performance",
                 style='Info.TLabel').pack(pady=(5, 0))
        
        # Detection Settings Section
        detection_frame = ttk.LabelFrame(self.control_frame, text="Detection Settings", padding="5")
        detection_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Sensitivity preset
        preset_frame = ttk.Frame(detection_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        ttk.Label(preset_frame, text="Sensitivity:", style='Header.TLabel').pack(side=tk.LEFT)
        presets = ["Custom", "Low", "Medium", "High"]
        self.sensitivity_var = tk.StringVar(value="Custom")
        sensitivity_combo = ttk.Combobox(preset_frame, textvariable=self.sensitivity_var,
                                       values=presets, width=10)
        sensitivity_combo.pack(side=tk.LEFT, padx=5)
        sensitivity_combo.bind('<<ComboboxSelected>>', self.on_sensitivity_change)
        
        # Sliders
        self.threshold_slider = SliderWithValue(
            detection_frame,
            text="Pixel Difference Threshold",
            variable=self.threshold,
            from_=0, to=255,
            tooltip="Lower values detect subtle changes",
            unit=""
        )
        self.threshold_slider.pack(fill=tk.X, pady=2)
        
        self.min_area_slider = SliderWithValue(
            detection_frame,
            text="Minimum Change Area",
            variable=self.min_area,
            from_=0, to=100,
            tooltip="Higher values require larger changes",
            unit="%"
        )
        self.min_area_slider.pack(fill=tk.X, pady=2)
        
        self.noise_slider = SliderWithValue(
            detection_frame,
            text="Noise Reduction",
            variable=self.noise_reduction,
            from_=0, to=5,
            tooltip="Reduce false detections",
            unit=""
        )
        self.noise_slider.pack(fill=tk.X, pady=2)
        
        # Light compensation
        ttk.Checkbutton(detection_frame, text="Light Change Compensation",
                       variable=self.light_compensation).pack(anchor=tk.W, pady=5)
        
        # Detection Zones
        zone_frame = ttk.LabelFrame(detection_frame, text="Detection Zones", padding="5")
        zone_frame.pack(fill=tk.X, pady=5)
        
        zone_buttons = ttk.Frame(zone_frame)
        zone_buttons.pack(fill=tk.X)
        
        ttk.Button(zone_buttons, text="Add Zone", width=12,
                  command=self.start_zone_drawing).pack(side=tk.LEFT, padx=2)
        ttk.Button(zone_buttons, text="Clear Zones", width=12,
                  command=self.clear_zones).pack(side=tk.LEFT, padx=2)
        
        # Save Settings Section
        save_frame = ttk.LabelFrame(self.control_frame, text="Save Settings", padding="5")
        save_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Save options
        ttk.Checkbutton(save_frame, text="Enable Frame Saving",
                       variable=self.save_frames).pack(anchor=tk.W)
        ttk.Checkbutton(save_frame, text="Save Original Frames",
                       variable=self.save_original).pack(anchor=tk.W)
        ttk.Checkbutton(save_frame, text="Save Difference Frames",
                       variable=self.save_diff).pack(anchor=tk.W)
        ttk.Checkbutton(save_frame, text="Create Date Folders",
                       variable=self.create_date_folders).pack(anchor=tk.W)
        
        # Save directory
        dir_frame = ttk.Frame(save_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dir_frame, text="Save Directory:", style='Header.TLabel').pack(anchor=tk.W)
        ttk.Entry(dir_frame, textvariable=self.save_directory).pack(fill=tk.X, pady=2)
        
        dir_buttons = ttk.Frame(save_frame)
        dir_buttons.pack(fill=tk.X)
        
        ttk.Button(dir_buttons, text="Browse", width=8,
                  command=self.select_save_directory).pack(side=tk.LEFT, padx=2)
        ttk.Button(dir_buttons, text="Open Folder", width=10,
                  command=self.open_save_directory).pack(side=tk.LEFT, padx=2)
        ttk.Button(dir_buttons, text="Reset Counter", width=12,
                  command=self.reset_frame_counter).pack(side=tk.LEFT, padx=2)
        
        # Status Section
        status_frame = ttk.LabelFrame(self.control_frame, text="Status", padding="5")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Status: Stopped",
                                    style='Header.TLabel')
        self.status_label.pack(pady=2)
        
        self.diff_label = ttk.Label(status_frame, text="Diff Score: 0.00%")
        self.diff_label.pack(pady=2)
        
        self.fps_label = ttk.Label(status_frame, text="FPS: 0")
        self.fps_label.pack(pady=2)
        
        self.frames_label = ttk.Label(status_frame, text="Frames Saved: 0")
        self.frames_label.pack(pady=2)
        
        # Performance Settings
        perf_frame = ttk.LabelFrame(self.control_frame, text="Performance", padding="5")
        perf_frame.pack(fill=tk.X)
        
        self.update_slider = SliderWithValue(
            perf_frame,
            text="Display Update Interval",
            variable=self.update_interval,
            from_=10, to=100,
            tooltip="Lower = smoother but more CPU",
            unit="ms"
        )
        self.update_slider.pack(fill=tk.X, pady=2)
        
        # Show difference overlay option
        ttk.Checkbutton(perf_frame, text="Show Difference Overlay",
                       variable=self.show_diff_overlay).pack(anchor=tk.W, pady=5)
        
    def get_available_cameras(self):
        """Get list of available camera devices"""
        camera_list = ['0']  # Default webcam
        
        # Try to detect additional cameras
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_list.append(str(i))
                cap.release()
        
        # Add option for video file
        camera_list.append("Video File")
        return camera_list
    
    def refresh_cameras(self):
        """Refresh the list of available cameras"""
        self.camera_combo['values'] = self.get_available_cameras()
    
    def select_save_directory(self):
        """Open directory selection dialog"""
        directory = filedialog.askdirectory(initialdir=self.save_directory.get())
        if directory:
            self.save_directory.set(directory)
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About", 
            "Frame Difference Detector\n\n"
            "A tool for detecting and saving frames with significant changes.\n\n"
            "Use the controls to adjust sensitivity and performance settings.")
    
    def reset_frame_counter(self):
        """Reset the saved frames counter"""
        self.frames_saved = 0
        self.frames_label.config(text="Frames Saved: 0")

    def open_save_directory(self):
        """Open the save directory in file explorer"""
        save_dir = self.save_directory.get()
        if os.path.exists(save_dir):
            os.startfile(save_dir) if os.name == 'nt' else os.system(f'open "{save_dir}"')
        else:
            messagebox.showwarning("Warning", "Save directory does not exist yet.")

    def get_save_path(self, frame_type="original"):
        """Get the save path for a frame based on current settings"""
        timestamp = datetime.now()
        
        # Create base directory
        base_dir = self.save_directory.get()
        
        # Add date folder if enabled
        if self.create_date_folders.get():
            date_folder = timestamp.strftime("%Y-%m-%d")
            base_dir = os.path.join(base_dir, date_folder)
        
        # Create directories if they don't exist
        os.makedirs(base_dir, exist_ok=True)
        
        # Create filename with frame number and timestamp
        filename = f"{frame_type}_frame_{self.frames_saved:05d}_{timestamp.strftime('%H%M%S_%f')}.jpg"
        
        return os.path.join(base_dir, filename)

    def load_settings(self):
        """Load settings from JSON file"""
        default_settings = {
            'threshold': 30,
            'min_area': 1,
            'frame_width': 640,
            'frame_height': 480,
            'save_frames': True,
            'save_original': True,
            'save_diff': False,
            'create_date_folders': True,
            'show_diff_overlay': True,
            'camera_source': '0',
            'save_directory': str(Path.home() / "frame_diff_detector"),
            'update_interval': 30
        }
        
        try:
            if os.path.exists('settings.json'):
                with open('settings.json', 'r') as f:
                    settings = json.load(f)
                    return {**default_settings, **settings}
        except Exception as e:
            print(f"Error loading settings: {e}")
        
        return default_settings

    def save_settings(self):
        """Save current settings to JSON file"""
        settings = {
            'threshold': self.threshold.get(),
            'min_area': self.min_area.get(),
            'frame_width': self.frame_width.get(),
            'frame_height': self.frame_height.get(),
            'save_frames': self.save_frames.get(),
            'save_original': self.save_original.get(),
            'save_diff': self.save_diff.get(),
            'create_date_folders': self.create_date_folders.get(),
            'show_diff_overlay': self.show_diff_overlay.get(),
            'camera_source': self.camera_source.get(),
            'save_directory': self.save_directory.get(),
            'update_interval': self.update_interval.get()
        }
        
        try:
            with open('settings.json', 'w') as f:
                json.dump(settings, f, indent=4)
            messagebox.showinfo("Success", "Settings saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def toggle_camera(self):
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        try:
            # Handle video file selection
            if self.camera_source.get() == "Video File":
                video_path = filedialog.askopenfilename(
                    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
                if not video_path:
                    return
                self.cap = cv2.VideoCapture(video_path)
            else:
                try:
                    self.cap = cv2.VideoCapture(int(self.camera_source.get()))
                except ValueError:
                    messagebox.showerror("Error", "Invalid camera source")
                    return
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Failed to open camera")
                return
            
            # Set frame size
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width.get())
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height.get())
            
            # Create save directory if needed
            if self.save_frames.get():
                os.makedirs(self.save_directory.get(), exist_ok=True)
            
            # Initialize detector with all parameters
            self.detector = FrameDiffDetector(
                threshold=self.threshold.get(),
                min_area_percentage=self.min_area.get(),
                noise_reduction=self.noise_reduction.get(),
                light_compensation=self.light_compensation.get()
            )
            
            self.is_running = True
            self.start_button.config(text="Stop Camera")
            self.update_status_label("status", "Status: Running", "green")
            
            # Reset error tracking
            self.error_count = 0
            self.last_error_time = None
            
            # Start video thread
            self.video_thread = threading.Thread(target=self.update_frame)
            self.video_thread.daemon = True
            self.video_thread.start()
            
            # Start GUI update
            self.update_gui()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
            self.stop_camera()
    
    def stop_camera(self):
        # Stop recording if active
        if self.recording:
            self.toggle_recording()
        
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        self.start_button.config(text="Start Camera")
        self.update_status_label("status", "Status: Stopped")
        
        # Clear queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
    
    def update_frame(self):
        while self.is_running:
            try:
                if not self.cap or not self.cap.isOpened():
                    self.handle_camera_error("Camera disconnected")
                    break

                ret, frame = self.cap.read()
                if not ret:
                    if self.camera_source.get() == "Video File":
                        self.stop_camera()  # End of video file
                        break
                    self.handle_camera_error("Failed to read frame")
                    continue

                with self.frame_update_lock:
                    # Resize frame if needed
                    if frame.shape[1] != self.frame_width.get() or frame.shape[0] != self.frame_height.get():
                        try:
                            frame = cv2.resize(frame, (self.frame_width.get(), self.frame_height.get()))
                        except Exception as e:
                            self.handle_camera_error(f"Resize error: {str(e)}")
                            continue

                    # Update detector parameters
                    if self.detector:
                        self.detector.threshold = int(self.threshold.get())
                        self.detector.min_area_percentage = self.min_area.get()
                        self.detector.noise_reduction = self.noise_reduction.get()
                        self.detector.light_compensation = self.light_compensation.get()

                        # Process frame
                        try:
                            diff_score, is_different, diff_frame, motion_info = self.detector.compute_frame_difference(frame)
                        except Exception as e:
                            self.handle_camera_error(f"Detection error: {str(e)}")
                            continue

                        # Save frames if enabled and different
                        if is_different and self.save_frames.get():
                            try:
                                if self.save_original.get():
                                    cv2.imwrite(self.get_save_path("original"), frame)
                                if self.save_diff.get():
                                    cv2.imwrite(self.get_save_path("diff"), diff_frame)
                                self.frames_saved += 1
                                self.update_status_label("frames", f"Frames Saved: {self.frames_saved}")
                            except Exception as e:
                                self.handle_camera_error(f"Save error: {str(e)}")

                        # Save video if recording
                        if self.recording and self.video_writer:
                            try:
                                self.video_writer.write(frame)
                            except Exception as e:
                                self.handle_camera_error(f"Recording error: {str(e)}")
                                self.toggle_recording()  # Stop recording on error

                        # Draw current zone if drawing
                        if self.is_drawing_zone and self.current_zone:
                            x, y, w, h = self.current_zone
                            cv2.rectangle(diff_frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

                        # Update display
                        display_frame = diff_frame if self.show_diff_overlay.get() else frame
                        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                        # Show motion direction if available
                        if motion_info and isinstance(motion_info, dict) and motion_info.get('direction'):
                            direction = motion_info['direction']
                            if direction != "None":
                                cv2.putText(display_frame, f"Motion: {direction}", 
                                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Update queue
                        try:
                            self.queue.put_nowait((display_frame, diff_score, is_different))
                        except queue.Full:
                            try:
                                self.queue.get_nowait()
                                self.queue.put_nowait((display_frame, diff_score, is_different))
                            except queue.Empty:
                                pass

            except Exception as e:
                self.handle_camera_error(f"Unexpected error: {str(e)}")
                continue

            # Add small delay to prevent CPU overload
            time.sleep(0.001)

    def update_gui(self):
        try:
            with self.gui_update_lock:
                # Get the latest frame
                display_frame, diff_score, is_different = self.queue.get_nowait()
                
                # Update FPS calculation
                current_time = datetime.now()
                time_diff = (current_time - self.last_frame_time).total_seconds()
                self.fps_update_counter += 1
                
                if time_diff >= 1.0:  # Update FPS every second
                    fps = self.fps_update_counter / time_diff
                    self.update_status_label("fps", f"FPS: {fps:.1f}")
                    self.fps_update_counter = 0
                    self.last_frame_time = current_time
                
                # Convert to PIL format
                try:
                    image = PIL.Image.fromarray(display_frame)
                    photo = PIL.ImageTk.PhotoImage(image=image)
                    
                    # Update GUI elements
                    self.video_label.config(image=photo)
                    self.video_label.image = photo  # Keep reference
                    self.update_status_label("diff", f"Diff Score: {diff_score:.2f}%", 
                                          color="red" if is_different else "black")
                except Exception as e:
                    self.handle_camera_error(f"Display error: {str(e)}")
                
        except queue.Empty:
            pass
        except Exception as e:
            self.handle_camera_error(f"GUI update error: {str(e)}")
        
        if self.is_running:
            self.window.after(self.update_interval.get(), self.update_gui)

    def handle_camera_error(self, error_msg):
        """Handle camera and processing errors"""
        current_time = datetime.now()
        
        # Reset error count if it's been more than 5 seconds since last error
        if self.last_error_time and (current_time - self.last_error_time).total_seconds() > 5:
            self.error_count = 0
        
        self.error_count += 1
        self.last_error_time = current_time
        
        # Update status with error message
        self.update_status_label("status", f"Error: {error_msg}", color="red")
        
        # Stop camera if too many errors occur in short time
        if self.error_count >= 5:
            self.stop_camera()
            messagebox.showerror("Error", f"Multiple errors occurred: {error_msg}\nCamera stopped.")
            self.error_count = 0

    def update_status_label(self, label_type, text, color="black"):
        """Thread-safe status label updates"""
        def update():
            if label_type == "status":
                self.status_label.config(text=text, foreground=color)
            elif label_type == "diff":
                self.diff_label.config(text=text, foreground=color)
            elif label_type == "fps":
                self.fps_label.config(text=text)
            elif label_type == "frames":
                self.frames_label.config(text=text)
        
        try:
            self.window.after(0, update)
        except Exception:
            pass  # Ignore if window is closed

    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to save settings before quitting?"):
            self.save_settings()
        self.stop_camera()
        self.window.destroy()

    def validate_number(self, value):
        """Validate that entry contains only numbers"""
        if value == "":
            return True
        try:
            int(value)
            return True
        except ValueError:
            return False
    
    def on_resolution_change(self, event):
        """Handle resolution preset selection"""
        resolution = self.resolution_var.get()
        if resolution == "Custom":
            return
        
        width, height = map(int, resolution.split()[0].split('x'))
        self.frame_width.set(width)
        self.frame_height.set(height)

    def start_zone_drawing(self):
        """Start drawing a detection zone"""
        self.is_drawing_zone = True
        self.update_status_label("status", "Status: Drawing Zone - Click and drag")
        
    def on_mouse_down(self, event):
        """Handle mouse down event for zone drawing"""
        if self.is_drawing_zone:
            self.start_point = (event.x, event.y)
            self.current_zone = None
            
    def on_mouse_move(self, event):
        """Handle mouse move event for zone drawing"""
        if self.is_drawing_zone and self.start_point:
            self.current_zone = (*self.start_point, 
                               event.x - self.start_point[0],
                               event.y - self.start_point[1])
            
    def on_mouse_up(self, event):
        """Handle mouse up event for zone drawing"""
        if self.is_drawing_zone and self.start_point:
            if self.current_zone:
                self.detection_zones.append(self.current_zone)
                if self.detector:
                    self.detector.add_detection_zone(*self.current_zone)
            self.is_drawing_zone = False
            self.start_point = None
            self.current_zone = None
            self.update_status_label("status", "Status: Zone Added")
            
    def clear_zones(self):
        """Clear all detection zones"""
        self.detection_zones = []
        if self.detector:
            self.detector.clear_detection_zones()
            
    def on_sensitivity_change(self, event):
        """Handle sensitivity preset change"""
        preset = self.sensitivity_var.get().lower()
        if preset != "custom" and self.detector:
            self.detector.set_sensitivity_preset(preset)
            # Update GUI controls to match preset
            settings = {
                'low': {'threshold': 50, 'min_area': 5, 'noise_reduction': 3},
                'medium': {'threshold': 30, 'min_area': 2, 'noise_reduction': 2},
                'high': {'threshold': 20, 'min_area': 1, 'noise_reduction': 1}
            }[preset]
            self.threshold.set(settings['threshold'])
            self.min_area.set(settings['min_area'])
            self.noise_reduction.set(settings['noise_reduction'])
            
    def toggle_recording(self):
        """Toggle video recording"""
        if not self.recording:
            # Start recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_directory.get(), f"recording_{timestamp}.avi")
            
            # Get current frame size
            frame_size = (self.frame_width.get(), self.frame_height.get())
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, frame_size)
            
            self.recording = True
            self.record_button.config(text="Stop Recording")
            self.update_status_label("status", "Status: Recording")
        else:
            # Stop recording
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            self.recording = False
            self.record_button.config(text="Start Recording")
            self.update_status_label("status", "Status: Recording Saved")

def main():
    root = tk.Tk()
    app = FrameDiffGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 