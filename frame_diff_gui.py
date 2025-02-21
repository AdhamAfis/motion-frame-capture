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

class FrameDiffGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Frame Difference Detector")
        
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
        
        # Create GUI elements
        self.create_widgets()
        self.create_menu()
        
        # Initialize video capture and detector
        self.cap = None
        self.detector = None
        
        # Bind window close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Performance monitoring
        self.fps_label = ttk.Label(self.control_frame, text="FPS: 0")
        self.fps_label.pack(pady=5)
        self.last_frame_time = datetime.now()
        self.fps_update_counter = 0
        
        # Frame counter
        self.frames_saved = 0
        self.frames_label = ttk.Label(self.control_frame, text="Frames Saved: 0")
        self.frames_label.pack(pady=5)
        
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
        # Create main frames with scrollbar
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control frame with scrollbar
        control_scroll = ttk.Scrollbar(main_frame)
        control_scroll.pack(side=tk.LEFT, fill=tk.Y)
        
        self.control_frame = ttk.Frame(main_frame)
        self.control_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.Y)
        
        video_frame = ttk.Frame(main_frame)
        video_frame.pack(side=tk.LEFT, padx=10, pady=5, expand=True, fill=tk.BOTH)
        
        # Camera source selection
        ttk.Label(self.control_frame, text="Camera Source:").pack(pady=2)
        camera_frame = ttk.Frame(self.control_frame)
        camera_frame.pack(fill=tk.X, padx=5)
        
        self.camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_source)
        self.camera_combo['values'] = self.get_available_cameras()
        self.camera_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(camera_frame, text="Refresh", command=self.refresh_cameras).pack(side=tk.LEFT, padx=2)
        
        # Frame size controls
        size_frame = ttk.LabelFrame(self.control_frame, text="Frame Size")
        size_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(size_frame, text="Width:").pack()
        ttk.Entry(size_frame, textvariable=self.frame_width).pack(fill=tk.X, padx=5)
        ttk.Label(size_frame, text="Height:").pack()
        ttk.Entry(size_frame, textvariable=self.frame_height).pack(fill=tk.X, padx=5)
        
        # Detection settings
        settings_frame = ttk.LabelFrame(self.control_frame, text="Detection Settings")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Threshold:").pack()
        threshold_slider = ttk.Scale(settings_frame, from_=0, to=255,
                                   variable=self.threshold, orient=tk.HORIZONTAL)
        threshold_slider.pack(fill=tk.X, padx=5)
        
        ttk.Label(settings_frame, text="Min Area %:").pack()
        min_area_slider = ttk.Scale(settings_frame, from_=0, to=100,
                                  variable=self.min_area, orient=tk.HORIZONTAL)
        min_area_slider.pack(fill=tk.X, padx=5)
        
        # Performance settings
        perf_frame = ttk.LabelFrame(self.control_frame, text="Performance")
        perf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(perf_frame, text="Update Interval (ms):").pack()
        update_slider = ttk.Scale(perf_frame, from_=10, to=100,
                               variable=self.update_interval, orient=tk.HORIZONTAL)
        update_slider.pack(fill=tk.X, padx=5)
        
        # Save settings
        save_frame = ttk.LabelFrame(self.control_frame, text="Save Settings")
        save_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Save options
        save_options_frame = ttk.Frame(save_frame)
        save_options_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Checkbutton(save_options_frame, text="Enable Frame Saving",
                       variable=self.save_frames).pack(anchor=tk.W)
        ttk.Checkbutton(save_options_frame, text="Save Original Frames",
                       variable=self.save_original).pack(anchor=tk.W)
        ttk.Checkbutton(save_options_frame, text="Save Difference Frames",
                       variable=self.save_diff).pack(anchor=tk.W)
        ttk.Checkbutton(save_options_frame, text="Create Date Folders",
                       variable=self.create_date_folders).pack(anchor=tk.W)
        
        # Save directory
        save_dir_frame = ttk.Frame(save_frame)
        save_dir_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(save_dir_frame, text="Save Directory:").pack(anchor=tk.W)
        dir_entry = ttk.Entry(save_dir_frame, textvariable=self.save_directory)
        dir_entry.pack(fill=tk.X, expand=True, pady=2)
        
        dir_buttons_frame = ttk.Frame(save_dir_frame)
        dir_buttons_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(dir_buttons_frame, text="Browse",
                  command=self.select_save_directory).pack(side=tk.LEFT, padx=2)
        ttk.Button(dir_buttons_frame, text="Open Folder",
                  command=self.open_save_directory).pack(side=tk.LEFT, padx=2)
        ttk.Button(dir_buttons_frame, text="Reset Counter",
                  command=self.reset_frame_counter).pack(side=tk.LEFT, padx=2)
        
        # Control buttons
        self.start_button = ttk.Button(self.control_frame, text="Start",
                                     command=self.toggle_camera)
        self.start_button.pack(pady=10)
        
        # Status indicators
        status_frame = ttk.LabelFrame(self.control_frame, text="Status")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Status: Stopped")
        self.status_label.pack(pady=2)
        
        self.diff_label = ttk.Label(status_frame, text="Diff Score: 0.00%")
        self.diff_label.pack(pady=2)
        
        # Video display
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(expand=True)
        
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
            self.status_label.config(text="Status: Camera Error")
            return
        
        # Set frame size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width.get())
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height.get())
        
        # Create save directory if needed
        if self.save_frames.get():
            os.makedirs(self.save_directory.get(), exist_ok=True)
        
        self.detector = FrameDiffDetector(
            threshold=self.threshold.get(),
            min_area_percentage=self.min_area.get()
        )
        
        self.is_running = True
        self.start_button.config(text="Stop")
        self.status_label.config(text="Status: Running")
        
        # Start video thread
        self.video_thread = threading.Thread(target=self.update_frame)
        self.video_thread.daemon = True
        self.video_thread.start()
        
        # Start GUI update
        self.update_gui()
    
    def stop_camera(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        self.start_button.config(text="Start")
        self.status_label.config(text="Status: Stopped")
        
        # Clear queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
    
    def update_frame(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Resize frame if needed
                if frame.shape[1] != self.frame_width.get() or frame.shape[0] != self.frame_height.get():
                    frame = cv2.resize(frame, (self.frame_width.get(), self.frame_height.get()))
                
                # Update detector parameters
                self.detector.threshold = int(self.threshold.get())
                self.detector.min_area_percentage = self.min_area.get()
                
                # Process frame
                diff_score, is_different, diff_frame = self.detector.compute_frame_difference(frame)
                
                # Save frames if enabled and different
                if is_different and self.save_frames.get():
                    if self.save_original.get():
                        cv2.imwrite(self.get_save_path("original"), frame)
                    
                    if self.save_diff.get():
                        cv2.imwrite(self.get_save_path("diff"), diff_frame)
                    
                    self.frames_saved += 1
                    self.frames_label.config(text=f"Frames Saved: {self.frames_saved}")
                
                # Prepare display frame
                display_frame = diff_frame if self.show_diff_overlay.get() else frame
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Update queue (drop frames if full)
                try:
                    self.queue.put_nowait((display_frame, diff_score, is_different))
                except queue.Full:
                    try:
                        self.queue.get_nowait()
                        self.queue.put_nowait((display_frame, diff_score, is_different))
                    except queue.Empty:
                        pass
    
    def update_gui(self):
        try:
            # Get the latest frame
            display_frame, diff_score, is_different = self.queue.get_nowait()
            
            # Update FPS calculation
            current_time = datetime.now()
            time_diff = (current_time - self.last_frame_time).total_seconds()
            self.fps_update_counter += 1
            
            if time_diff >= 1.0:  # Update FPS every second
                fps = self.fps_update_counter / time_diff
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                self.fps_update_counter = 0
                self.last_frame_time = current_time
            
            # Convert to PIL format
            image = PIL.Image.fromarray(display_frame)
            photo = PIL.ImageTk.PhotoImage(image=image)
            
            # Update GUI elements
            self.video_label.config(image=photo)
            self.video_label.image = photo
            self.diff_label.config(
                text=f"Diff Score: {diff_score:.2f}%",
                foreground="red" if is_different else "black"
            )
            
        except queue.Empty:
            pass
        
        if self.is_running:
            self.window.after(self.update_interval.get(), self.update_gui)
    
    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to save settings before quitting?"):
            self.save_settings()
        self.stop_camera()
        self.window.destroy()

def main():
    root = tk.Tk()
    app = FrameDiffGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 