# Frame Difference Detector
A Python application for detecting and capturing frames with significant changes from video sources. This tool uses computer vision techniques to identify visual differences between consecutive frames and automatically saves frames when meaningful changes are detected.

## Features

### Core Functionality
- Real-time frame difference detection
- Support for multiple video sources (webcams and video files)
- Adjustable sensitivity settings
- Automatic frame saving with customizable options
- Live visualization of differences
- Performance monitoring and optimization

### Video Input
- Multiple camera support with automatic detection
- Video file support (mp4, avi, mov, mkv)
- Configurable frame dimensions
- Dynamic camera source switching
- Camera refresh capability

### Detection Settings
- Adjustable threshold for pixel difference sensitivity (0-255)
- Minimum area percentage for change detection (0-100%)
- Real-time parameter adjustment
- Visual feedback for detected changes

### Frame Saving
- Organized file structure with timestamp-based naming
- Optional date-based folder organization
- Separate saving of original and difference frames
- Sequential frame numbering
- Customizable save directory
- Automatic directory creation

### Performance Features
- Multi-threaded processing
- Frame dropping prevention
- Configurable update interval
- FPS monitoring
- Memory-efficient queue management

### User Interface
- Modern and intuitive GUI
- Real-time status indicators
- FPS display
- Frame counter
- Progress monitoring
- Customizable settings

## Installation

1. Clone the repository or download the source code
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Required dependencies:
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- Pillow >= 10.0.0

## Usage

### Starting the Application
```bash
python frame_diff_gui.py
```

### Main Controls

#### Camera Settings
- **Camera Source**: Select from available cameras or video files
- **Refresh**: Update the list of available cameras
- **Frame Size**: Adjust the width and height of captured frames

#### Detection Settings
- **Threshold**: Adjust pixel difference sensitivity
  - Higher values: Less sensitive to small changes
  - Lower values: More sensitive to subtle changes
- **Min Area %**: Set minimum percentage of changed pixels
  - Higher values: Only detect larger changes
  - Lower values: Detect smaller changes

#### Performance Settings
- **Update Interval**: Control GUI update frequency (10-100ms)
  - Lower values: Smoother display but higher CPU usage
  - Higher values: Lower CPU usage but less smooth display

#### Save Settings
- **Enable Frame Saving**: Toggle frame saving functionality
- **Save Original Frames**: Save unmodified frames
- **Save Difference Frames**: Save frames showing detected changes
- **Create Date Folders**: Organize saved frames by date
- **Save Directory**: Choose where to save detected frames
- **Reset Counter**: Reset the frame numbering sequence

### File Organization

The saved frames are organized in the following structure:
```
save_directory/
├── 2024-03-14/                     # Date folder (if enabled)
│   ├── original_frame_00001_123456.jpg
│   ├── diff_frame_00001_123456.jpg
│   ├── original_frame_00002_123457.jpg
│   └── diff_frame_00002_123457.jpg
└── 2024-03-15/
    ├── original_frame_00001_123456.jpg
    └── ...
```

### Status Indicators
- **FPS**: Current frames per second
- **Frames Saved**: Total number of saved frames
- **Diff Score**: Percentage of pixels changed
- **Status**: Current application state

### Menu Options

#### File Menu
- **Select Save Directory**: Choose save location
- **Save Settings**: Save current configuration
- **Exit**: Close application

#### View Menu
- **Show Difference Overlay**: Toggle difference visualization

#### Help Menu
- **About**: Application information

## Technical Details

### Frame Processing Pipeline
1. Frame Capture
   - Read frame from source
   - Resize to specified dimensions
   - Convert to grayscale for processing

2. Difference Detection
   - Compare with previous frame
   - Apply threshold to differences
   - Calculate percentage of changed pixels
   - Determine if change is significant

3. Frame Saving
   - Generate unique filename
   - Create necessary directories
   - Save original and/or difference frames
   - Update frame counter

4. Display Update
   - Convert frame to RGB
   - Apply difference overlay (if enabled)
   - Update GUI elements
   - Monitor performance

### Performance Optimization
- Limited queue size to prevent memory buildup
- Frame dropping when processing can't keep up
- Multi-threaded design for smooth operation
- Configurable update interval
- Efficient frame resizing

### Settings Management
- Automatic settings saving/loading
- JSON configuration file
- Default values for first-time use
- Settings persistence between sessions

## Troubleshooting

### Common Issues
1. **Camera Not Found**
   - Check camera connections
   - Click "Refresh" to update camera list
   - Try different camera source

2. **Performance Issues**
   - Increase update interval
   - Reduce frame size
   - Close other applications

3. **Save Errors**
   - Check write permissions
   - Verify save directory exists
   - Ensure sufficient disk space

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 