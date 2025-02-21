import cv2
import numpy as np
from datetime import datetime
from functools import lru_cache

class FrameDiffDetector:
    def __init__(self, threshold=30, min_area_percentage=1, noise_reduction=0, 
                 light_compensation=False, detection_zones=None):
        """
        Initialize the frame difference detector
        
        Args:
            threshold (int): Threshold for pixel difference (0-255)
            min_area_percentage (float): Minimum percentage of changed pixels
            noise_reduction (int): Noise reduction level (0-5)
            light_compensation (bool): Enable light change compensation
            detection_zones (list): List of (x, y, w, h) tuples for detection zones
        """
        self.threshold = threshold
        self.min_area_percentage = min_area_percentage
        self.noise_reduction = noise_reduction
        self.light_compensation = light_compensation
        self.detection_zones = detection_zones or []
        self.previous_frame = None
        self.motion_history = np.array([])
        
        # Cache for expensive operations
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self._kernel_cache = {}
        self._zone_masks = {}
        
    @lru_cache(maxsize=8)
    def _get_blur_kernel(self, noise_reduction):
        """Get cached Gaussian blur kernel"""
        size = 2 * noise_reduction + 1
        return (size, size), noise_reduction
    
    def _get_zone_mask(self, shape, zone):
        """Get cached zone mask"""
        key = (shape[0], shape[1], *zone)
        if key not in self._zone_masks:
            mask = np.zeros(shape, dtype=np.uint8)
            x, y, w, h = zone
            mask[y:y+h, x:x+w] = 1
            self._zone_masks[key] = mask
        return self._zone_masks[key]
    
    def apply_noise_reduction(self, frame):
        """Apply noise reduction to frame"""
        if self.noise_reduction > 0:
            kernel = self._get_blur_kernel(self.noise_reduction)
            return cv2.GaussianBlur(frame, *kernel)
        return frame
    
    def apply_light_compensation(self, frame):
        """Compensate for lighting changes"""
        if self.light_compensation:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = self._clahe.apply(l)
            return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
        return frame
    
    def detect_motion_direction(self, contours):
        """Detect overall motion direction from contours"""
        if not contours or len(self.motion_history) < 2:
            return "None"
        
        # Calculate centroid of all contours using numpy operations
        points = np.vstack([c.reshape(-1, 2) for c in contours])
        current_center = np.mean(points, axis=0)
        
        if len(self.motion_history) > 0:
            prev_center = self.motion_history[-1]
            dx = current_center[0] - prev_center[0]
            dy = current_center[1] - prev_center[1]
            
            # Update motion history efficiently
            if len(self.motion_history) >= 10:
                self.motion_history = np.roll(self.motion_history, -1, axis=0)
                self.motion_history[-1] = current_center
            else:
                self.motion_history = np.vstack([self.motion_history, current_center])
            
            # Determine direction
            angle = np.arctan2(dy, dx) * 180 / np.pi
            if abs(dx) < 5 and abs(dy) < 5:
                return "Static"
            if -45 <= angle <= 45:
                return "Right"
            elif 45 < angle <= 135:
                return "Down"
            elif -135 <= angle < -45:
                return "Up"
            else:
                return "Left"
        else:
            self.motion_history = np.array([current_center])
            return "None"
    
    def compute_frame_difference(self, current_frame):
        """
        Compute the difference between current frame and previous frame
        
        Returns:
            tuple: (difference_score, is_different, diff_frame, motion_info)
        """
        # Apply preprocessing
        if self.light_compensation:
            current_frame = self.apply_light_compensation(current_frame)
        
        # Convert frame to grayscale and apply noise reduction
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if self.noise_reduction > 0:
            gray = self.apply_noise_reduction(gray)
        
        # Initialize if this is the first frame
        if self.previous_frame is None:
            self.previous_frame = gray
            return 0, False, current_frame, {}
        
        # Calculate absolute difference between frames
        frame_diff = cv2.absdiff(self.previous_frame, gray)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Process detection zones efficiently
        zone_results = []
        if self.detection_zones:
            for i, zone in enumerate(self.detection_zones):
                mask = self._get_zone_mask(thresh.shape, zone)
                zone_thresh = thresh * mask
                zone_diff_percentage = (np.count_nonzero(zone_thresh) / (zone[2] * zone[3])) * 100
                zone_results.append({
                    'zone_id': i,
                    'diff_percentage': zone_diff_percentage,
                    'is_active': zone_diff_percentage > self.min_area_percentage
                })
        
        # Calculate overall difference percentage
        diff_percentage = (np.count_nonzero(thresh) / thresh.size) * 100
        is_different = diff_percentage > self.min_area_percentage
        
        # Only process contours if needed
        motion_direction = "None"
        if is_different:
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter small contours efficiently
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
            
            if contours:
                motion_direction = self.detect_motion_direction(contours)
        else:
            contours = []
        
        # Create visualization only if needed
        diff_frame = current_frame.copy()
        
        # Draw zones and motion tracking efficiently
        if self.detection_zones or (contours and is_different):
            # Draw zones
            for i, (x, y, w, h) in enumerate(self.detection_zones):
                color = (0, 0, 255) if zone_results[i]['is_active'] else (0, 255, 0)
                cv2.rectangle(diff_frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw motion tracking
            if contours and is_different:
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(diff_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                if motion_direction != "None":
                    cv2.putText(diff_frame, f"Motion: {motion_direction}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Create motion information dictionary
        motion_info = {
            'contours': contours,
            'direction': motion_direction,
            'zones': zone_results,
            'timestamp': datetime.now()
        }
        
        # Overlay difference mask efficiently
        if self.show_diff_overlay:
            diff_mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            cv2.addWeighted(diff_frame, 1, diff_mask, 0.3, 0, diff_frame)
        
        # Update previous frame
        self.previous_frame = gray
        
        return diff_percentage, is_different, diff_frame, motion_info
    
    def add_detection_zone(self, x, y, w, h):
        """Add a new detection zone"""
        self.detection_zones.append((x, y, w, h))
        # Clear zone mask cache
        self._zone_masks.clear()
    
    def clear_detection_zones(self):
        """Remove all detection zones"""
        self.detection_zones = []
        self._zone_masks.clear()
    
    def set_sensitivity_preset(self, preset):
        """Set detection sensitivity preset"""
        presets = {
            'low': {'threshold': 50, 'min_area': 5, 'noise_reduction': 3},
            'medium': {'threshold': 30, 'min_area': 2, 'noise_reduction': 2},
            'high': {'threshold': 20, 'min_area': 1, 'noise_reduction': 1}
        }
        if preset in presets:
            settings = presets[preset]
            self.threshold = settings['threshold']
            self.min_area_percentage = settings['min_area']
            self.noise_reduction = settings['noise_reduction']

def main():
    # Initialize video capture (0 for webcam, or provide video file path)
    cap = cv2.VideoCapture(0)
    
    # Initialize frame difference detector
    detector = FrameDiffDetector(threshold=30, min_area_percentage=1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Compute frame difference
        diff_score, is_different, diff_frame, motion_info = detector.compute_frame_difference(frame)
        
        # Display information on frame
        info_text = f"Diff Score: {diff_score:.2f}%"
        cv2.putText(diff_frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if is_different:
            cv2.putText(diff_frame, "MOTION DETECTED", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Save different frame (optional)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(f"diff_frame_{timestamp}.jpg", frame)
        
        # Display the frame
        cv2.imshow('Frame Difference Detection', diff_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 