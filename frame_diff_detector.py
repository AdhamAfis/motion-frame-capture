import cv2
import numpy as np
from datetime import datetime

class FrameDiffDetector:
    def __init__(self, threshold=30, min_area_percentage=1):
        """
        Initialize the frame difference detector
        
        Args:
            threshold (int): Threshold for pixel difference (0-255)
            min_area_percentage (float): Minimum percentage of changed pixels to consider frame as different
        """
        self.threshold = threshold
        self.min_area_percentage = min_area_percentage
        self.previous_frame = None
        
    def compute_frame_difference(self, current_frame):
        """
        Compute the difference between current frame and previous frame
        
        Returns:
            tuple: (difference_score, is_different, diff_frame)
        """
        # Convert frame to grayscale
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize if this is the first frame
        if self.previous_frame is None:
            self.previous_frame = gray
            return 0, False, current_frame
        
        # Calculate absolute difference between frames
        frame_diff = cv2.absdiff(self.previous_frame, gray)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of changed pixels
        diff_percentage = (np.count_nonzero(thresh) / thresh.size) * 100
        
        # Determine if frame is different based on threshold
        is_different = diff_percentage > self.min_area_percentage
        
        # Create visual difference frame
        diff_frame = current_frame.copy()
        diff_mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        diff_frame = cv2.addWeighted(diff_frame, 1, diff_mask, 0.5, 0)
        
        # Update previous frame
        self.previous_frame = gray
        
        return diff_percentage, is_different, diff_frame

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
        diff_score, is_different, diff_frame = detector.compute_frame_difference(frame)
        
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