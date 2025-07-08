# import cv2
# import numpy as np
# import math
# import argparse

# class LengthMeasurer:
#     def __init__(self, reference_width=None):
#         """
#         Initialize the length measurer
#         reference_width: Known width of reference object in real units (e.g., inches, cm)
#         """
#         self.reference_width = reference_width
#         self.pixels_per_unit = None
#         self.cap = None
        
#     def setup_camera(self, camera_id=0):
#         """Initialize camera capture"""
#         self.cap = cv2.VideoCapture(camera_id)
#         if not self.cap.isOpened():
#             raise Exception("Could not open camera")
        
#     def find_contours(self, frame):
#         """Find contours in the frame"""
#         # Convert to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Apply Gaussian blur
#         blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
#         # Edge detection
#         edged = cv2.Canny(blurred, 50, 100)
        
#         # Dilate and erode to close gaps
#         edged = cv2.dilate(edged, None, iterations=1)
#         edged = cv2.erode(edged, None, iterations=1)
        
#         # Find contours
#         contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         return contours, edged
    
#     def midpoint(self, ptA, ptB):
#         """Calculate midpoint between two points"""
#         return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
    
#     def calculate_distance(self, ptA, ptB):
#         """Calculate Euclidean distance between two points"""
#         return math.sqrt((ptA[0] - ptB[0]) ** 2 + (ptA[1] - ptB[1]) ** 2)
    
#     def get_bounding_box_dimensions(self, contour):
#         """Get dimensions of bounding box for a contour"""
#         # Get minimum area rectangle
#         rect = cv2.minAreaRect(contour)
#         box = cv2.boxPoints(rect)
#         box = np.array(box, dtype="int")
        
#         # Order the points
#         box = self.order_points(box)
        
#         # Calculate dimensions
#         (tl, tr, br, bl) = box
        
#         # Calculate width and height in pixels
#         width_pixels = self.calculate_distance(tl, tr)
#         height_pixels = self.calculate_distance(tr, br)
        
#         return box, width_pixels, height_pixels
    
#     def order_points(self, pts):
#         """Order points in top-left, top-right, bottom-right, bottom-left order"""
#         rect = np.zeros((4, 2), dtype="float32")
        
#         # Sum and difference to find corners
#         s = pts.sum(axis=1)
#         diff = np.diff(pts, axis=1)
        
#         rect[0] = pts[np.argmin(s)]      # top-left
#         rect[2] = pts[np.argmax(s)]      # bottom-right
#         rect[1] = pts[np.argmin(diff)]   # top-right
#         rect[3] = pts[np.argmax(diff)]   # bottom-left
        
#         return rect
    
#     def set_reference(self, frame):
#         """Set reference object for scale calibration"""
#         contours, _ = self.find_contours(frame)
        
#         if len(contours) == 0:
#             return False, "No objects found for reference"
        
#         # Find the largest contour (assuming it's our reference)
#         largest_contour = max(contours, key=cv2.contourArea)
        
#         if cv2.contourArea(largest_contour) < 1000:  # Minimum area threshold
#             return False, "Reference object too small"
        
#         # Get dimensions
#         _, width_pixels, _ = self.get_bounding_box_dimensions(largest_contour)
        
#         if self.reference_width is None:
#             return False, "Reference width not specified"
        
#         # Calculate pixels per unit
#         self.pixels_per_unit = width_pixels / self.reference_width
        
#         return True, f"Reference set: {width_pixels:.1f} pixels = {self.reference_width} units"
    
#     def measure_objects(self, frame):
#         """Measure all objects in the frame"""
#         if self.pixels_per_unit is None:
#             return frame, "No reference set. Press 'r' to set reference."
        
#         contours, edged = self.find_contours(frame)
        
#         # Create a copy of the frame for drawing
#         output = frame.copy()
        
#         measurements = []
        
#         for i, contour in enumerate(contours):
#             # Filter small contours
#             if cv2.contourArea(contour) < 1000:
#                 continue
            
#             # Get bounding box and dimensions
#             box, width_pixels, height_pixels = self.get_bounding_box_dimensions(contour)
            
#             # Convert to real-world units
#             width_real = width_pixels / self.pixels_per_unit
#             height_real = height_pixels / self.pixels_per_unit
            
#             # Draw the bounding box
#             cv2.drawContours(output, [box.astype("int")], -1, (0, 255, 0), 2)
            
#             # Draw measurements
#             (tl, tr, br, bl) = box
            
#             # Width measurement
#             cv2.line(output, tuple(tl.astype(int)), tuple(tr.astype(int)), (255, 0, 0), 2)
#             mid_top = self.midpoint(tl, tr)
#             cv2.putText(output, f"{width_real:.1f}", 
#                        (int(mid_top[0] - 15), int(mid_top[1] - 10)),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
#             # Height measurement
#             cv2.line(output, tuple(tr.astype(int)), tuple(br.astype(int)), (255, 0, 0), 2)
#             mid_right = self.midpoint(tr, br)
#             cv2.putText(output, f"{height_real:.1f}", 
#                        (int(mid_right[0] + 10), int(mid_right[1])),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
#             measurements.append({
#                 'object': i + 1,
#                 'width': width_real,
#                 'height': height_real,
#                 'area_pixels': cv2.contourArea(contour)
#             })
        
#         status = f"Found {len(measurements)} objects"
#         return output, status
    
#     def run(self):
#         """Main measurement loop"""
#         if self.cap is None:
#             self.setup_camera()
        
#         print("Object Length Measurement Tool")
#         print("Controls:")
#         print("- 'r': Set reference object (focus on object of known size)")
#         print("- 's': Save current frame")
#         print("- 'q': Quit")
#         print("- Space: Pause/Resume")
        
#         paused = False
        
#         while True:
#             if not paused:
#                 ret, frame = self.cap.read()
#                 if not ret:
#                     break
                
#                 # Flip frame horizontally for mirror effect
#                 frame = cv2.flip(frame, 1)
#                 current_frame = frame.copy()
            
#             # Process frame for measurements
#             output, status = self.measure_objects(current_frame)
            
#             # Add status text
#             cv2.putText(output, status, (10, 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
#             # Add instructions
#             if self.pixels_per_unit is None:
#                 cv2.putText(output, "Press 'r' to set reference object", (10, 60), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
#             cv2.imshow("Length Measurement", output)
            
#             key = cv2.waitKey(1) & 0xFF
            
#             if key == ord('q'):
#                 break
#             elif key == ord('r'):
#                 success, message = self.set_reference(current_frame)
#                 print(f"Reference: {message}")
#             elif key == ord('s'):
#                 cv2.imwrite(f"measurement_{cv2.getTickCount()}.jpg", output)
#                 print("Frame saved")
#             elif key == ord(' '):
#                 paused = not paused
#                 print("Paused" if paused else "Resumed")
    
#     def cleanup(self):
#         """Clean up resources"""
#         if self.cap is not None:
#             self.cap.release()
#         cv2.destroyAllWindows()

# def main():
#     parser = argparse.ArgumentParser(description='Measure object lengths using camera')
#     parser.add_argument('--reference-width', type=float, 
#                        help='Known width of reference object in your preferred units')
#     parser.add_argument('--camera', type=int, default=0, 
#                        help='Camera ID (default: 0)')
    
#     args = parser.parse_args()
    
#     # Get reference width if not provided
#     reference_width = args.reference_width
#     if reference_width is None:
#         try:
#             reference_width = float(input("Enter the width of your reference object (in cm/inches): "))
#         except ValueError:
#             print("Invalid input. Using default reference width of 2.5 cm")
#             reference_width = 2.5
    
#     # Create measurer
#     measurer = LengthMeasurer(reference_width=reference_width)
    
#     try:
#         measurer.setup_camera(args.camera)
#         measurer.run()
#     except KeyboardInterrupt:
#         print("\nStopping measurement tool...")
#     except Exception as e:
#         print(f"Error: {e}")
#     finally:
#         measurer.cleanup()

# if __name__ == "__main__":
#     main()







import cv2
import numpy as np
import math

class DepthEstimationMeasurer:
    def __init__(self):
        """
        Initialize the depth-based measurer
        Uses camera focal length and assumed distance for estimation
        """
        self.cap = None
        # Camera parameters (these are estimates - vary by camera)
        self.focal_length_mm = 4.0  # Typical smartphone camera focal length
        self.sensor_width_mm = 5.76  # Typical smartphone sensor width
        self.assumed_distance_m = 0.5  # Assume objects are 50cm from camera
        
    def setup_camera(self, camera_id=0):
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # Get camera resolution
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate pixels per mm on sensor
        self.pixels_per_mm = self.frame_width / self.sensor_width_mm
        
    def find_contours(self, frame):
        """Find contours in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, edged
    
    def estimate_real_size(self, pixel_size):
        """
        Estimate real-world size from pixel size
        Using similar triangles: real_size/distance = pixel_size/focal_length
        """
        # Convert pixel size to sensor size in mm
        sensor_size_mm = pixel_size / self.pixels_per_mm
        
        # Calculate real size using similar triangles
        real_size_mm = (sensor_size_mm * self.assumed_distance_m * 1000) / self.focal_length_mm
        
        # Convert to meters
        real_size_m = real_size_mm / 1000
        
        return real_size_m
    
    def get_bounding_box_dimensions(self, contour):
        """Get dimensions of bounding box for a contour"""
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")
        
        # Calculate width and height in pixels
        width_pixels = max(abs(box[0][0] - box[1][0]), abs(box[1][0] - box[2][0]))
        height_pixels = max(abs(box[0][1] - box[1][1]), abs(box[1][1] - box[2][1]))
        
        return box, width_pixels, height_pixels
    
    def measure_objects(self, frame):
        """Estimate measurements for all objects in the frame"""
        contours, edged = self.find_contours(frame)
        output = frame.copy()
        measurements = []
        
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 1000:
                continue
            
            # Get bounding box and dimensions
            box, width_pixels, height_pixels = self.get_bounding_box_dimensions(contour)
            
            # Estimate real-world dimensions
            width_m = self.estimate_real_size(width_pixels)
            height_m = self.estimate_real_size(height_pixels)
            
            # Draw the bounding box
            cv2.drawContours(output, [box.astype("int")], -1, (0, 255, 0), 2)
            
            # Draw measurements in meters
            center = np.mean(box, axis=0).astype(int)
            
            # Width measurement
            cv2.putText(output, f"W: {width_m:.3f}m", 
                       (center[0] - 50, center[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Height measurement
            cv2.putText(output, f"H: {height_m:.3f}m", 
                       (center[0] - 50, center[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            measurements.append({
                'object': i + 1,
                'width_m': width_m,
                'height_m': height_m,
                'width_pixels': width_pixels,
                'height_pixels': height_pixels
            })
        
        return output, measurements
    
    def adjust_distance(self, new_distance):
        """Adjust the assumed distance for better accuracy"""
        self.assumed_distance_m = new_distance
        print(f"Distance adjusted to {new_distance:.2f}m")
    
    def run(self):
        """Main measurement loop"""
        if self.cap is None:
            self.setup_camera()
        
        print("Depth-Based Length Estimation Tool")
        print("Controls:")
        print("- '+': Increase assumed distance")
        print("- '-': Decrease assumed distance")
        print("- 'd': Display current distance")
        print("- 's': Save current frame")
        print("- 'q': Quit")
        print(f"Current assumed distance: {self.assumed_distance_m}m")
        print("Note: Measurements are estimates and may not be accurate!")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Estimate measurements
            output, measurements = self.measure_objects(frame)
            
            # Add status information
            cv2.putText(output, f"Distance: {self.assumed_distance_m:.2f}m", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(output, "ESTIMATES ONLY - Use +/- to adjust", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.imshow("Depth Estimation", output)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                self.adjust_distance(min(self.assumed_distance_m + 0.1, 3.0))
            elif key == ord('-') or key == ord('_'):
                self.adjust_distance(max(self.assumed_distance_m - 0.1, 0.1))
            elif key == ord('d'):
                print(f"Current distance: {self.assumed_distance_m:.2f}m")
            elif key == ord('s'):
                cv2.imwrite(f"estimation_{cv2.getTickCount()}.jpg", output)
                print("Frame saved")
                for m in measurements:
                    print(f"Object {m['object']}: {m['width_m']:.3f}m x {m['height_m']:.3f}m")
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

# Alternative approach using human body proportions
class ProportionBasedMeasurer:
    def __init__(self):
        self.cap = None
        # Average human proportions (in meters)
        self.human_proportions = {
            'head_height': 0.23,  # Average head height
            'hand_width': 0.18,   # Average hand span
            'foot_length': 0.26,  # Average foot length
            'face_width': 0.14    # Average face width
        }
    
    def setup_camera(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
    
    def detect_human_features(self, frame):
        """Detect human features for scale reference"""
        # Load pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        return faces
    
    def run_with_human_reference(self):
        """Measure using human proportions as reference"""
        if self.cap is None:
            self.setup_camera()
        
        print("Human Proportion-Based Measurement")
        print("Show your face in the camera for scale reference")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            faces = self.detect_human_features(frame)
            
            output = frame.copy()
            
            if len(faces) > 0:
                # Use first detected face
                (x, y, w, h) = faces[0]
                
                # Calculate pixels per meter using face width
                pixels_per_meter = w / self.human_proportions['face_width']
                
                # Draw face rectangle
                cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(output, f"Face: {self.human_proportions['face_width']:.2f}m", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Now detect and measure other objects
                contours, _ = self.find_contours(frame)
                
                for contour in contours:
                    if cv2.contourArea(contour) < 1000:
                        continue
                    
                    # Get bounding box
                    x2, y2, w2, h2 = cv2.boundingRect(contour)
                    
                    # Skip if this is likely the face
                    if abs(x2 - x) < 50 and abs(y2 - y) < 50:
                        continue
                    
                    # Calculate real dimensions
                    width_m = w2 / pixels_per_meter
                    height_m = h2 / pixels_per_meter
                    
                    # Draw measurements
                    cv2.rectangle(output, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)
                    cv2.putText(output, f"{width_m:.3f}m x {height_m:.3f}m", 
                               (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                cv2.putText(output, "Scale set from face", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(output, "Show face for scale reference", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Human Proportion Measurement", output)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def find_contours(self, frame):
        """Find contours in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, edged
    
    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    print("Choose measurement method:")
    print("1. Depth estimation (rough estimates)")
    print("2. Human face reference (more accurate)")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        measurer = DepthEstimationMeasurer()
        try:
            measurer.setup_camera()
            measurer.run()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            measurer.cleanup()
    
    elif choice == "2":
        measurer = ProportionBasedMeasurer()
        try:
            measurer.setup_camera()
            measurer.run_with_human_reference()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            measurer.cleanup()
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()