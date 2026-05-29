#!/usr/bin/env python3
"""
Face Detection and Recognition Application
CODSOFT AI Internship - Task 5

This script uses OpenCV's pre-trained Viola-Jones Haar Cascade front-face classifier 
to detect faces in static images, highlight them with bounding boxes, and save outputs.
"""

import os
import sys

# Verify OpenCV is available before proceeding
try:
    import cv2
except ImportError:
    print("Error: The 'opencv-python' library is required to run Face Detection.")
    print("Please install it using: pip install opencv-python-headless pillow")
    sys.exit(1)


class FaceDetector:
    def __init__(self):
        # Load the pre-trained Frontal Face Haar Cascade XML included in OpenCV
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if not os.path.exists(cascade_path):
            raise FileNotFoundError("Error: Haar Cascade frontal face XML file not found inside OpenCV data folder.")
            
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        print("Haar Cascade Frontal Face Classifier loaded successfully!")

    def detect_and_save(self, image_path, output_path="detected_faces.jpg"):
        """Detects faces in an image, draws bounding boxes, and saves the result."""
        if not os.path.exists(image_path):
            return False, f"Error: Image file '{image_path}' does not exist."
            
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return False, "Error: Could not read image file."
                
            # Convert to grayscale (Haar Cascade operates on grayscale)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            # scaleFactor: how much the image size is reduced at each image scale
            # minNeighbors: how many neighbors each candidate rectangle should have to retain it
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            num_faces = len(faces)
            print(f"Detected {num_faces} face(s) in the image.")
            
            # Draw bounding boxes around detected faces
            for (x, y, w, h) in faces:
                # Color: Neon Green (B=0, G=255, R=0), Thickness: 3px
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                # Add text label (Optional academic decoration)
                cv2.putText(
                    img, 
                    "Face Detected", 
                    (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 0), 
                    2
                )
                
            # Save the processed image
            cv2.imwrite(output_path, img)
            return True, {"num_faces": num_faces, "output_path": output_path}
            
        except Exception as e:
            return False, f"An error occurred during face detection: {str(e)}"


def main():
    print("=" * 60)
    print("        👤 WELCOME TO THE FACE DETECTION AI SYSTEM 👤")
    print("             Created for the CODSOFT AI Internship")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\nUsage: python face_detector.py <path_to_image_file> [output_file_name]")
        print("\nNo image file specified in command line arguments.")
        image_path = input("Enter the absolute or relative path to a portrait image: ").strip()
        if not image_path:
            print("No path entered. Exiting...")
            sys.exit(0)
    else:
        image_path = sys.argv[1]
        
    output_path = sys.argv[2] if len(sys.argv) > 2 else "detected_faces.jpg"
    
    try:
        detector = FaceDetector()
        print(f"\nProcessing face detection on: '{image_path}'...")
        success, result = detector.detect_and_save(image_path, output_path)
        
        if success:
            print("\n" + "=" * 45)
            print("🎉 Success!")
            print(f"Face(s) Detected: {result['num_faces']}")
            print(f"Processed output image saved to: '{result['output_path']}'")
            print("=" * 45 + "\n")
        else:
            print(f"\nFailure: {result}")
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
