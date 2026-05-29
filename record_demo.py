#!/usr/bin/env python3
"""
Automated Screen Recorder Utility
CODSOFT AI Internship - Portfolio Showcase Helper

This utility records a portion of your screen or the entire screen for a set duration,
downscales the video frames, and compresses them into a very small, lightweight MP4 file,
making it incredibly easy to upload to Google Drive or submission forms.
"""

import os
import sys
import time
from PIL import ImageGrab
import numpy as np

# Verify required libraries are installed
try:
    import cv2
except ImportError:
    print("Error: The 'opencv-python' library is required to run the screen recorder.")
    print("Please install it by running: pip install opencv-python pillow numpy")
    sys.exit(1)


def record_screen(duration_seconds=30, fps=10, output_name="portfolio_demo.mp4"):
    # Output path: save directly to their Desktop for easy access
    desktop_path = os.path.expanduser("~/Desktop")
    if os.path.exists(desktop_path):
        output_file = os.path.join(desktop_path, output_name)
    else:
        output_file = output_name

    print("=" * 60)
    print("        🎥 AUTOMATED SHOWCASE PORTFOLIO RECORDER 🎥")
    print("=" * 60)
    print(f"Goal: Record your screen for {duration_seconds} seconds.")
    print(f"Format: Highly-compressed MP4 (small file size guaranteed).")
    print(f"Saving to: '{output_file}'\n")

    # Countdown to let user switch to their web browser window
    print("Switch to your web browser showing the Streamlit app now!")
    for i in range(5, 0, -1):
        print(f"Starting in {i} seconds...", end="\r")
        time.sleep(1)
    print("\n🔴 RECORDING STARTED! Play with your Streamlit app now...")

    # Grab initial screen dimensions
    screen = ImageGrab.grab()
    width, height = screen.size

    # Set downscaling dimensions to guarantee a very small file size (e.g. max width 1024)
    scale_factor = min(1.0, 1024.0 / width)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Ensure width and height are even for MP4 codecs
    if new_width % 2 != 0:
        new_width += 1
    if new_height % 2 != 0:
        new_height += 1

    # Define video codec and create VideoWriter object
    # 'mp4v' is a standard, highly-compatible MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (new_width, new_height))

    start_time = time.time()
    frames_captured = 0
    interval = 1.0 / fps

    try:
        while (time.time() - start_time) < duration_seconds:
            loop_start = time.time()
            
            # Take screenshot
            screenshot = ImageGrab.grab()
            
            # Convert PIL image to numpy array (RGB)
            frame = np.array(screenshot)
            
            # Convert RGB to BGR (OpenCV uses BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Downscale frame for ultra-small file size
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Write frame to video file
            out.write(resized_frame)
            frames_captured += 1
            
            # Print countdown status
            elapsed = time.time() - start_time
            remaining = max(0.0, duration_seconds - elapsed)
            print(f"Recording... {int(remaining)}s left | Captured: {frames_captured} frames", end="\r")
            
            # Control frame rate
            sleep_time = interval - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nRecording stopped early by user.")

    finally:
        # Release everything
        out.release()
        print("\n\n🟢 RECORDING COMPLETED SUCCESSFULLY!")
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"File saved: '{output_file}'")
        print(f"File Size: {file_size_mb:.2f} MB (Extremely lightweight & easy to upload!)")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Record for 35 seconds at 10 FPS (plenty of time to demo all 5 tabs!)
    record_screen(duration_seconds=35, fps=10)
