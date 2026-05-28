#!/usr/bin/env python3
"""
Image Captioning Generator
CODSOFT AI Internship - Task 3

This script uses the pre-trained Hugging Face BLIP (Bootstrapped Language-Image Pre-training) 
Transformer model to generate highly accurate captions for any input image.
"""

import os
import sys
from PIL import Image

# Ensure required libraries are available before proceeding
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ImportError:
    print("Error: The 'transformers' library is required to run the BLIP model.")
    print("Please install it using: pip install transformers torch torchvision pillow")
    sys.exit(1)


class ImageCaptioner:
    def __init__(self):
        print("Loading pre-trained BLIP Transformer model (this may take a minute on the first run)...")
        # Load the processor and model from Hugging Face hub
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("Model loaded successfully!")

    def generate(self, image_path):
        """Generates a caption for the image at the specified path."""
        if not os.path.exists(image_path):
            return f"Error: Image file '{image_path}' does not exist."
            
        try:
            # Load and convert image to RGB
            raw_image = Image.open(image_path).convert('RGB')
            
            # Preprocess the image
            inputs = self.processor(raw_image, return_tensors="pt")
            
            # Generate the caption using greedy decoding
            out = self.model.generate(**inputs, max_new_tokens=40)
            
            # Decode the generated tokens to text
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Capitalize the first letter and return
            return caption.strip().capitalize()
            
        except Exception as e:
            return f"Error processing image: {str(e)}"


def main():
    print("=" * 60)
    print("         📸 WELCOME TO IMAGE CAPTIONING AI SYSTEM 📸")
    print("             Created for the CODSOFT AI Internship")
    print("=" * 60)
    
    # Check if an image path was provided as a command-line argument
    if len(sys.argv) < 2:
        print("\nUsage: python caption_generator.py <path_to_image_file>")
        print("\nNo image file provided. We will wait for you to specify one.")
        image_path = input("Enter the absolute or relative path to an image: ").strip()
        if not image_path:
            print("No path entered. Exiting...")
            sys.exit(0)
    else:
        image_path = sys.argv[1]

    # Instantiate model and generate
    try:
        captioner = ImageCaptioner()
        print(f"\nGenerating caption for image: '{image_path}'...")
        caption = captioner.generate(image_path)
        print("\n" + "=" * 40)
        print(f"Generated Caption:\n> {caption}")
        print("=" * 40 + "\n")
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")


if __name__ == "__main__":
    main()
