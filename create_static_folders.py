import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the necessary directory structure for static files."""
    base_dir = Path(__file__).parent
    
    # Create static directories
    static_dir = base_dir / 'static'
    css_dir = static_dir / 'css'
    js_dir = static_dir / 'js'
    images_dir = static_dir / 'images'
    
    # Create directories if they don't exist
    for directory in [static_dir, css_dir, js_dir, images_dir]:
        directory.mkdir(exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create sample images if they don't exist
    create_placeholder_image(images_dir / 'squat.png', "Squat")
    create_placeholder_image(images_dir / 'push_up.png', "Push Up")
    create_placeholder_image(images_dir / 'hammer_curl.png', "Hammer Curl")
    
    print("\nDirectory structure created successfully!")
    print(f"Static files should be placed in: {static_dir}")
    print("Make sure the following files exist:")
    print(f"  - {images_dir / 'squat.png'}")
    print(f"  - {images_dir / 'push_up.png'}")
    print(f"  - {images_dir / 'hammer_curl.png'}")

def create_placeholder_image(filepath, text="Exercise"):
    """Create a simple placeholder image using PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a blank image with a colored background
        img = Image.new('RGB', (200, 200), color=(73, 109, 137))
        d = ImageDraw.Draw(img)
        
        # Try to use a default font
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw text in the center of the image
        d.text((100, 100), text, fill=(255, 255, 255), anchor="mm", font=font)
        
        # Save the image
        img.save(filepath)
        print(f"Created placeholder image: {filepath}")
        
    except ImportError:
        # If PIL is not available, create an empty file
        print("PIL not installed. Installing empty image files.")
        with open(filepath, 'wb') as f:
            f.write(b'')
        print(f"Created empty file: {filepath}")

if __name__ == "__main__":
    create_directory_structure()
    print("\nRun 'pip install pillow' if you want to generate proper placeholder images.")
