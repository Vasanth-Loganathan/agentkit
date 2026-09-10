import base64
import os

def encode_image(image_path: str) -> str:
    """Reads a local image file and converts it to a Base64 string."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
        
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')