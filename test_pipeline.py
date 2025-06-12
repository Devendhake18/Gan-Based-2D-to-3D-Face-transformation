import os
import cv2
import numpy as np
import mediapipe as mp
import shutil
from PIL import Image
import sys
from flask import Flask

# Create output directories
temp_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'pipeline_test')
os.makedirs(temp_dir, exist_ok=True)

app = Flask(__name__, static_folder='frontend', static_url_path='')

def test_face_crop(input_image_path):
    """Test the face crop functionality separately"""
    print("\n=== Testing Face Crop ===")
    output_dir = os.path.join(temp_dir, 'face_crop')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the input image
        print(f"Loading image: {input_image_path}")
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"ERROR: Could not load image {input_image_path}")
            return False
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize MediaPipe face mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
        
        # Detect facial landmarks
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            print("ERROR: No face detected in the image.")
            return False
        
        print("Face detected successfully!")
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape
        
        # Convert normalized landmarks to pixel coordinates
        landmarks = []
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmarks.append((x, y))
        
        # Compute convex hull
        landmarks_np = np.array(landmarks, dtype=np.int32)
        hull = cv2.convexHull(landmarks_np)
        
        # Create a black mask and fill the convex hull with white
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, (255, 255, 255))
        
        # Extract the face from original image using the mask
        face_only = cv2.bitwise_and(image, mask)
        
        # Crop the bounding box
        x, y, w_box, h_box = cv2.boundingRect(hull)
        face_crop = face_only[y:y+h_box, x:x+w_box]
        mask_crop = mask[y:y+h_box, x:x+w_box]
        
        # Convert to BGRA with alpha channel
        b, g, r = cv2.split(face_crop)
        alpha = cv2.cvtColor(mask_crop, cv2.COLOR_BGR2GRAY)
        result = cv2.merge((b, g, r, alpha))
        
        # Save transparent PNG to the output directory
        output_path = os.path.join(output_dir, "face_texture.png")
        cv2.imwrite(output_path, result)
        print(f"Saved face texture to: {output_path}")
        
        # Also save a JPEG version for compatibility with the morph step
        jpeg_path = os.path.join(output_dir, "face_texture.jpg")
        cv2.imwrite(jpeg_path, cv2.cvtColor(face_crop, cv2.COLOR_BGRA2BGR))
        print(f"Saved JPEG version to: {jpeg_path}")
        
        print("Face crop test successful!")
        return True
        
    except Exception as e:
        print(f"ERROR in face crop: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_morph(input_image_path):
    """Test the morph functionality separately"""
    print("\n=== Testing Morph ===")
    output_dir = os.path.join(temp_dir, 'morph')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from Morph import create_face_mesh
        
        print(f"Running morphing with texture: {input_image_path}")
        success, output_path = create_face_mesh(input_image_path, output_dir)
        
        if success:
            print(f"Morphing completed successfully!")
            print(f"Output path: {output_path}")
            return True
        else:
            print("Morphing failed!")
            return False
            
    except Exception as e:
        print(f"ERROR in morph: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== Pipeline Test ===")
    
    # Get input image path 
    if len(sys.argv) > 1:
        input_image_path = sys.argv[1]
    else:
        input_image_path = input("Enter path to test image: ")
    
    if not os.path.exists(input_image_path):
        print(f"ERROR: Image file not found: {input_image_path}")
        return
    
    # Test face crop
    if test_face_crop(input_image_path):
        # If face crop successful, test morph
        face_texture = os.path.join(temp_dir, 'face_crop', 'face_texture.jpg')
        if os.path.exists(face_texture):
            test_morph(face_texture)
    
    print("\nTest results can be found in:", temp_dir)

if __name__ == "__main__":
    main() 