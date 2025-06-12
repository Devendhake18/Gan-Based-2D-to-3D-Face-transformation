from flask import Flask, request, jsonify, send_file, send_from_directory
import os
import uuid
import subprocess
import time
from werkzeug.utils import secure_filename
import shutil
from flask_cors import CORS
import tempfile  # Import tempfile module
import ctypes  # For admin check
import torch
import numpy as np
from PIL import Image
import base64
import io
import json
import logging
import traceback
import zipfile
from io import BytesIO

# Import pipeline components
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from check_models import check_and_setup_models

# Function to check if running as admin
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

# Show admin warning
if not is_admin():
    print("\n" + "="*80)
    print("WARNING: Application is not running with administrator privileges.")
    print("You may encounter permission issues with file operations.")
    print("Consider running the application as administrator.")
    print("="*80 + "\n")

# Create base temp directory using current user's documents folder which should have write permissions
user_documents = os.path.join(os.path.expanduser('~'), 'Documents')
BASE_TEMP_DIR = os.path.join(user_documents, 'face_generation_temp')
os.makedirs(BASE_TEMP_DIR, exist_ok=True)
print(f"Using user documents directory for temporary files: {BASE_TEMP_DIR}")

app = Flask(__name__, static_folder='frontend/build', static_url_path='/')
CORS(app)  # Enable CORS for all routes

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_TEMP_DIR, 'temp_uploads')
app.config['STYLEGAN_OUTPUT'] = os.path.join(BASE_TEMP_DIR, 'stylegan_output')
app.config['SRGAN_OUTPUT'] = os.path.join(BASE_TEMP_DIR, 'srgan_output')
app.config['FACE_CROP_OUTPUT'] = os.path.join(BASE_TEMP_DIR, 'face_crop_output')
app.config['MORPH_OUTPUT'] = os.path.join(BASE_TEMP_DIR, 'morph_output')
app.config['FINAL_OUTPUT'] = 'final_output'  # Keep this in the application directory for access
app.config['TEMP_FOLDER'] = 'temp'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STYLEGAN_OUTPUT'], exist_ok=True)
os.makedirs(app.config['SRGAN_OUTPUT'], exist_ok=True)
os.makedirs(app.config['FACE_CROP_OUTPUT'], exist_ok=True)
os.makedirs(app.config['MORPH_OUTPUT'], exist_ok=True)
os.makedirs(app.config['FINAL_OUTPUT'], exist_ok=True)
os.makedirs('epochs', exist_ok=True)
os.makedirs('temp_face_crop', exist_ok=True)  # Temporary directory for face cropping if permissions fail
os.makedirs('temp_morph_output', exist_ok=True)  # Temporary directory for morphing if permissions fail

# Check directory permissions
def check_directory_permissions():
    """Check write permissions for critical directories"""
    directories = [
        app.config['UPLOAD_FOLDER'],
        app.config['STYLEGAN_OUTPUT'],
        app.config['SRGAN_OUTPUT'],
        app.config['FACE_CROP_OUTPUT'],
        app.config['MORPH_OUTPUT'],
        app.config['FINAL_OUTPUT']
    ]
    
    all_ok = True
    
    for directory in directories:
        test_file = os.path.join(directory, "test_permissions.txt")
        try:
            with open(test_file, 'w') as f:
                f.write("testing write permissions")
            os.remove(test_file)
            print(f"✓ Directory has write permissions: {directory}")
        except Exception as e:
            print(f"✗ Permission error on directory {directory}: {e}")
            print(f"  Operations involving this directory may fail!")
            all_ok = False
    
    if all_ok:
        print("All directories have proper write permissions.")
    else:
        print("WARNING: Some directories have permission issues. Application may not function correctly.")
        
    return all_ok

# Check permissions at startup
check_directory_permissions()

# Check for model files
models_ok = check_and_setup_models()
if not models_ok:
    print("WARNING: Some model files are missing. The application may not work correctly.")

# Utility functions
def generate_unique_id():
    return str(uuid.uuid4())

def cleanup_temp_files(job_id):
    # Remove temporary files after processing
    job_path = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    if os.path.exists(job_path):
        shutil.rmtree(job_path)

# Model paths
app.config['MODELS'] = {
    'ffhq': os.path.join('Stylegan Model', 'ffhq', 'network-snapshot-000160.pkl'),
    'celeba': os.path.join('Stylegan Model', 'celeba', 'network-snapshot-000160.pkl')
}

# Pipeline step 1: Generate StyleGAN image
def generate_stylegan_image(job_id, seed=0, model_type='ffhq'):
    output_dir = os.path.join(app.config['STYLEGAN_OUTPUT'], job_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find an available model
    available_models = {}
    for key, path in app.config['MODELS'].items():
        if os.path.exists(path):
            available_models[key] = path
            print(f"Found model: {key} at {path}")
    
    # Check if requested model exists
    if model_type not in available_models:
        if not available_models:
            # No models available
            error_msg = f"No StyleGAN models found. Please install at least one model."
            print(f"Error: {error_msg}")
            return create_error_image(output_dir, seed, error_msg)
        
        # Use first available model as fallback
        fallback_model = list(available_models.keys())[0]
        print(f"Requested model '{model_type}' not found, using '{fallback_model}' as fallback")
        model_type = fallback_model
    
    # Use the model path
    model_path = available_models[model_type]
    
    # Add flag to use CPU if CUDA is not available
    command = [
        "python", "stylegan3/gen_images.py",
        f"--outdir={output_dir}",
        "--trunc=1",
        f"--seeds={seed}",
        f"--network={model_path}"
    ]
    
    # Try to detect if CUDA is available
    if not torch.cuda.is_available():
        command.append("--device=cpu")
    
    try:
        print(f"Running StyleGAN with model: {model_type}, seed: {seed}")
        subprocess.run(command, check=True)
        
        # Return the path to the generated image
        generated_image_path = os.path.join(output_dir, f"seed{seed:04d}.png")
        if os.path.exists(generated_image_path):
            # Copy the file to the final output directory
            final_output_dir = os.path.join(app.config['FINAL_OUTPUT'], job_id)
            os.makedirs(final_output_dir, exist_ok=True)
            
            final_image_path = os.path.join(final_output_dir, f"seed{seed:04d}.png")
            shutil.copy(generated_image_path, final_image_path)
            print(f"Copied StyleGAN image to: {final_image_path}")
            
            return generated_image_path
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running StyleGAN: {e}")
        # Create a fallback image with an error message
        return create_error_image(output_dir, seed, str(e))

def create_error_image(output_dir, seed, error_message):
    """Create a placeholder image when StyleGAN fails"""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a blank image
    img = Image.new('RGB', (1024, 1024), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Add error text
    draw.text((50, 50), f"Error generating image with seed {seed}", fill=(0, 0, 0))
    draw.text((50, 100), error_message, fill=(255, 0, 0))
    draw.text((50, 150), "Please check if StyleGAN model is properly configured", fill=(0, 0, 0))
    
    # Save the image
    filename = f"error_{seed}.png"
    error_path = os.path.join(output_dir, filename)
    img.save(error_path)
    
    # Extract job_id from the output_dir
    job_id = os.path.basename(output_dir)
    
    # Copy to final output
    try:
        final_output_dir = os.path.join(app.config['FINAL_OUTPUT'], job_id)
        os.makedirs(final_output_dir, exist_ok=True)
        
        final_error_path = os.path.join(final_output_dir, filename)
        shutil.copy(error_path, final_error_path)
        print(f"Copied error image to: {final_error_path}")
        
        return error_path
    except Exception as e:
        print(f"Error copying error image: {e}")
        return error_path

# Pipeline step 2: Apply SRGAN upscaling
def apply_srgan(input_image_path, job_id):
    output_dir = os.path.join(app.config['SRGAN_OUTPUT'], job_id)
    os.makedirs(output_dir, exist_ok=True)
    
    output_image_path = os.path.join(output_dir, "upscaled.jpg")
    
    try:
        # Import shutil here to ensure it's available
        import shutil
        
        # Import necessary libraries
        import torch
        import torch.nn as nn
        import math
        from PIL import Image
        from torchvision.transforms import ToTensor, ToPILImage
        
        # Define the Generator model inline to avoid import issues
        class Generator(nn.Module):
            def __init__(self, scale_factor):
                upsample_block_num = int(math.log(scale_factor, 2))

                super(Generator, self).__init__()
                self.block1 = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=9, padding=4),
                    nn.PReLU()
                )
                self.block2 = ResidualBlock(64)
                self.block3 = ResidualBlock(64)
                self.block4 = ResidualBlock(64)
                self.block5 = ResidualBlock(64)
                self.block6 = ResidualBlock(64)
                self.block7 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64)
                )
                block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
                block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
                self.block8 = nn.Sequential(*block8)

            def forward(self, x):
                block1 = self.block1(x)
                block2 = self.block2(block1)
                block3 = self.block3(block2)
                block4 = self.block4(block3)
                block5 = self.block5(block4)
                block6 = self.block6(block5)
                block7 = self.block7(block6)
                block8 = self.block8(block1 + block7)

                return (torch.tanh(block8) + 1) / 2

        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super(ResidualBlock, self).__init__()
                self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(channels)
                self.prelu = nn.PReLU()
                self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(channels)

            def forward(self, x):
                residual = self.conv1(x)
                residual = self.bn1(residual)
                residual = self.prelu(residual)
                residual = self.conv2(residual)
                residual = self.bn2(residual)

                return x + residual

        class UpsampleBLock(nn.Module):
            def __init__(self, in_channels, up_scale):
                super(UpsampleBLock, self).__init__()
                self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
                self.pixel_shuffle = nn.PixelShuffle(up_scale)
                self.prelu = nn.PReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.pixel_shuffle(x)
                x = self.prelu(x)
                return x
        
        # Look for model file in multiple locations
        model_paths = [
            'epochs/netG_epoch_4_49.pth',  # Original path
            'SRGAN Model/netG_epoch_4_100.pth',  # User's path
            'D:/CODDING STUFF/Sem 6/CV_GAN/Backend/SRGAN Model/netG_epoch_4_100.pth'  # Absolute path
        ]
        
        # Find the first available model file
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Found SRGAN model at: {model_path}")
                break
        
        if model_path is None:
            print("No SRGAN model file found in any of the expected locations")
            raise FileNotFoundError("SRGAN model file not found")
        
        # Setup SRGAN model
        upscale_factor = 4  # Upscale factor of 4
        model = Generator(upscale_factor).eval()
        
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"SRGAN running on: {device}")
        
        # Load the model weights
        try:
            if device.type == 'cuda':
                model = model.to(device)
                model.load_state_dict(torch.load(model_path))
            else:
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                
            # Process the image - ensure we use the full path
            print(f"Loading image for SRGAN: {input_image_path}")
            image = Image.open(input_image_path)
            image_tensor = ToTensor()(image).unsqueeze(0)
            if device.type == 'cuda':
                image_tensor = image_tensor.to(device)
            
            # Use torch.no_grad() for inference
            with torch.no_grad():
                output = model(image_tensor)
            
            # Convert output tensor to image and save
            out_img = ToPILImage()(output[0].cpu())
            out_img.save(output_image_path)
            
            # Copy to final output directory
            final_output_dir = os.path.join(app.config['FINAL_OUTPUT'], job_id)
            os.makedirs(final_output_dir, exist_ok=True)
            
            final_image_path = os.path.join(final_output_dir, "upscaled.jpg")
            shutil.copy(output_image_path, final_image_path)
            print(f"Copied SRGAN upscaled image to: {final_image_path}")
            
            return output_image_path
            
        except Exception as model_error:
            print(f"Error loading or running SRGAN model: {model_error}")
            raise
        
    except Exception as e:
        print(f"Error in SRGAN upscaling: {e}")
        
        # Create a fallback image (just copy the input image)
        import shutil
        try:
            shutil.copy(input_image_path, output_image_path)
            
            # Also copy to final output
            final_output_dir = os.path.join(app.config['FINAL_OUTPUT'], job_id)
            os.makedirs(final_output_dir, exist_ok=True)
            final_image_path = os.path.join(final_output_dir, "upscaled.jpg")
            shutil.copy(output_image_path, final_image_path)
            
            print(f"Using original image as fallback for SRGAN")
            return output_image_path
        except Exception as copy_error:
            print(f"Error copying original image: {copy_error}")
            # If copying fails, create an error image
            error_filename = create_error_image(output_dir, "srgan_error", str(e))
            
            # Return the full path
            return os.path.join(output_dir, error_filename)

# Pipeline step 3: Crop face
def crop_face(input_image_path, job_id):
    try:
        # Check if input image exists
        if not os.path.exists(input_image_path):
            print(f"Error: Input image does not exist: {input_image_path}")
            return None
            
        # Create a directory in user's Documents folder with a simple name
        simple_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'face_mesh_temp')
        
        # Make sure the directory exists with loose permissions
        if not os.path.exists(simple_dir):
            os.makedirs(simple_dir, exist_ok=True)
            
            # Try to set looser permissions on Windows
            try:
                import stat
                os.chmod(simple_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # Full permissions for everyone
            except:
                pass  # Ignore errors setting permissions
                
        print(f"Using directory for face crop: {simple_dir}")
        
        # Import required libraries for face cropping
        import cv2
        import numpy as np
        import mediapipe as mp
        
        # Load the input image
        print(f"Loading image for face crop: {input_image_path}")
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Error: Could not load image {input_image_path}")
            return None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize MediaPipe face mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
        
        # Detect facial landmarks
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            print("No face detected in the image.")
            return None
        
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
        output_path = os.path.join(simple_dir, "face_texture.png")
        cv2.imwrite(output_path, result)
        print(f"Saved face texture to: {output_path}")
        
        # Since we don't have a mesh creation function, we'll create a simple placeholder
        mesh_path = os.path.join(simple_dir, "face_mesh.obj")
        with open(mesh_path, 'w') as f:
            f.write("# Face mesh placeholder\n")
            f.write("# This file will be replaced by proper 3D morphing in the next step\n")
        
        # Create a simple MTL file
        mtl_path = os.path.join(simple_dir, "face_mesh.mtl")
        with open(mtl_path, 'w') as f:
            f.write("# MTL file for face mesh\n")
            f.write("newmtl material0\n")
            f.write("Ka 1.000000 1.000000 1.000000\n")
            f.write("Kd 1.000000 1.000000 1.000000\n")
            f.write("Ks 0.000000 0.000000 0.000000\n")
            f.write("Ns 10.000000\n")
            f.write("d 1.000000\n")
            f.write("map_Kd face_texture.png\n")
        
        # Also save a JPEG version for compatibility with the morph step
        jpeg_path = os.path.join(simple_dir, "face_texture.jpg")
        cv2.imwrite(jpeg_path, cv2.cvtColor(face_crop, cv2.COLOR_BGRA2BGR))
        
        # Copy output to the final output directory
        try:
            final_output_dir = os.path.join(app.config['FINAL_OUTPUT'], job_id)
            os.makedirs(final_output_dir, exist_ok=True)
            
            for src_file, filename in [(output_path, "face_texture.png"), 
                                     (mesh_path, "face_mesh.obj"), 
                                     (mtl_path, "face_mesh.mtl"),
                                     (jpeg_path, "face_texture.jpg")]:
                if os.path.exists(src_file):
                    dest_file = os.path.join(final_output_dir, filename)
                    shutil.copy(src_file, dest_file)
                    print(f"Copied {filename} to final output directory")
            
            final_mesh_path = os.path.join(final_output_dir, "face_mesh.obj")
            return final_mesh_path
        except Exception as copy_error:
            print(f"Error copying to final output: {copy_error}")
            return mesh_path
    
    except Exception as e:
        print(f"Error in face crop step: {e}")
        import traceback
        traceback.print_exc()  # Print full exception details
        return None

# Pipeline step 4: Apply morphing/3D processing
def apply_morph(input_mesh_path, job_id):
    try:
        # Check if input mesh exists
        if not os.path.exists(input_mesh_path):
            print(f"Error: Input mesh does not exist: {input_mesh_path}")
            return None
            
        # Create a directory in user's Documents folder with a simple name
        simple_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'morph_temp')
        
        # Make sure the directory exists with loose permissions
        if not os.path.exists(simple_dir):
            os.makedirs(simple_dir, exist_ok=True)
            
            # Try to set looser permissions on Windows
            try:
                import stat
                os.chmod(simple_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # Full permissions for everyone
            except:
                pass  # Ignore errors setting permissions
                
        print(f"Using directory for morphing: {simple_dir}")
        
        # Import morph module
        from Morph import create_face_mesh
        
        # Get the paths from the previous step
        input_dir = os.path.dirname(input_mesh_path)
        texture_path = None
        
        # Look for texture files in several possible formats
        for ext in ['jpg', 'png']:
            texture_file = os.path.join(input_dir, f"face_texture.{ext}")
            if os.path.exists(texture_file):
                texture_path = texture_file
                break
        
        if not texture_path:
            print(f"ERROR: Could not find texture file in {input_dir}")
            return None
        
        print(f"Found texture at: {texture_path}")
        print(f"Running morphing with texture: {texture_path}")
        
        # Process the face with morphing
        success, output_path = create_face_mesh(texture_path, simple_dir)
        
        if success:
            print(f"Morphing completed successfully in: {simple_dir}")
            
            # Copy output to the final output directory
            try:
                final_output_dir = os.path.join(app.config['FINAL_OUTPUT'], job_id)
                os.makedirs(final_output_dir, exist_ok=True)
                
                for file in os.listdir(simple_dir):
                    src_file = os.path.join(simple_dir, file)
                    dst_file = os.path.join(final_output_dir, file)
                    shutil.copy(src_file, dst_file)
                    print(f"Copied {file} to final output directory")
                
                final_mesh_path = os.path.join(final_output_dir, "face_mesh.obj")
                return final_mesh_path
            except Exception as copy_error:
                print(f"Error copying morphed files to final output: {copy_error}")
                # If copying to final output fails, return the temp path
                return os.path.join(simple_dir, "face_mesh.obj")
        else:
            print(f"Morphing failed, returned status: {success}")
            return None
    
    except Exception as e:
        print(f"Error in morph step: {e}")
        import traceback
        traceback.print_exc()  # Print full exception details
        return None

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# API endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Backend is running"}), 200

@app.route('/api/generate', methods=['POST'])
def generate_model():
    try:
        job_id = generate_unique_id()
        
        # Get parameters from request, with defaults
        data = request.get_json(silent=True) or {}
        seed = data.get('seed', 0)
        model_type = data.get('model_type', 'ffhq')
        
        # Step 1: Generate StyleGAN image
        stylegan_image_path = generate_stylegan_image(job_id, seed, model_type)
        if not stylegan_image_path:
            return jsonify({"error": "Failed to generate StyleGAN image"}), 500
        
        stylegan_filename = os.path.basename(stylegan_image_path)
        
        # Step 2: Apply SRGAN upscaling
        upscaled_image_path = apply_srgan(stylegan_image_path, job_id)
        if not upscaled_image_path:
            return jsonify({"error": "Failed to upscale image with SRGAN"}), 500
        
        upscaled_filename = "upscaled.jpg"
        
        # Step 3: Crop face and create mesh
        face_mesh_path = crop_face(upscaled_image_path, job_id)
        if not face_mesh_path:
            return jsonify({"error": "Failed to crop face and create mesh"}), 500
        
        face_mesh_filename = "face_mesh.obj"
        
        # Step 4: Apply morphing/3D processing
        final_mesh_path = apply_morph(face_mesh_path, job_id)
        if not final_mesh_path:
            return jsonify({"error": "Failed to apply morphing"}), 500
        
        final_mesh_filename = "face_mesh.obj"
        
        # Success! Return job ID and file paths - using filenames, not full paths
        response = {
            "job_id": job_id,
            "stylegan_image": stylegan_filename,
            "upscaled_image": upscaled_filename,
            "face_mesh": face_mesh_filename,
            "final_model": final_mesh_filename
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        app.logger.error(f"Error in generate_model: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/results/<job_id>', methods=['GET'])
def get_results(job_id):
    try:
        # Get paths to all generated files
        final_output_dir = os.path.join(app.config['FINAL_OUTPUT'], job_id)
        
        if not os.path.exists(final_output_dir):
            return jsonify({"error": "Job not found or processing incomplete"}), 404
        
        files = {}
        for file in os.listdir(final_output_dir):
            file_path = os.path.join(final_output_dir, file)
            files[file] = f"/api/download/{job_id}/{file}"
        
        return jsonify({
            "job_id": job_id,
            "files": files
        }), 200
    except Exception as e:
        app.logger.error(f"Error in get_results: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/download/<job_id>/<filename>', methods=['GET'])
def download_file(job_id, filename):
    try:
        # Special case for downloading all files as a zip
        if filename == 'all':
            # Create a BytesIO object to store the zip file
            memory_file = BytesIO()
            
            # Create the zip file
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Get the job directory
                job_dir = os.path.join(app.config['FINAL_OUTPUT'], job_id)
                
                if not os.path.exists(job_dir):
                    return jsonify({"error": "Job directory not found"}), 404
                
                # Add all relevant files to the zip
                files_to_zip = ['face_mesh.obj', 'face_mesh.mtl', 'face_texture.jpg', 'face_texture.png']
                for file in files_to_zip:
                    file_path = os.path.join(job_dir, file)
                    if os.path.exists(file_path):
                        # Add file to zip with just the filename (not full path)
                        zipf.write(file_path, file)
            
            # Seek to the beginning of the BytesIO object
            memory_file.seek(0)
            
            # Return the zip file
            return send_file(
                memory_file,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f'face_3d_model_{job_id}.zip'
            )
                
        # Secure the filename to prevent directory traversal
        filename = secure_filename(filename)
        file_path = os.path.join(app.config['FINAL_OUTPUT'], job_id, filename)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return jsonify({"error": "File not found"}), 404
        
        print(f"Serving file: {file_path}")
        
        # Set content types based on file extension
        content_type = 'application/octet-stream'  # Default
        if filename.endswith('.obj'):
            content_type = 'model/obj'
        elif filename.endswith('.mtl'):
            content_type = 'text/plain'
        elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
            content_type = 'image/jpeg'
        elif filename.endswith('.png'):
            content_type = 'image/png'
            
        # Add response headers for better cross-origin support
        response = send_file(file_path, mimetype=content_type)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response
    except Exception as e:
        print(f"Error in download_file: {str(e)}")
        app.logger.error(f"Error in download_file: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add static assets route
@app.route('/api/assets/<filename>', methods=['GET'])
def serve_assets(filename):
    try:
        assets_path = os.path.join('frontend', 'public', 'assets')
        if not os.path.exists(assets_path):
            os.makedirs(assets_path, exist_ok=True)
            
        file_path = os.path.join(assets_path, filename)
        
        if not os.path.exists(file_path):
            print(f"Asset file not found: {file_path}")
            return jsonify({"error": "Asset file not found"}), 404
            
        content_type = 'application/octet-stream'  # Default
        if filename.endswith('.mp4'):
            content_type = 'video/mp4'
        elif filename.endswith('.webm'):
            content_type = 'video/webm'
        elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
            content_type = 'image/jpeg'
        elif filename.endswith('.png'):
            content_type = 'image/png'
            
        response = send_file(file_path, mimetype=content_type)
        response.headers['Cache-Control'] = 'max-age=86400'  # Cache for 24 hours
        return response
        
    except Exception as e:
        print(f"Error serving asset: {str(e)}")
        app.logger.error(f"Error serving asset: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Function to create a run as admin script
def create_admin_script():
    admin_script = "run_as_admin.bat"
    
    with open(admin_script, 'w') as f:
        f.write('@echo off\n')
        f.write('echo Running Face Generation Backend with Admin Rights\n')
        f.write('echo.\n')
        f.write('powershell -Command "Start-Process cmd -ArgumentList \'/c python app.py\' -Verb RunAs"\n')
        f.write('echo If a UAC prompt appears, please allow the application to run with admin rights.\n')
        f.write('echo.\n')
        f.write('pause\n')
    
    print(f"Created '{admin_script}' - Run this script to start the application with admin privileges.")

# Create the admin script
create_admin_script()

@app.route('/api/models', methods=['GET'])
def get_available_models():
    try:
        available_models = {}
        
        # Check each configured model if its file exists
        for model_name, model_path in app.config['MODELS'].items():
            available_models[model_name] = os.path.exists(model_path)
            
        return jsonify({
            "available_models": available_models
        }), 200
        
    except Exception as e:
        app.logger.error(f"Error in get_available_models: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 