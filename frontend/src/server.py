from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import subprocess
import os
import time
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Add GPU memory cleanup at the start
torch.cuda.empty_cache()  # Clear any cached GPU memory

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Path configurations
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = r'C:\Users\Acer\Downloads\GAN_CV\ganCV\src\Models\saved_images'
srgan_model_path = os.path.join(current_dir, 'Models', 'srgan_generator.pt')

print(f"Current directory: {current_dir}")
print(f"Output directory: {output_dir}")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory ensured: {output_dir}")

# Global model variable
global srgan_model

# SRGAN Model definition
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return x + self.bn2(self.conv2(self.prelu(self.bn1(self.conv1(x))))) 

class Generator(nn.Module):
    def __init__(self, num_residual_blocks=8):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.upsample1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        residual = self.prelu(self.conv1(x))
        x = self.res_blocks(residual)
        x = self.bn2(self.conv2(x)) + residual
        x = self.upsample1(x)
        x = self.upsample2(x)
        return torch.tanh(self.conv3(x))

# Load SRGAN model
def load_srgan_model():
    try:
        if not os.path.exists(srgan_model_path):
            print(f"SRGAN model not found at {srgan_model_path}")
            return None
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try to load as a TorchScript model first
        try:
            generator = torch.jit.load(srgan_model_path, map_location=device)
            print(f"SRGAN model loaded as TorchScript from {srgan_model_path}")
            return generator
        except Exception as e:
            print(f"Failed to load as TorchScript, trying regular model: {str(e)}")
            
        # If TorchScript loading fails, try regular model loading
        generator = Generator().to(device)
        generator.load_state_dict(torch.load(srgan_model_path, map_location=device), strict=False)
        generator.eval()
        print(f"SRGAN model loaded from {srgan_model_path}")
        return generator
    except Exception as e:
        print(f"Error loading SRGAN model: {str(e)}")
        return None

# Initialize srgan_model at module level
srgan_model = None

# Update the upscale_image function with better error reporting
def upscale_image(input_path, output_path):
    try:
        global srgan_model
        print(f"Starting upscale from {input_path} to {output_path}")
        
        if srgan_model is None:
            print("SRGAN model not loaded")
            return False

        if not os.path.exists(input_path):
            print(f"Input image not found at {input_path}")
            return False
            
        print(f"Input image exists, size: {os.path.getsize(input_path)} bytes")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Prepare transforms
        try:
            img = Image.open(input_path).convert('RGB')
            print(f"Image opened successfully: {img.size}")
            lr_img = transforms.ToTensor()(img).unsqueeze(0).to(device)
            print(f"Image tensor shape: {lr_img.shape}")
        except Exception as e:
            print(f"Error preparing image: {str(e)}")
            return False

        # Free up GPU memory before processing
        torch.cuda.empty_cache()
        
        # --- Super-resolve ---
        try:
            with torch.no_grad():
                print("Starting image upscaling...")
                sr_img = srgan_model(lr_img)
                print(f"Upscaling completed, tensor shape: {sr_img.shape}")
                sr_img = sr_img.squeeze(0).clamp(0, 1).cpu()
                print("Tensor processed")
        except Exception as e:
            print(f"Error during upscaling: {str(e)}")
            return False

        # --- Save output image ---
        try:
            output_img = transforms.ToPILImage()(sr_img)
            print(f"Output PIL image created: {output_img.size}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            print(f"Attempting to save to: {output_path}")
            output_img.save(output_path)
            
            # Verify the file was actually written
            if os.path.exists(output_path):
                print(f"Image confirmed saved to {output_path}, size: {os.path.getsize(output_path)} bytes")
                return True
            else:
                print(f"ERROR: File not found after save operation: {output_path}")
                return False
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return False
    except Exception as e:
        print(f"Error in upscale_image function: {str(e)}")
        return False

@app.route('/')
def index():
    return jsonify({'message': 'Welcome to the StyleGAN and SRGAN API'})

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images with no-cache headers"""
    if not os.path.exists(os.path.join(output_dir, filename)):
        return jsonify({'error': 'Image not found'}), 404
        
    response = send_from_directory(output_dir, filename)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
    return response

@app.route('/generate-image', methods=['POST'])
def generate_image():
    """Generate StyleGAN image using Python subprocess"""
    # Use Path for better cross-platform compatibility
    python_path = r'C:\Users\Acer\Anaconda3\envs\stylegan-pytorch\python.exe'
    gen_script = r'C:\Users\Acer\Documents\GAN\stylegan3\gen_images.py'
    model_path = r'C:\Users\Acer\Downloads\st-08\network-snapshot-000160.pkl'
    conda_activate = r'C:\Users\Acer\Anaconda3\Scripts\activate.bat'
    
    # Verify file paths
    for path_info in [
        {'path': python_path, 'name': 'Python interpreter'},
        {'path': gen_script, 'name': 'StyleGAN script'},
        {'path': model_path, 'name': 'Model file'},
        {'path': conda_activate, 'name': 'Conda activate script'}
    ]:
        if not os.path.exists(path_info['path']):
            error_msg = f"{path_info['name']} not found at {path_info['path']}"
            print(error_msg)
            return jsonify({'error': 'Execution failed', 'details': error_msg}), 500
    
    # Generate a unique filename based on timestamp
    timestamp = int(time.time())
    expected_image = f'seed0000_{timestamp}.png'
    image_path = os.path.join(output_dir, expected_image)
    
    # Modified command with memory optimization parameters
    command = f'set CUDA_LAUNCH_BLOCKING=1 && set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 && "{conda_activate}" stylegan-pytorch && set CUDA_VISIBLE_DEVICES=0 && "{python_path}" "{gen_script}" --outdir="{output_dir}" --trunc=1 --seeds=0 --network="{model_path}" '
    
    try:
        print(f'Executing command: {command}')
        # Execute command
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(timeout=120)  # Add timeout to prevent hanging
        
        # More detailed error checking
        if process.returncode != 0 or stderr:
            print(f'Return code: {process.returncode}')
            print(f'Python stderr:\n{stderr}')
            print(f'Python stdout:\n{stdout}')
            
            # Check for CUDA memory error
            if 'Memory allocation failure' in stderr or 'CUDA error' in stderr:
                return jsonify({
                    'error': 'GPU memory error', 
                    'details': 'Not enough GPU memory to generate the image. Try closing other GPU applications.',
                    'stderr': stderr
                }), 500
                
            if process.returncode != 0 or ('error' in stderr.lower() and 'failed' in stderr.lower()):
                return jsonify({'error': 'Python script error', 'details': stderr}), 500
        
        print(f'Python stdout:\n{stdout}')
        
        # Wait for file to be written (StyleGAN may take a moment to finalize)
        attempts = 0
        max_attempts = 12  # Wait up to 60 seconds (5s * 12)
        
        while attempts < max_attempts:
            if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                print(f'Image found: {image_path}')
                
                # Create name for upscaled image with proper extension
                upscaled_image = f'upscaled_{expected_image}'
                upscaled_file_path = os.path.join(output_dir, upscaled_image)
                
                # Free GPU memory before upscaling
                torch.cuda.empty_cache()
                
                # Apply SRGAN upscaling
                upscale_success = upscale_image(image_path, upscaled_file_path)
                
                if upscale_success and os.path.exists(upscaled_file_path):
                    return jsonify({
                        'success': True,
                        'imageUrl': f'/images/{expected_image}',
                        'upscaledImageUrl': f'/images/{upscaled_image}'
                    })
                else:
                    return jsonify({
                        'success': True,
                        'imageUrl': f'/images/{expected_image}',
                        'upscaledImageUrl': None,
                        'warning': 'Failed to upscale image'
                    })
            
            print(f'Waiting for image: {image_path}')
            time.sleep(5)
            attempts += 1
        
        return jsonify({'error': 'Execution failed', 'details': 'Image not generated within timeout'}), 500
    
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Execution timed out', 'details': 'The generation process took too long'}), 504
    except Exception as e:
        error_msg = f'Execution failed: {str(e)}'
        print(error_msg)
        return jsonify({'error': 'Execution failed', 'details': str(e)}), 500

@app.route('/upscale-image', methods=['POST'])
def upscale_image_api():
    """API endpoint to upscale an existing image"""
    try:
        data = request.get_json()
        if not data or 'imageName' not in data:
            return jsonify({'error': 'Missing image name'}), 400
            
        image_name = data['imageName']
        input_path = os.path.join(output_dir, image_name)
        
        if not os.path.exists(input_path):
            return jsonify({'error': f'Image {image_name} not found'}), 404
            
        upscaled_name = f'upscaled_{image_name}'
        output_path = os.path.join(output_dir, upscaled_name)
        
        # Free GPU memory before upscaling
        torch.cuda.empty_cache()
        
        if upscale_image(input_path, output_path) and os.path.exists(output_path):
            return jsonify({
                'success': True,
                'originalImageUrl': f'/images/{image_name}',
                'upscaledImageUrl': f'/images/{upscaled_name}'
            })
        else:
            return jsonify({'error': 'Failed to upscale image'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add test endpoint to verify SRGAN upscaling
@app.route('/test-upscale', methods=['GET'])
def test_upscale():
    """Test SRGAN model with a sample image"""
    try:
        # Use an existing sample image that's known to exist
        sample_image_path = os.path.join(current_dir, 'sample.png')
        
        # If no sample exists, create a dummy image
        if not os.path.exists(sample_image_path):
            from PIL import Image
            import numpy as np
            # Create a simple test image
            img = Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))
            img.save(sample_image_path)
            print(f"Created test image at {sample_image_path}")
        
        test_output_path = os.path.join(output_dir, 'test_upscaled.png')
        
        # Free GPU memory before upscaling
        torch.cuda.empty_cache()
        
        success = upscale_image(sample_image_path, test_output_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Upscaling test successful',
                'inputPath': sample_image_path,
                'outputPath': test_output_path
            })
        else:
            return jsonify({
                'error': 'Upscaling test failed',
                'inputPath': sample_image_path
            }), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Clear GPU memory before loading models
    torch.cuda.empty_cache()
    
    # Load the SRGAN model once when starting the app
    srgan_model = load_srgan_model()
    
    # Print model information
    if srgan_model is None:
        print("WARNING: SRGAN model failed to load. Upscaling will not work!")
    else:
        print(f"SRGAN model loaded successfully: {type(srgan_model).__name__}")
    
    # Start the Flask server
    app.run(debug=True, host='0.0.0.0', port=5000)