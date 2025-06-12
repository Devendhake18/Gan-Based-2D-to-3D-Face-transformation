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
import torch

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Output directory configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, '..', '..', 'Downloads', 'GAN_CV', 'ganCV', 'src', 'Models', 'saved_images')
# Fix path separator to be OS-independent
srgan_model_path = os.path.join(current_dir, 'Models', 'generator_epoch299(1).pth')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory ensured: {output_dir}")

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

# Load SRGAN model with proper GPU check
def load_srgan_model():
    try:
        # Check if CUDA is available and print detailed info
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1e6} MB")
            print(f"PyTorch CUDA version: {torch.version.cuda}")
        
        generator = Generator().to(device)
        generator.load_state_dict(torch.load(srgan_model_path, map_location=device), strict=False)
        generator.eval()
        
        # Verify model is on correct device
        device_info = next(generator.parameters()).device
        print(f"Model device: {device_info}")
        print(f"SRGAN model loaded from {srgan_model_path}")
        
        # For mixed precision acceleration if using recent GPU
        if device.type == 'cuda' and torch.cuda.get_device_capability(0)[0] >= 7:
            print("Enabling mixed precision acceleration")
            # No need to set up amp specifically here, we'll handle it in inference
        
        return generator
    except Exception as e:
        print(f"Error loading SRGAN model: {str(e)}")
        return None

# Apply SRGAN upscaling to an image with GPU monitoring
def upscale_image(input_path, output_path):
    try:
        if srgan_model is None:
            raise Exception("SRGAN model not loaded")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Prepare transforms
        transform = transforms.ToTensor()
        img = Image.open(input_path).convert('RGB')
        
        # Move input to device and track memory usage
        if device.type == 'cuda':
            torch.cuda.empty_cache()  # Clear any cached memory
            print(f"Before loading image - CUDA Memory: {torch.cuda.memory_allocated(0) / 1e6} MB")
        
        lr_img = transform(img).unsqueeze(0).to(device)
        
        if device.type == 'cuda':
            print(f"After loading image - CUDA Memory: {torch.cuda.memory_allocated(0) / 1e6} MB")
            print(f"Input tensor device: {lr_img.device}")

        # --- Super-resolve with GPU synchronization ---
        with torch.no_grad():
            # Run inference
            sr_img = srgan_model(lr_img)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()  # Make sure GPU computation is finished
                print(f"After processing - CUDA Memory: {torch.cuda.memory_allocated(0) / 1e6} MB")
            
            sr_img = sr_img.squeeze(0).clamp(0, 1).cpu()

        # --- Save output image ---
        output_img = transforms.ToPILImage()(sr_img)
        output_img.save(output_path)

        print(f"Image upscaled and saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error upscaling image: {str(e)}")
        return False

# Initialize SRGAN model
srgan_model = load_srgan_model()

@app.route('/')
def index():
    return jsonify({'message': 'Welcome to the StyleGAN and SRGAN API'})

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images with no-cache headers"""
    response = send_from_directory(output_dir, filename)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
    return response

@app.route('/check-gpu')
def check_gpu():
    """Check if GPU is available and return detailed information"""
    gpu_available = torch.cuda.is_available()
    gpu_info = {
        'available': gpu_available,
        'device_count': torch.cuda.device_count() if gpu_available else 0
    }
    
    if gpu_available:
        gpu_info.update({
            'device_name': torch.cuda.get_device_name(0),
            'memory_allocated': float(torch.cuda.memory_allocated(0) / 1e6),  # MB
            'memory_reserved': float(torch.cuda.memory_reserved(0) / 1e6),    # MB
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version() if hasattr(torch.backends.cudnn, 'version') else 'Unknown'
        })
        
        # Check if CUDNN is enabled
        gpu_info['cudnn_enabled'] = torch.backends.cudnn.enabled
        
        # Run a small test tensor calculation on GPU
        try:
            test_tensor = torch.randn(100, 100).cuda()
            result = torch.matmul(test_tensor, test_tensor)
            torch.cuda.synchronize()
            gpu_info['test_calculation'] = 'Success'
        except Exception as e:
            gpu_info['test_calculation'] = f'Failed: {str(e)}'
    
    return jsonify(gpu_info)

@app.route('/generate-image', methods=['POST'])
def generate_image():
    """Generate StyleGAN image using Python subprocess with GPU support"""
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
    
    # Set environment variable to force CUDA visibility
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first GPU
    
    # Command to run Python script with cuda env variables in Conda environment
    command = f'"{conda_activate}" stylegan-pytorch && set CUDA_VISIBLE_DEVICES=0 && "{python_path}" "{gen_script}" --outdir="{output_dir}" --trunc=1 --seeds=0 --network="{model_path}" --gpu=0'
    
    try:
        print(f'Executing command: {command}')
        # Execute command with environment variables
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env  # Pass environment variables
        )
        stdout, stderr = process.communicate()
        
        print(f'Python stdout:\n{stdout}')
        if stderr:
            print(f'Python stderr:\n{stderr}')
        
        if process.returncode != 0 or (stderr and ('error' in stderr.lower() or 'failed' in stderr.lower())):
            return jsonify({'error': 'Python script error', 'details': stderr}), 500
        
        # Wait for file to be written (StyleGAN may take a moment to finalize)
        expected_image = 'seed0000.png'
        image_path = os.path.join(output_dir, expected_image)
        attempts = 0
        max_attempts = 12  # Wait up to 60 seconds (5s * 12)
        
        while attempts < max_attempts:
            if os.path.exists(image_path):
                print(f'Image found: {image_path}')
                
                # Create name for upscaled image
                upscaled_image = 'upscaled_' + expected_image
                upscaled_path = os.path.join(output_dir, upscaled_image)
                
                # Apply SRGAN upscaling
                upscale_success = upscale_image(image_path, upscaled_path)
                
                # Proper handling of upscale result
                if upscale_success and os.path.exists(upscaled_path):
                    return jsonify({
                        'imageUrl': f'/images/{expected_image}',
                        'upscaledImageUrl': f'/images/{upscaled_image}'
                    })
                else:
                    return jsonify({
                        'imageUrl': f'/images/{expected_image}',
                        'upscaledImageUrl': None,
                        'warning': 'Failed to upscale image'
                    })
            
            print(f'Waiting for image: {image_path}')
            time.sleep(5)
            attempts += 1
        
        return jsonify({'error': 'Execution failed', 'details': 'Image not generated within timeout'}), 500
    
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
        
        # Check if upscale function succeeded AND if the output file exists
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

@app.route('/benchmark-gpu', methods=['GET'])
def benchmark_gpu():
    """Run a simple GPU benchmark to verify performance"""
    if not torch.cuda.is_available():
        return jsonify({'error': 'CUDA not available'}), 400
    
    try:
        # Record start time
        start_time = time.time()
        
        # Create large tensors
        size = 2000  # Matrix size
        a = torch.randn(size, size, device='cuda')
        b = torch.randn(size, size, device='cuda')
        
        # Perform matrix multiplication (compute-intensive operation)
        torch.cuda.synchronize()  # Ensure CUDA operations are synchronized
        mult_start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()  # Wait for GPU computation to finish
        mult_end = time.time()
        
        # Free memory
        del a, b, c
        torch.cuda.empty_cache()
        
        end_time = time.time()
        
        return jsonify({
            'success': True,
            'total_time_ms': (end_time - start_time) * 1000,
            'computation_time_ms': (mult_end - mult_start) * 1000,
            'matrix_size': size,
            'gpu_device': torch.cuda.get_device_name(0)
        })
    except Exception as e:
        return jsonify({'error': f'GPU benchmark failed: {str(e)}'}), 500

# Set CUDA optimization flags at startup
if torch.cuda.is_available():
    # Enable cuDNN auto-tuner for better performance
    torch.backends.cudnn.benchmark = True
    print("CUDA optimizations enabled")
    
    # Optional: Enable deterministic algorithms for consistent results
    # (may reduce performance but increase reproducibility)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    print(f'Server running at http://localhost:5000')
    # Print CUDA status on startup
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available")
    app.run(host='0.0.0.0', port=5000, debug=False)