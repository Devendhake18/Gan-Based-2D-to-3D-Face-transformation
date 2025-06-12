import os
import shutil
import sys

def setup_model_directories():
    """Create the necessary directory structure for StyleGAN models"""
    # Base directory
    base_dir = "Stylegan Model"
    
    # Model subdirectories
    model_dirs = ["ffhq", "celeba"]
    
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        print(f"Creating base directory: {base_dir}")
        os.makedirs(base_dir)
    else:
        print(f"Base directory already exists: {base_dir}")
    
    # Create subdirectories
    for model_dir in model_dirs:
        model_path = os.path.join(base_dir, model_dir)
        if not os.path.exists(model_path):
            print(f"Creating model directory: {model_path}")
            os.makedirs(model_path)
        else:
            print(f"Model directory already exists: {model_path}")
    
    # Create placeholder files 
    for model_dir in model_dirs:
        model_path = os.path.join(base_dir, model_dir)
        # Use the same file name for both models
        file_name = "network-snapshot-000160.pkl"
        readme_path = os.path.join(model_path, "README.txt")
        
        # Check if model file exists
        model_file_path = os.path.join(model_path, file_name)
        if os.path.exists(model_file_path):
            file_size = os.path.getsize(model_file_path) / (1024 * 1024)  # Size in MB
            print(f"Model file exists: {model_file_path} ({file_size:.2f} MB)")
        else:
            print(f"Model file missing: {model_file_path}")
            
            # Create README file with instructions
            with open(readme_path, 'w') as f:
                f.write(f"Place the {file_name} file in this directory.\n\n")
                f.write("You can download StyleGAN models from NVIDIA's StyleGAN repository or other sources.\n")
                f.write("Make sure to rename the file to match the expected filename.\n\n")
                f.write("For more information, see the MODEL_SETUP.md file in the root directory.\n")
            
            print(f"Created instruction file: {readme_path}")
    
    print("\nDirectory setup complete!")
    print("Please place the model files in their respective directories.")
    print("See MODEL_SETUP.md for detailed instructions.")

if __name__ == "__main__":
    setup_model_directories() 