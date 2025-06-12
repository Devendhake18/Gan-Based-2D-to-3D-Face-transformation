import os
import sys
import shutil
import tempfile

# Create base temp directory in system temp folder
BASE_TEMP_DIR = os.path.join(tempfile.gettempdir(), 'face_generation_temp')

def check_directory(dir_path):
    """Check if directory exists and create it if not"""
    if not os.path.exists(dir_path):
        print(f"Creating directory: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)
    return os.path.exists(dir_path)

def check_file(file_path, alt_paths=None):
    """Check if file exists in primary location or any of the alternative locations"""
    exists = os.path.exists(file_path)
    if exists:
        print(f"✓ Found file: {file_path}")
        return True, file_path
    
    # Check alternative paths if provided
    if alt_paths:
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"✓ Found file (alternative location): {alt_path}")
                return True, alt_path
                
    # File not found in any location
    print(f"✗ Missing file: {file_path}")
    return False, None

def check_and_setup_models():
    """Check for required model files and directories"""
    print("Checking model files and directories...")
    
    # Create the base temp directory
    os.makedirs(BASE_TEMP_DIR, exist_ok=True)
    print(f"Using system temp directory: {BASE_TEMP_DIR}")
    
    # Check for required directories
    check_directory(os.path.join(BASE_TEMP_DIR, "temp_uploads"))
    check_directory(os.path.join(BASE_TEMP_DIR, "stylegan_output"))
    check_directory(os.path.join(BASE_TEMP_DIR, "srgan_output"))
    check_directory(os.path.join(BASE_TEMP_DIR, "face_crop_output"))
    check_directory(os.path.join(BASE_TEMP_DIR, "morph_output"))
    check_directory("final_output")  # Keep this in the application directory
    check_directory("epochs")  # Keep this in the application directory
    
    # Test if we can write to the system temp directory
    test_file = os.path.join(BASE_TEMP_DIR, "test_permissions.txt")
    try:
        with open(test_file, 'w') as f:
            f.write("testing write permissions")
        os.remove(test_file)
        print(f"✓ System temp directory has write permissions: {BASE_TEMP_DIR}")
    except Exception as e:
        print(f"✗ Permission error on system temp directory: {e}")
        print(f"  Application may not function correctly without write permissions to temp directory!")
        return False
    
    # Check for model files
    stylegan_model = "Stylegan Model/network-snapshot-000160.pkl"
    srgan_model_paths = [
        'epochs/netG_epoch_4_49.pth',  # Original path
        'SRGAN Model/netG_epoch_4_100.pth',  # User's path
        'D:/CODDING STUFF/Sem 6/CV_GAN/Backend/SRGAN Model/netG_epoch_4_100.pth'  # Absolute path
    ]
    
    stylegan_ok, _ = check_file(stylegan_model)
    srgan_ok, srgan_path = check_file(srgan_model_paths[0], srgan_model_paths[1:])
    
    # Create symlink if found in alternative location but not in primary
    if srgan_ok and srgan_path != srgan_model_paths[0]:
        try:
            print(f"Found SRGAN model at: {srgan_path}")
            print(f"Model will be used from this location")
        except Exception as e:
            print(f"Note: Using model from alternative location: {srgan_path}")
    
    if not stylegan_ok:
        print(f"\nMissing StyleGAN model file: {stylegan_model}")
        print("Please download the StyleGAN model file and place it in the 'Stylegan Model' directory.")
    
    if not srgan_ok:
        print(f"\nMissing SRGAN model file. Checked these locations:")
        for path in srgan_model_paths:
            print(f"  - {path}")
        print("Please download an SRGAN model file and place it in one of these locations.")
    
    if not stylegan_ok or not srgan_ok:
        print("\nMissing required model files. The application may not work correctly.")
        return False
    
    print("\nAll required models and directories are available.")
    return True

if __name__ == "__main__":
    check_and_setup_models() 