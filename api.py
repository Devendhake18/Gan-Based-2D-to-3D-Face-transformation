import requests
import json
import os

class FaceGenerationAPI:
    """Utility class for interacting with the face generation backend API"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        
    def health_check(self):
        """Check if the backend API is running"""
        try:
            response = requests.get(f"{self.base_url}/api/health")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def generate_face(self, seed=0):
        """Initiate the face generation pipeline with a specific seed"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"seed": seed}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error = response.json().get('error', 'Unknown error')
                print(f"Error generating face: {error}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
    
    def get_results(self, job_id):
        """Get the results for a specific job"""
        try:
            response = requests.get(f"{self.base_url}/api/results/{job_id}")
            
            if response.status_code == 200:
                return response.json()
            else:
                error = response.json().get('error', 'Unknown error')
                print(f"Error getting results: {error}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
    
    def download_file(self, job_id, filename, output_path=None):
        """Download a specific file from a job"""
        if output_path is None:
            # Use the current directory if no output path is specified
            output_path = os.path.join(os.getcwd(), filename)
        
        try:
            response = requests.get(
                f"{self.base_url}/api/download/{job_id}/{filename}",
                stream=True
            )
            
            if response.status_code == 200:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
                # Save the file
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                return output_path
            else:
                error = response.json().get('error', 'Unknown error')
                print(f"Error downloading file: {error}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None

# Example usage
if __name__ == "__main__":
    api = FaceGenerationAPI()
    
    # Check if the API is running
    if not api.health_check():
        print("API is not running!")
        exit(1)
    
    # Generate a face with seed 42
    result = api.generate_face(seed=42)
    
    if result:
        print(f"Job ID: {result['job_id']}")
        print(f"Generated files:")
        print(f"  StyleGAN image: {result['stylegan_image']}")
        print(f"  Upscaled image: {result['upscaled_image']}")
        print(f"  Face mesh: {result['face_mesh']}")
        print(f"  Final model: {result['final_model']}")
        
        # Download the final model
        job_id = result['job_id']
        downloaded_path = api.download_file(job_id, "face_mesh.obj", "downloaded_model.obj")
        
        if downloaded_path:
            print(f"Downloaded model to: {downloaded_path}")
    else:
        print("Failed to generate face!") 