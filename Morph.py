import numpy as np
import cv2
import os
import trimesh
import open3d as o3d
import mediapipe as mp
from scipy.spatial import Delaunay

def extract_face_with_mediapipe(image_path):
    """Extract face texture and landmarks using MediaPipe"""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None, None, None
    
    h, w, _ = image.shape
    print(f"Image dimensions: {w}x{h}")
    
    # Convert to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    
    # Process image to get facial landmarks
    results = face_mesh.process(rgb_image)
    
    if not results.multi_face_landmarks:
        print("No face detected in the image!")
        return None, None, None
    
    # Get the first face
    face_landmarks = results.multi_face_landmarks[0]
    
    # Convert MediaPipe normalized coordinates to image pixels
    landmarks_3d = []
    landmarks_2d = []
    
    for landmark in face_landmarks.landmark:
        x, y, z = int(landmark.x * w), int(landmark.y * h), landmark.z
        landmarks_2d.append((x, y))
        landmarks_3d.append([x, y, z * 1000])  # Scale Z for better visibility
    
    # Convert to NumPy arrays
    landmarks_2d = np.array(landmarks_2d, np.int32)
    landmarks_3d = np.array(landmarks_3d, np.float32)
    
    # Create a mask for the face
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Define the convex hull of the face
    hull = cv2.convexHull(landmarks_2d)
    
    # Fill the face region on the mask
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Extract the face region using the mask
    face_texture = cv2.bitwise_and(image, image, mask=mask)
    
    return face_texture, landmarks_3d, hull

def create_delaunay_mesh(vertices):
    """Create a mesh using Delaunay triangulation"""
    # For 3D vertices, project to a plane for triangulation
    vertices_2d = vertices[:, :2]  # Project to XY plane
    tri = Delaunay(vertices_2d)
    return vertices, tri.simplices

def improved_generate_texture_coordinates(vertices):
    """Generate improved texture coordinates directly from the face texture image"""
    # Find the bounding box of the vertices
    min_x, min_y = np.min(vertices[:, 0]), np.min(vertices[:, 1])
    max_x, max_y = np.max(vertices[:, 0]), np.max(vertices[:, 1])
    
    # Calculate width and height for normalization
    width = max_x - min_x
    height = max_y - min_y
    
    # Create UV coordinates (texture coordinates)
    uv_coords = np.zeros((len(vertices), 2))
    
    for i, vertex in enumerate(vertices):
        # Calculate UV coordinates based on direct image-space mapping
        u = (vertex[0] - min_x) / width
        v = 1.0 - (vertex[1] - min_y) / height  # Flip V to match image coordinates
        
        # Ensure UVs are within proper bounds
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        
        uv_coords[i] = [u, v]
    
    # Apply a small margin to avoid texture bleeding at edges
    margin = 0.02
    uv_coords = uv_coords * (1.0 - 2 * margin) + margin
    
    return uv_coords

def create_texture_image(face_texture, output_path):
    """Create and save a clean texture image without alpha transparency issues"""
    # Create a white background
    h, w, _ = face_texture.shape
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Create a mask for non-black pixels in the face texture
    mask = np.any(face_texture > 5, axis=2).astype(np.uint8) * 255
    
    # Combine face texture with white background using the mask
    result = np.where(np.stack([mask, mask, mask], axis=2) > 0, face_texture, white_bg)
    
    # Save the result
    cv2.imwrite(output_path, result)
    print(f"Texture image saved to: {output_path}")
    return result

def save_textured_obj(vertices, faces, texture_coords, texture_image_path, output_path):
    """Save mesh with texture information to OBJ and MTL files"""
    # Create MTL file path
    mtl_path = output_path.replace('.obj', '.mtl')
    texture_filename = os.path.basename(texture_image_path)
    mtl_filename = os.path.basename(mtl_path)
    
    # Write MTL file
    with open(mtl_path, 'w') as f:
        f.write("# MTL file for textured face mesh\n")
        f.write("newmtl material0\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")  # ambient color
        f.write("Kd 1.000000 1.000000 1.000000\n")  # diffuse color
        f.write("Ks 0.000000 0.000000 0.000000\n")  # specular color
        f.write("Ns 10.000000\n")  # specular exponent
        f.write("d 1.000000\n")  # transparency
        f.write(f"map_Kd {texture_filename}\n")  # texture map
    
    # Write OBJ file
    with open(output_path, 'w') as f:
        f.write(f"mtllib {mtl_filename}\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # Write texture coordinates
        for uv in texture_coords:
            f.write(f"vt {uv[0]} {uv[1]}\n")
        
        # Write material
        f.write("usemtl material0\n")
        
        # Write faces with texture coordinates
        for face in faces:
            # OBJ indices start at 1
            f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")
    
    print(f"Textured mesh saved to: {output_path}")
    print(f"Material file saved to: {mtl_path}")

def create_face_mesh(image_path, output_dir='.'):
    """Create a textured 3D face mesh from an image using MediaPipe landmarks"""
    print(f"Processing image: {image_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract face and landmarks
    face_texture, landmarks_3d, hull = extract_face_with_mediapipe(image_path)
    if face_texture is None:
        return False, None
    
    # Create a mesh using Delaunay triangulation
    vertices, faces = create_delaunay_mesh(landmarks_3d)
    
    # Create a clean texture image
    texture_path = os.path.join(output_dir, "face_texture.jpg")
    clean_texture = create_texture_image(face_texture, texture_path)
    
    # Generate texture coordinates
    texture_coords = improved_generate_texture_coordinates(vertices)
    
    # Save the textured mesh
    obj_path = os.path.join(output_dir, "face_mesh.obj")
    save_textured_obj(vertices, faces, texture_coords, texture_path, obj_path)
    
    print("\nFace mesh creation complete!")
    print(f"Final output saved to: {obj_path}")
    print("You can view this file in any 3D viewer that supports OBJ files with textures")
    
    return True, obj_path

def visualize_mesh(mesh_path):
    """Visualize the mesh using either Open3D or Trimesh"""
    print(f"Attempting to visualize mesh from: {mesh_path}")
    
    try:
        # Try Open3D visualization
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        
        # Create a visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add the mesh
        vis.add_geometry(mesh)
        
        # Set rendering options
        opt = vis.get_render_option()
        opt.mesh_show_back_face = True
        
        # Try to enable textures
        try:
            opt.texture_rendering_mode = o3d.visualization.TextureRenderingMode.TextureRendering
        except:
            pass
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()
        return True
            
    except Exception as e:
        print(f"Open3D visualization error: {e}")
        print("Trying Trimesh visualization...")
        
        try:
            import trimesh
            mesh = trimesh.load(mesh_path)
            mesh.show()
            return True
        except Exception as e:
            print(f"Trimesh visualization error: {e}")
            print("Please use an external 3D viewer to view the mesh.")
            return False

def main():
    # Get image path from user
    image_path = input("Enter path to face image: ")
    
    # Validate the image path
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return
    
    # Create output directory
    output_dir = input("Enter output directory (default: current directory): ") or "."
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the image
    success, mesh_path = create_face_mesh(image_path, output_dir)
    
    if success:
        print("\nProcess completed successfully!")
        
        # Ask if user wants to visualize the mesh
        visualize = input("Would you like to visualize the mesh? (y/n): ").lower()
        if visualize == 'y':
            visualize_mesh(mesh_path)

if __name__ == "__main__":
    main()