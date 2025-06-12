import cv2
import numpy as np
import mediapipe as mp

# Load your image
image_path = r'D:\Facial_LandMark\Facial_landmark\high.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Detect facial landmarks
results = face_mesh.process(image_rgb)

if not results.multi_face_landmarks:
    print("No face detected.")
    exit()

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

# Save as transparent PNG
cv2.imwrite("transparent_face.png", result)
print("Saved: transparent_face.png")