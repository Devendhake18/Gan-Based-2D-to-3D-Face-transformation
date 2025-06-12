# Demo Video Setup

To enable the demo video feature in your application, follow these steps:

## 1. Prepare Your Demo Video

Create a video demonstrating how to use the 3D model files in software like Blender, Maya, or any other 3D modeling software. This should show users how to:
- Import the OBJ file
- Ensure textures are correctly applied
- Basic manipulation of the 3D face model
- Tips for using the model in projects

## 2. Place the Video in the Correct Location

The video file must be named `demo.mp4` and placed in this directory:
```
frontend/public/assets/demo.mp4
```

## 3. Verify the Video

After placing the video file, restart your application and generate a 3D face. 
The demo video section should appear automatically after results are displayed, showing your instructional video.

## Technical Notes

- The video is served from the `/api/assets/demo.mp4` endpoint
- Maximum recommended file size: 20MB (to ensure fast loading)
- Recommended video resolution: 1280x720 (HD)
- Format: MP4 with H.264 encoding for best browser compatibility
- If your video is in another format, convert it using a tool like FFmpeg or an online converter

## Troubleshooting

If the video doesn't appear:
1. Check that the file is correctly named and placed in the proper directory
2. Ensure the application has been restarted to recognize the new file
3. Check browser console for any errors related to loading the video
4. Verify the video codec is compatible with web browsers 