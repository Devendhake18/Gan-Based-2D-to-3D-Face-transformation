# Face-to-3D: AI-Powered Face Generation & 3D Modeling

This application generates realistic 3D face models using StyleGAN and advanced 3D morphing techniques. It provides a web interface to create, view, and download 3D face models.

## Features

- Generate realistic face images using StyleGAN
- Upscale images with SRGAN for enhanced quality
- Automatic 3D face model generation
- Interactive 3D model viewer
- Global access option via ngrok

## Requirements

- Python 3.8+ with pip
- Node.js and npm
- Windows OS (tested on Windows 10/11)
- GPU recommended but not required (CPU mode supported)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Backend
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Download Model Files

Ensure you have the following model files in the correct locations:
- StyleGAN model: `Stylegan Model/network-snapshot-000160.pkl`
- SRGAN model: `SRGAN Model/netG_epoch_4_100.pth`

## Running the Application

### Local Development

1. **Start the backend server:**
   ```bash
   python app.py
   ```

2. **Start the frontend development server in a new terminal:**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access the application:**
   - Backend API: http://localhost:5000
   - Frontend development server: http://localhost:5173

### Production Deployment

For a production-ready build with backend and frontend served from the same server:

```bash
powershell -File deploy.ps1
```

This script:
- Builds the React frontend
- Starts the Flask server to serve both the API and frontend
- Accesses the application at http://localhost:5000

## Global Deployment with ngrok

To make your locally running application accessible globally:

### 1. Install ngrok

ngrok is already included in the repository in the `frontend` directory.

### 2. Sign up for ngrok

Create a free account at [ngrok.com](https://ngrok.com) and get your authtoken.

### 3. Configure ngrok (one-time setup)

```bash
./frontend/ngrok.exe config add-authtoken YOUR_AUTHTOKEN
```
Replace `YOUR_AUTHTOKEN` with the token from your ngrok dashboard.

### 4. Deploy Globally

There are two options to deploy your application globally:

#### Option 1: Deploy Everything at Once

Use the `deploy-global.ps1` script to build the frontend, start the backend, and expose it with ngrok:

```bash
powershell -File deploy-global.ps1
```

#### Option 2: Use with an Already Running Server

If your backend server is already running, use the `start-ngrok.ps1` script:

```bash
powershell -File start-ngrok.ps1
```

### 5. Access Globally

When ngrok starts, it will display a "Forwarding" URL like:
```
Forwarding https://xxxx-xxxx.ngrok-free.app -> http://localhost:5000
```

Share this URL with anyone worldwide to access your application running on your local machine.

## Using the Application

1. **Generate a Face:**
   - Enter a seed value (0-999) or use the random seed button
   - Click "Generate 3D Face"

2. **View the Results:**
   - The application will display the generated StyleGAN image, upscaled version, and 3D model
   - The 3D model can be rotated and zoomed in the viewer

3. **Download Files:**
   - Use the download buttons to get the OBJ, MTL, and texture files
   - These files can be imported into any 3D software that supports OBJ format

## Troubleshooting

- **Permissions Issues:** Run the application as administrator or use the provided `run_as_admin.bat` script
- **3D Model Not Loading:** Ensure your browser supports WebGL. The application will show a fallback view with a texture image if the 3D viewer fails.
- **ngrok Connection Issues:** Check your internet connection and firewall settings. Make sure port 5000 is allowed for outgoing connections.

## Notes

- With the free ngrok plan, you'll get a new URL each time you start ngrok
- Your computer must stay turned on and connected to the internet while sharing your application globally
- The free plan has limitations on bandwidth and connections

## License

[Include license information here] 