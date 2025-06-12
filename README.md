# Face-to-3D: AI-Powered Face Generation & 3D Modeling

This application transforms 2D face images into realistic 3D face models using advanced AI techniques. It provides a user-friendly web interface for creating, viewing, and downloading 3D face models.

## Features

- Generate realistic face images using AI
- High-resolution image upscaling
- Automatic 3D face model generation
- Interactive 3D model viewer in browser
- Download 3D models in OBJ format with textures
- Global access option via ngrok

## System Requirements

- Python 3.8 or higher
- Node.js 14+ and npm
- Windows OS (tested on Windows 10/11)
- CUDA-capable GPU recommended (but CPU mode is supported)
- At least 8GB RAM
- 2GB free disk space

## Project Setup

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Create and activate a virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Environment Configuration

Create a `.env` file in the root directory with the following variables:
```env
FLASK_ENV=development
FLASK_APP=app.py
PORT=5000
```

## Running the Application

### Development Mode

1. **Start the Backend Server:**
```bash
# From the root directory
python app.py
```

2. **Start the Frontend Development Server:**
```bash
# In a new terminal
cd frontend
npm run dev
```

3. **Access the Application:**
- Backend API: http://localhost:5000
- Frontend development server: http://localhost:5173

### Production Mode

For a production deployment with backend and frontend served together:

```bash
# From the root directory
powershell -File deploy.ps1
```

The application will be available at http://localhost:5000

## Global Access with ngrok (Optional)

To make your local application accessible globally:

1. **Install and Configure ngrok:**
   - Sign up at [ngrok.com](https://ngrok.com)
   - Get your authtoken
   - Configure ngrok:
     ```bash
     ./frontend/ngrok.exe config add-authtoken YOUR_AUTHTOKEN
     ```

2. **Deploy Globally:**
   ```bash
   # Option 1: Deploy everything at once
   powershell -File deploy-global.ps1

   # Option 2: If server is already running
   powershell -File start-ngrok.ps1
   ```

## Project Structure

```
├── app.py              # Main Flask application
├── api.py             # API endpoints
├── frontend/          # React frontend application
├── requirements.txt   # Python dependencies
├── .env              # Environment variables
└── deploy scripts    # Various deployment scripts
```

## Troubleshooting

### Common Issues and Solutions

1. **Port Already in Use**
   ```bash
   # Check what's using port 5000
   netstat -ano | findstr :5000
   # Kill the process if needed (replace PID with actual process ID)
   taskkill /PID <PID> /F
   ```

2. **CUDA/GPU Issues**
   - Ensure you have compatible NVIDIA drivers installed
   - The application will fallback to CPU mode if CUDA is unavailable

3. **Memory Issues**
   - Close other memory-intensive applications
   - Ensure you have at least 8GB of RAM available

4. **Permission Issues**
   - Run as administrator using `run_as_admin.bat`
   - Check file permissions in the project directory

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Flask and React
- Uses advanced AI techniques for face generation and 3D modeling
- Powered by PyTorch for deep learning operations 
