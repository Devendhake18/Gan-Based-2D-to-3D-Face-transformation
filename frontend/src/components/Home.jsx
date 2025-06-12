import React, { useState, useEffect, useRef, Suspense, useMemo, Component } from 'react';
import * as THREE from 'three';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, Text, useTexture, Loader } from '@react-three/drei';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js';
import { motion } from 'framer-motion';
import './FaceGeneration.css';

// Register GSAP plugins
gsap.registerPlugin(ScrollTrigger);

// Default API URL as fallback
const DEFAULT_API_URL = window.location.origin;

// Platform component that renders a rotating wheel/platform
function Platform({ children, autoRotate = false }) {
  const platformRef = useRef();
  const [isRotating, setIsRotating] = useState(autoRotate);
  const [rotationSpeed, setRotationSpeed] = useState(0.003);
  
  // Create a blue gradient background texture
  const texture = useMemo(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const context = canvas.getContext('2d');
    
    // Create a circular gradient
    const gradient = context.createRadialGradient(
      256, 256, 0,
      256, 256, 256
    );
    
    gradient.addColorStop(0, '#2A2A4A');
    gradient.addColorStop(0.5, '#353564');
    gradient.addColorStop(0.8, '#4040A0');
    gradient.addColorStop(1, '#4545C0');
    
    // Fill with gradient
    context.fillStyle = gradient;
    context.fillRect(0, 0, 512, 512);
    
    // Add some circular patterns
    context.strokeStyle = 'rgba(180, 200, 255, 0.2)';
    for (let i = 1; i <= 3; i++) {
      context.beginPath();
      context.arc(256, 256, i * 80, 0, Math.PI * 2);
      context.lineWidth = 1;
      context.stroke();
    }
    
    // Create THREE texture from canvas
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(1, 1);
    
    return tex;
  }, []);
  
  // Handle rotation animation
  useFrame(() => {
    if (platformRef.current && isRotating) {
      platformRef.current.rotation.y += rotationSpeed;
    }
  });

  return (
    <group>
      {/* Control buttons */}
      <group position={[0, -1.2, 0]}>
        <mesh 
          position={[-0.6, 0, 0]} 
          onClick={() => setIsRotating(!isRotating)} 
          onPointerOver={() => document.body.style.cursor = 'pointer'} 
          onPointerOut={() => document.body.style.cursor = 'default'}
        >
          <boxGeometry args={[0.3, 0.1, 0.1]} />
          <meshStandardMaterial color={isRotating ? "#00ff00" : "#ff0000"} />
        </mesh>
        <Text position={[-0.6, 0.15, 0]} fontSize={0.08} color="white">
          {isRotating ? "Stop" : "Rotate"}
        </Text>
        
        {/* Speed controls */}
        <mesh 
          position={[0, 0, 0]} 
          onClick={() => setRotationSpeed(Math.max(0.001, rotationSpeed - 0.001))} 
          onPointerOver={() => document.body.style.cursor = 'pointer'} 
          onPointerOut={() => document.body.style.cursor = 'default'}
        >
          <boxGeometry args={[0.2, 0.1, 0.1]} />
          <meshStandardMaterial color={"#ffcc00"} />
        </mesh>
        <Text position={[0, 0.15, 0]} fontSize={0.08} color="white">
          -
        </Text>
        
        <mesh 
          position={[0.6, 0, 0]} 
          onClick={() => setRotationSpeed(Math.min(0.01, rotationSpeed + 0.001))} 
          onPointerOver={() => document.body.style.cursor = 'pointer'} 
          onPointerOut={() => document.body.style.cursor = 'default'}
        >
          <boxGeometry args={[0.2, 0.1, 0.1]} />
          <meshStandardMaterial color={"#ffcc00"} />
        </mesh>
        <Text position={[0.6, 0.15, 0]} fontSize={0.08} color="white">
          +
        </Text>
      </group>
      
      {/* Rotating platform */}
      <group ref={platformRef}>
        {/* The actual wheel/platform */}
        <mesh position={[0, -0.5, 0]} rotation={[Math.PI/2, 0, 0]}>
          <cylinderGeometry args={[0.8, 0.8, 0.03, 32]} />
          <meshStandardMaterial map={texture} metalness={0.3} roughness={0.6} />
        </mesh>
        
        {/* Container for the 3D model */}
        <group position={[0, 0, 0]}>
          {children}
        </group>
      </group>
    </group>
  );
}

function DefaultModel() {
  return (
    <>
      <mesh>
        <sphereGeometry args={[0.5, 32, 32]} />
        <meshStandardMaterial color="cyan" />
      </mesh>
      <OrbitControls 
        enableZoom={true}
        zoomSpeed={0.5}
        minDistance={0.5}
        maxDistance={5}
        enablePan={false}
        minPolarAngle={0}
        maxPolarAngle={Math.PI}
        rotateSpeed={0.3}
        enableDamping={true}
        dampingFactor={0.05}
      />
    </>
  );
}

const ObjModelViewer = ({ jobId }) => {
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const modelRef = useRef(new THREE.Group());
  const API_BASE_URL = getApiBaseUrl();
  
  useEffect(() => {
    if (!jobId) return;
    
    setLoading(true);
    setError(null);
    
    // Clean up any previous models
    if (modelRef.current) {
      while(modelRef.current.children.length > 0) {
        const child = modelRef.current.children[0];
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach(m => m.dispose());
          } else {
            child.material.dispose();
          }
        }
        if (child.geometry) child.geometry.dispose();
        modelRef.current.remove(child);
      }
    }
    
    console.log("Loading 3D face model for job:", jobId);
    
    // Create loaders
    const mtlLoader = new MTLLoader();
    const objLoader = new OBJLoader();
    
    // Set paths for direct loading instead of loading from path
    const mtlUrl = `${API_BASE_URL}/api/download/${jobId}/face_mesh.mtl`;
    const objUrl = `${API_BASE_URL}/api/download/${jobId}/face_mesh.obj`;
    const textureUrl = `${API_BASE_URL}/api/download/${jobId}/face_texture.jpg`;
    
    // First check if files exist using fetch
    fetch(mtlUrl)
      .then(response => {
        if (!response.ok) throw new Error(`MTL file HTTP error: ${response.status}`);
        return response.text(); // Get the actual MTL content
      })
      .then(mtlContent => {
        // Parse the MTL content directly
        const materials = mtlLoader.parse(mtlContent, '');
        materials.preload();
        
        // Now load the OBJ file
        fetch(objUrl)
          .then(response => {
            if (!response.ok) throw new Error(`OBJ file HTTP error: ${response.status}`);
            return response.text();
          })
          .then(objContent => {
            // Parse the OBJ content directly
            objLoader.setMaterials(materials);
            const object = objLoader.parse(objContent);
            
            // Center the model
            const box = new THREE.Box3().setFromObject(object);
            const center = box.getCenter(new THREE.Vector3());
            object.position.sub(center);
            
            // Load the texture
            const textureLoader = new THREE.TextureLoader();
            textureLoader.load(
              textureUrl,
              (texture) => {
                // Use non-premultiplied alpha textures to avoid WebGL warnings
                texture.premultiplyAlpha = false;
                texture.flipY = false;
                
                // Apply texture to model materials
                object.traverse((child) => {
                  if (child instanceof THREE.Mesh) {
                    if (Array.isArray(child.material)) {
                      child.material.forEach(mat => {
                        mat.map = texture;
                        mat.transparent = true;
                        mat.needsUpdate = true;
                      });
                    } else {
                      child.material.map = texture;
                      child.material.transparent = true;
                      child.material.needsUpdate = true;
                    }
                  }
                });
                
                // Apply scaling and rotation
                object.scale.set(0.014, 0.014, 0.014);
                object.rotation.set(0, Math.PI, 0);
                object.position.set(0, 0.1, 0);
                
                // Add to scene
                modelRef.current.add(object);
                setModel(object);
                setLoading(false);
              },
              undefined,
              (error) => {
                console.error("Error loading texture:", error);
                // Still show model without texture
                object.scale.set(0.014, 0.014, 0.014);
                object.rotation.set(0, Math.PI, 0);
                object.position.set(0, 0.1, 0);
                
                modelRef.current.add(object);
                setModel(object);
                setLoading(false);
              }
            );
          })
          .catch(error => {
            console.error("Error loading OBJ file:", error);
            setError("Failed to load 3D model file");
            setLoading(false);
          });
      })
      .catch(error => {
        console.error("Error loading MTL file:", error);
        setError("Failed to load 3D model materials");
        setLoading(false);
      });
    
    // Cleanup on unmount
    return () => {
      if (modelRef.current) {
        while(modelRef.current.children.length > 0) {
          const child = modelRef.current.children[0];
          if (child.material) {
            if (Array.isArray(child.material)) {
              child.material.forEach(m => m.dispose());
            } else {
              child.material.dispose();
            }
          }
          if (child.geometry) child.geometry.dispose();
          modelRef.current.remove(child);
        }
      }
    };
  }, [jobId]);

  // Return appropriate content based on loading state
  if (error) {
    return (
      <mesh>
        <sphereGeometry args={[0.5, 32, 32]} />
        <meshStandardMaterial color="red" />
      </mesh>
    );
  }

  if (loading) {
    return (
      <mesh>
        <sphereGeometry args={[0.5, 32, 32]} />
        <meshStandardMaterial color="cyan" wireframe={true} />
      </mesh>
    );
  }

  return <primitive object={modelRef.current} />;
};

// Add an error boundary to catch Three.js errors
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error("3D Model error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="model-error">
          <p>3D model could not be displayed.</p>
          <p>You can still download the model files below.</p>
        </div>
      );
    }

    return this.props.children;
  }
}

function FaceGeneration() {
  // State variables
  const [seed, setSeed] = useState(42);
  const [modelType, setModelType] = useState('ffhq'); // Default model type
  const [status, setStatus] = useState({ message: '', type: '', visible: false });
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentJobId, setCurrentJobId] = useState(null);
  const [progress, setProgress] = useState({ percent: 0, message: '' });
  const [results, setResults] = useState(null);
  const [apiHealthy, setApiHealthy] = useState(false);
  const [showModel, setShowModel] = useState(false);
  const [availableModels, setAvailableModels] = useState({ ffhq: true, celeba: true });
  
  // API endpoint constants
  const API_BASE_URL = getApiBaseUrl();
  const API_HEALTH = `${API_BASE_URL}/api/health`;
  const API_GENERATE = `${API_BASE_URL}/api/generate`;
  const API_MODELS = `${API_BASE_URL}/api/models`;
  
  // Refs for DOM elements and animations
  const canvasRef = useRef(null);
  const headerRef = useRef(null);
  const generateSectionRef = useRef(null);
  
  // Setup 3D background on component mount
  useEffect(() => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    const renderer = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      alpha: true,
      antialias: true,
    });

    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const particlesGeometry = new THREE.BufferGeometry();
    const particlesCount = 2000;
    const posArray = new Float32Array(particlesCount * 3);
    for (let i = 0; i < particlesCount * 3; i++) {
      posArray[i] = (Math.random() - 0.5) * 10;
    }

    particlesGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(posArray, 3)
    );

    const particlesMaterial = new THREE.PointsMaterial({
      size: 0.02,
      color: 0x00ffff,
      transparent: true,
      opacity: 0.8,
    });

    const particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
    scene.add(particlesMesh);

    camera.position.z = 3;
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const animate = () => {
      requestAnimationFrame(animate);
      particlesMesh.rotation.x += 0.0005;
      particlesMesh.rotation.y += 0.0005;
      renderer.render(scene, camera);
    };

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('resize', handleResize);
    animate();

    return () => {
      window.removeEventListener('resize', handleResize);
      scene.remove(particlesMesh);
      particlesGeometry.dispose();
      particlesMaterial.dispose();
      renderer.dispose();
    };
  }, []);
  
  // Handle scroll animations
  useEffect(() => {
    const updateHeaderStyle = () => {
      if (headerRef.current) {
        if (window.scrollY > 50) {
          headerRef.current.classList.add('scrolled');
        } else {
          headerRef.current.classList.remove('scrolled');
        }
      }
    };

    window.addEventListener('scroll', updateHeaderStyle);

    const sections = document.querySelectorAll('section');
    sections.forEach((section) => {
      gsap.fromTo(
        section.querySelectorAll('h2, p, button'),
        { y: 50, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          stagger: 0.2,
          duration: 0.8,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: section,
            start: 'top 70%',
            end: 'bottom 20%',
            toggleActions: 'play none none reverse',
          },
        }
      );
    });

    return () => {
      window.removeEventListener('scroll', updateHeaderStyle);
      ScrollTrigger.getAll().forEach((trigger) => trigger.kill());
    };
  }, []);
  
  // Check API health and available models on component mount
  useEffect(() => {
    checkApiHealth();
    checkAvailableModels();
  }, []);
  
  const checkApiHealth = async () => {
    try {
      const response = await fetch(API_HEALTH);
      if (response.ok) {
        showStatus('Backend API is running.', 'success');
        setApiHealthy(true);
      } else {
        showStatus('Backend API is not responding properly.', 'error');
        setApiHealthy(false);
      }
    } catch (error) {
      showStatus('Cannot connect to backend API. Make sure it\'s running.', 'error');
      setApiHealthy(false);
    }
  };
  
  const checkAvailableModels = async () => {
    try {
      const response = await fetch(API_MODELS);
      if (response.ok) {
        const data = await response.json();
        setAvailableModels(data.available_models);
        
        // If current model isn't available, switch to an available one
        if (data.available_models && !data.available_models[modelType]) {
          // Find first available model
          const availableModel = Object.keys(data.available_models).find(key => data.available_models[key]);
          if (availableModel) {
            setModelType(availableModel);
            console.log(`Switched to available model: ${availableModel}`);
          }
        }
      }
    } catch (error) {
      console.error('Error checking available models:', error);
      // Assume both models are available if we can't check
      setAvailableModels({ ffhq: true, celeba: true });
    }
  };
  
  const startGeneration = async () => {
    // Reset UI
    resetUI();
    
    // Show loading status
    showStatus('Starting face generation process...', 'loading');
    setIsGenerating(true);
    updateProgress(10, 'Initializing...');
    
    try {
      // Generate mode - send model type and seed
      const requestData = { 
        seed: parseInt(seed) || 0,
        model_type: modelType
      };
      
      const response = await fetch(API_GENERATE, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(requestData)
      });
      
      if (!response.ok) {
        let errorMessage = 'Unknown error occurred';
        try {
          const errorData = await response.json();
          errorMessage = errorData.error || errorMessage;
        } catch (parseError) {
          errorMessage = `Server error: ${response.status} ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }
      
      // Process the response
      let result;
      try {
        result = await response.json();
      } catch (parseError) {
        throw new Error(`Error parsing server response: ${parseError.message}`);
      }
      
      if (!result || !result.job_id) {
        throw new Error('Invalid response from server');
      }
      
      setCurrentJobId(result.job_id);
      
      // Simulated progress updates
      updateProgress(25, `StyleGAN (${modelType}) generating face...`);
      
      setTimeout(() => {
        updateProgress(50, 'SRGAN upscaling image...');
        
        setTimeout(() => {
          updateProgress(75, 'Cropping face and preparing 3D model...');
          
          setTimeout(() => {
            // Update UI
            updateProgress(100, 'Generation complete!');
            showStatus('Face generation completed successfully!', 'success');
            
            // Display results
            setResults(result);
            setShowModel(true);
            setIsGenerating(false);
          }, 1000);
        }, 1000);
      }, 1000);
      
    } catch (error) {
      showStatus(`Error: ${error.message}`, 'error');
      updateProgress(0, 'Failed');
      setIsGenerating(false);
    }
  };
  
  const getFilenameFromPath = (path) => {
    // Extract the filename from a full path
    return path ? path.split('/').pop() : '';
  };
  
  const resetUI = () => {
    // Clear previous results
    setResults(null);
    
    // Reset progress
    updateProgress(0, '');
    
    // Clear previous job ID
    setCurrentJobId(null);
    
    // Hide 3D model
    setShowModel(false);
  };
  
  const showStatus = (message, type) => {
    setStatus({
      message,
      type,
      visible: true
    });
  };
  
  const updateProgress = (percent, message) => {
    setProgress({
      percent,
      message: message || `${percent}% complete`
    });
  };

  const generateRandomSeed = () => {
    const randomSeed = Math.floor(Math.random() * 1000);
    setSeed(randomSeed);
  };

  const renderModelViewer = () => {
    // Check if running in production on ngrok
    const isNgrokProduction = 
      import.meta.env.PROD && 
      window.location.hostname.includes('ngrok');
      
    // If using ngrok in production, use a simplified fallback
    if (isNgrokProduction) {
      return (
        <div className="model-fallback">
          <div className="model-fallback-message">
            <img 
              src={`${API_BASE_URL}/api/download/${currentJobId}/face_texture.jpg`}
              alt="3D Face Texture" 
              className="model-fallback-image"
            />
            <p>3D model ready! Use the download links below to view in your preferred 3D software.</p>
          </div>
        </div>
      );
    }
    
    // Normal 3D viewer for local development
    return (
      <div className="model-viewer">
        <Canvas 
          camera={{ 
            position: [0, 0, 2.5], 
            fov: 40, 
            near: 0.01, 
            far: 1000 
          }}
          gl={{ 
            antialias: true, 
            alpha: true,
            pixelRatio: window.devicePixelRatio
          }}
        >
          <color attach="background" args={["#262660"]} />
          <ambientLight intensity={1.0} />
          <directionalLight 
            position={[1, 1, 5]} 
            intensity={1.5} 
            castShadow
          />
          <spotLight
            position={[0, 1, 3]}
            angle={0.3}
            penumbra={0.8}
            intensity={1.0}
            castShadow
          />
          <Suspense fallback={
            <mesh>
              <sphereGeometry args={[0.5, 32, 32]} />
              <meshStandardMaterial color="cyan" wireframe={true} />
              <Text position={[0, 0, 0]} fontSize={0.1} color="white" anchorX="center" anchorY="middle">
                Loading...
              </Text>
            </mesh>
          }>
            <Platform autoRotate={true}>
              <ObjModelViewer jobId={currentJobId} />
            </Platform>
            <Environment preset="studio" />
          </Suspense>
          <OrbitControls 
            enableZoom={true}
            zoomSpeed={0.5}
            minDistance={1.0}
            maxDistance={10}
            enablePan={false}
            target={[0, 0, 0]}
            enableDamping={true}
            dampingFactor={0.05}
          />
        </Canvas>
        <Loader />
      </div>
    );
  };

  // Function to determine the API base URL
  function getApiBaseUrl() {
    // Check if we're running from ngrok (empty VITE_API_URL in production)
    if (import.meta.env.PROD && !import.meta.env.VITE_API_URL) {
      // Use the current window location as the base URL
      return window.location.origin;
    }
    // Otherwise use the configured API URL or default to localhost
    return import.meta.env.VITE_API_URL || 'http://localhost:5000';
  }

  return (
    <>
      <canvas ref={canvasRef} className="background-canvas" />
      <header ref={headerRef}>
        <div className="logo">
          <span className="logo-text">Face</span>
          <span className="logo-separator">→</span>
          <span className="logo-text">3D</span>
        </div>
        <nav>
          <ul className="menu">
            <li><a href="#home">Home</a></li>
            <li><a href="#about">About</a></li>
            <li><a href="#generate">Generate</a></li>
          </ul>
        </nav>
      </header>
      
      <main id="home" className="hero">
        <div className="hero-content">
          <h1 className="glitch-text" data-text="Face to 3D">
            Face to 3D
          </h1>
          <p className="hero-subtitle">
            Generate realistic 3D face models from StyleGAN with our
            AI-powered pipeline technology
          </p>
          <div className="hero-cta">
            <a href="#generate" className="cta-button primary">
              Start Creating
            </a>
            <a href="#about" className="cta-button secondary">
              Learn More
            </a>
          </div>
        </div>
      </main>
      
      <section id="about" className="about-section">
        <div className="section-inner">
          <h2>
            About <span className="highlight">Our Project</span>
          </h2>
          <div className="about-text">
            <p>
              This Face Generation and 3D Modeling project was developed as part of our Computer Vision and Generative Adversarial Networks course 
              at the Symbiosis Institute of Technology. It demonstrates the practical application of advanced 
              AI technologies in creating realistic 3D face models from generated 2D images.
            </p>
            <p>
              Our project combines multiple state-of-the-art technologies:
            </p>
          </div>
          <div className="about-grid">
            <div className="about-card">
              <div className="card-icon">
                <span className="material-icons">auto_awesome</span>
              </div>
              <h3>StyleGAN Image Generation</h3>
              <p>
                Our pipeline uses NVIDIA's StyleGAN to generate photorealistic face images that never existed before, 
                providing a diverse range of starting points for 3D modeling.
              </p>
            </div>
            <div className="about-card">
              <div className="card-icon">
                <span className="material-icons">hd</span>
              </div>
              <h3>SRGAN Upscaling</h3>
              <p>
                We implement Super-Resolution GAN technology to enhance image quality by 4x, providing the detail 
                necessary for high-quality textures in the 3D models.
              </p>
            </div>
            <div className="about-card">
              <div className="card-icon">
                <span className="material-icons">view_in_ar</span>
              </div>
              <h3>3D Face Morphing</h3>
              <p>
                Our custom face morphing algorithm extracts facial features using MediaPipe and generates a detailed 
                3D mesh with accurate texture mapping for realistic rendering.
              </p>
            </div>
          </div>
          <div className="about-conclusion">
            <p>
              This project demonstrates how multiple AI technologies can be combined in a unified pipeline to 
              create practical applications. The resulting 3D models can be used in games, virtual reality, 
              movie production, or for academic research.
            </p>
            <p>
              All processing happens on your local machine, ensuring your data remains private and secure.
            </p>
          </div>
        </div>
      </section>
      
      <section id="generate" ref={generateSectionRef} className="generate-section">
        <div className="section-inner">
          <h2>
            Generate Your <span className="highlight">3D Face</span>
          </h2>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.3 }}
            viewport={{ once: true }}
            className="generation-options"
          >
            <div className="model-selection">
              <label>Select Model:</label>
              {!availableModels.ffhq && !availableModels.celeba && (
                <div className="model-warning">
                  <p>No StyleGAN models available. Please check the server configuration.</p>
                </div>
              )}
              <div className="model-options">
                <label className={`model-option ${modelType === 'ffhq' ? 'selected' : ''} ${!availableModels.ffhq ? 'unavailable' : ''}`}>
                  <input
                    type="radio"
                    name="modelType"
                    value="ffhq"
                    checked={modelType === 'ffhq'}
                    onChange={() => availableModels.ffhq && setModelType('ffhq')}
                    disabled={!availableModels.ffhq}
                  />
                  <span className="checkmark"></span>
                  <div className="model-info">
                    <span className="model-name">FFHQ {!availableModels.ffhq && '(Not Available)'}</span>
                    <span className="model-desc">Diverse face generation with high quality details</span>
                  </div>
                </label>
                <label className={`model-option ${modelType === 'celeba' ? 'selected' : ''} ${!availableModels.celeba ? 'unavailable' : ''}`}>
                  <input
                    type="radio"
                    name="modelType"
                    value="celeba"
                    checked={modelType === 'celeba'}
                    onChange={() => availableModels.celeba && setModelType('celeba')}
                    disabled={!availableModels.celeba}
                  />
                  <span className="checkmark"></span>
                  <div className="model-info">
                    <span className="model-name">CelebA {!availableModels.celeba && '(Not Available)'}</span>
                    <span className="model-desc">Celebrity-like faces with consistent styling</span>
                  </div>
                </label>
              </div>
              {(!availableModels.ffhq || !availableModels.celeba) && (
                <div className="model-setup-hint">
                  <p>Some models are missing. See <code>MODEL_SETUP.md</code> for setup instructions.</p>
                  <p>The app will automatically use available models as fallback.</p>
                </div>
              )}
            </div>
            
            <div className="seed-input-container">
              <div className="seed-input-wrapper">
                <label htmlFor="seed">Seed Value (0-999):</label>
                <input
                  type="number"
                  id="seed"
                  min="0"
                  max="999"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                />
                <button 
                  className="random-seed-btn" 
                  onClick={generateRandomSeed}
                  title="Generate random seed"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M19 8L15 12H18C18 15.31 15.31 18 12 18C10.99 18 10.03 17.75 9.2 17.3L7.74 18.76C8.97 19.54 10.43 20 12 20C16.42 20 20 16.42 20 12H23L19 8Z" fill="currentColor"/>
                    <path d="M6 12C6 8.69 8.69 6 12 6C13.01 6 13.97 6.25 14.8 6.7L16.26 5.24C15.03 4.46 13.57 4 12 4C7.58 4 4 7.58 4 12H1L5 16L9 12H6Z" fill="currentColor"/>
                  </svg>
                </button>
              </div>
            </div>
            
            <button
              className="generate-btn"
              onClick={startGeneration}
              disabled={!apiHealthy || isGenerating}
            >
              <span>{isGenerating ? "Processing..." : "Generate 3D Face"}</span>
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M5 12H19"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M12 5L19 12L12 19"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </motion.div>
          
          {status.visible && (
            <div className={`status-message ${status.type}`}>
              {status.message}
            </div>
          )}
          
          {(progress.percent > 0 || isGenerating) && (
            <motion.div
              className="progress-bar-container"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <div className="progress-bar">
                <motion.div
                  className="progress-fill"
                  initial={{ width: 0 }}
                  animate={{ width: `${progress.percent}%` }}
                  transition={{ ease: "easeInOut" }}
                />
              </div>
              <div className="progress-text">{progress.message}</div>
            </motion.div>
          )}
          
          {results && (
            <div className="conversion-demo">
              <div className="conversion-images">
                <div className="image-container">
                  <h3>StyleGAN Image</h3>
                  <div className="demo-image">
                    {results.stylegan_image && (
                      <img 
                        src={`${API_BASE_URL}/api/download/${currentJobId}/${results.stylegan_image}`}
                        alt="StyleGAN Generated Image"
                        className="generated-image"
                      />
                    )}
                  </div>
                </div>
                
                <div className="conversion-arrow">
                  <svg
                    width="48"
                    height="48"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M5 12H19"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <path
                      d="M12 5L19 12L12 19"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </div>
                
                <div className="image-container">
                  <h3>Upscaled Image</h3>
                  <div className="demo-image">
                    {results.upscaled_image && (
                      <img 
                        src={`${API_BASE_URL}/api/download/${currentJobId}/${results.upscaled_image}`}
                        alt="SRGAN Upscaled Image"
                        className="generated-image"
                      />
                    )}
                  </div>
                </div>
                
                <div className="conversion-arrow">
                  <svg
                    width="48"
                    height="48"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M5 12H19"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <path
                      d="M12 5L19 12L12 19"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </div>
                
                <div className="image-container">
                  <h3>3D Files Ready</h3>
                  <div className="demo-image download-ready-container">
                    <div className="download-ready">
                      <div className="download-icon">
                        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M12 16L12 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                          <path d="M9 13L12 16L15 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                          <path d="M20 16.7428C21.2215 15.734 22 14.2079 22 12.5C22 9.46243 19.5376 7 16.5 7C16.2815 7 16.0771 6.886 15.9661 6.69774C14.6621 4.48484 12.2544 3 9.5 3C5.35786 3 2 6.35786 2 10.5C2 12.5661 2.83545 14.4371 4.18695 15.7935" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                          <path d="M8 17H16V21H8V17Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      </div>
                      <p>3D model has been generated successfully!</p>
                      <div className="download-all-button-container">
                        <a 
                          href={`${API_BASE_URL}/api/download/${currentJobId}/all`}
                          className="download-all-button"
                          download
                        >
                          <span>Download All Files</span>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 15L12 3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                            <path d="M7 10L12 15L17 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                            <path d="M20 21H4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                          </svg>
                        </a>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              {currentJobId && (
                <div className="download-section">
                  <h3>Download Individual Files</h3>
                  <div className="download-buttons">
                    <a 
                      href={`${API_BASE_URL}/api/download/${currentJobId}/face_mesh.obj`}
                      download
                      className="download-button"
                    >
                      Download OBJ
                    </a>
                    <a 
                      href={`${API_BASE_URL}/api/download/${currentJobId}/face_mesh.mtl`}
                      download
                      className="download-button"
                    >
                      Download MTL
                    </a>
                    <a 
                      href={`${API_BASE_URL}/api/download/${currentJobId}/face_texture.jpg`}
                      download
                      className="download-button"
                    >
                      Download Texture
                    </a>
                  </div>
                </div>
              )}
              
              {/* Demo Video Section */}
              {results && (
                <div className="demo-video-section">
                  <h3>How to Use Your 3D Model</h3>
                  <div className="demo-video-container">
                    <video 
                      className="demo-video" 
                      controls 
                      poster={`${API_BASE_URL}/api/assets/Screenshot 2025-04-22 205832.png`}
                    >
                      <source src={`${API_BASE_URL}/api/assets/demo.mp4`} type="video/mp4" />
                      Your browser does not support the video tag.
                    </video>
                  </div>
                  <div className="demo-instructions">
                    <h4>Follow these steps to use your 3D model:</h4>
                    <ol>
                      <li>Download all files using the button above</li>
                      <li>Extract the ZIP file to a folder on your computer</li>
                      <li>Open your preferred 3D software (Blender, Maya, 3DS Max, etc.)</li>
                      <li>Import the OBJ file - ensure the MTL and texture file are in the same folder</li>
                      <li>The model will automatically load with textures applied</li>
                      <li>You can now view, edit, animate, or export the model as needed</li>
                    </ol>
                    <p className="demo-note">
                      <strong>Note:</strong> The video above demonstrates how to import and work with your 3D face model in 
                      popular 3D software applications like Blender, MeshLab, Maya, 3DS Max, or other similar programs. 
                      The same principles apply to most 3D modeling software.
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </section>
      
      <footer>
        <div className="footer-content">
          <div className="footer-logo">
            <div className="logo">
              <span className="logo-text">Face</span>
              <span className="logo-separator">→</span>
              <span className="logo-text">3D</span>
            </div>
          </div>
        </div>
        <div className="footer-bottom">
          <p>© 2024 Face-to-3D. All rights reserved.</p>
        </div>
      </footer>
    </>
  );
}

export default FaceGeneration;