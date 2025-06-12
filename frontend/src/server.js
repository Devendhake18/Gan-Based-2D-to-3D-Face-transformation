import express from 'express';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import cors from 'cors';
import fs from 'fs/promises';
import { fileURLToPath } from 'url';

// Convert __dirname for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 5000;

const execPromise = promisify(exec);

app.use(cors({ origin: 'http://localhost:5173' }));
app.use(express.json());

// Output directory
const outputDir = path.join(
  __dirname,
  '..',
  '..',
  'Downloads',
  'GAN_CV',
  'ganCV',
  'src',
  'Models',
  'saved_images'
);

// Serve images with no-cache headers
app.use('/images', (req, res, next) => {
  res.set('Cache-Control', 'no-store, no-cache, must-revalidate, private');
  express.static(outputDir)(req, res, next);
});

// Ensure output directory exists
(async () => {
  try {
    await fs.mkdir(outputDir, { recursive: true });
    console.log(`Output directory ensured: ${outputDir}`);
  } catch (error) {
    console.error(`Failed to create output directory: ${error.message}`);
  }
})();

app.post('/generate-image', async (req, res) => {
  const pythonPath = 'C:\\Users\\Acer\\Anaconda3\\envs\\stylegan-pytorch\\python.exe';
  const genScript = 'C:\\Users\\Acer\\Documents\\GAN\\stylegan3\\gen_images.py';
  const modelPath = 'C:\\Users\\Acer\\Downloads\\st-08\\network-snapshot-000160.pkl';
  const condaActivateCommand = '"C:\\Users\\Acer\\Anaconda3\\Scripts\\activate.bat" stylegan-pytorch';

  // Command to run Python script in Conda environment
  const pythonCommand = `"${pythonPath}" "${genScript}" --outdir="${outputDir}" --trunc=1 --seeds=0 --network="${modelPath}"`;
  const fullCommand = `${condaActivateCommand} && ${pythonCommand}`;

  try {
    // Verify file paths
    const checks = [
      { path: pythonPath, name: 'Python' },
      { path: genScript, name: 'gen_images.py' },
      { path: modelPath, name: 'Model .pkl' },
    ];
    for (const { path: filePath, name } of checks) {
      if (!(await fs.access(filePath).then(() => true).catch(() => false))) {
        throw new Error(`${name} not found at ${filePath}`);
      }
    }

    console.log('Executing command:', fullCommand);

    // Execute command
    const { stdout, stderr } = await execPromise(fullCommand, { shell: 'cmd.exe' });

    if (stderr) {
      console.error(`Python stderr:\n${stderr}`);
      if (stderr.includes('error') || stderr.includes('failed')) {
        throw new Error(`Python script error: ${stderr}`);
      }
    }
    console.log(`Python stdout:\n${stdout}`);

    // Wait for file to be written (StyleGAN may take a moment to finalize)
    const expectedImage = 'seed0000.png';
    const imagePath = path.join(outputDir, expectedImage);
    let attempts = 0;
    const maxAttempts = 12; // Wait up to 60 seconds (5s * 12)
    while (attempts < maxAttempts) {
      if (await fs.access(imagePath).then(() => true).catch(() => false)) {
        console.log(`Image found: ${imagePath}`);
        return res.json({ imageUrl: `/images/${expectedImage}` });
      }
      console.log(`Waiting for image: ${imagePath}`);
      await new Promise((resolve) => setTimeout(resolve, 5000));
      attempts++;
    }

    throw new Error('Image not generated within timeout');
  } catch (error) {
    console.error(`Execution failed:\n${error.message}`);
    res.status(500).json({ error: 'Execution failed', details: error.message });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});