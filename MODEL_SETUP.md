# Setting Up StyleGAN Models

This application supports multiple StyleGAN models for face generation. To use the models, you need to download and set them up in the correct directories.

## Directory Structure

The application expects the following directory structure:

```
Stylegan Model/
├── ffhq/
│   └── network-snapshot-000160.pkl
└── celeba/
    └── network-snapshot-000160.pkl
```

## Setup Instructions

1. **Create the directories**:
   - Create a folder named `Stylegan Model` in the application root directory
   - Inside that folder, create two subdirectories: `ffhq` and `celeba`

2. **Download the models**:

   For FFHQ model:
   - Download the FFHQ model file from [NVIDIA's StyleGAN repository](https://github.com/NVlabs/stylegan3) or another source
   - Rename it to `network-snapshot-000160.pkl` if needed
   - Place it in the `Stylegan Model/ffhq/` directory

   For CelebA model:
   - Download the CelebA model file from [NVIDIA's StyleGAN repository](https://github.com/NVlabs/stylegan3) or another source
   - Rename it to `network-snapshot-000160.pkl` if needed
   - Place it in the `Stylegan Model/celeba/` directory

3. **Verify the setup**:
   - Restart the application
   - Both models should be detected and available in the dropdown menu

## Alternative Models

If you have other StyleGAN models you'd like to use:

1. Create a new subdirectory inside the `Stylegan Model` folder with the model name
2. Place the model file inside that directory
3. Update the `app.py` file to include your new model:

```python
app.config['MODELS'] = {
    'ffhq': os.path.join('Stylegan Model', 'ffhq', 'network-snapshot-000160.pkl'),
    'celeba': os.path.join('Stylegan Model', 'celeba', 'network-snapshot-000160.pkl'),
    'your-model': os.path.join('Stylegan Model', 'your-model', 'network-snapshot-000160.pkl')
}
```

4. Restart the application to load the new model

## Troubleshooting

If you're having issues with the models:

- Ensure the file paths are correct
- Check that the model files have the correct format
- Verify that the model files have appropriate permissions
- Look for error messages in the console output
- Try running the application as administrator if permission issues occur 