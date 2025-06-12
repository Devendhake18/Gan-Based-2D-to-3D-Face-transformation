import subprocess

command = [
    "python", "D:\\CODDING STUFF\\Sem 6\\CV_GAN\\Backend\\stylegan3\\gen_images.py",
    "--outdir=D:\\CODDING STUFF\\Sem 6\\CV_GAN\\Backend\\stylegan_output",  # Set your desired output directory
    "--trunc=1",
    "--seeds=0",
    "--network=D:\\CODDING STUFF\\Backend\\Stylegan Model\\network-snapshot-000160.pkl"  # Set your .pkl model path here
]

subprocess.run(command)
