import subprocess

command = [
    "python", "D:\\CODDING STUFF\\Sem 6\\CV_GAN\\Backend\\stylegan3\\train.py",
    "--outdir=D:\\ODDING STUFF\\Sem 6\\CV_GAN\Backend\\stylegan3\experiments",
    "--data=D:\\CODDING STUFF\\Sem 6\\CV_GAN\Backend\\stylegan3\\celeba_tfr_new",
    "--gpus=1",
    "--batch=16",
    "--snap=10",
    "--metrics=fid50k_full,is50k",
    "--gamma=8",
    "--aug=ada",
    "--cfg=stylegan3-t"
]

subprocess.run(command)
