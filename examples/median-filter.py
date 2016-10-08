import pyopencl as cl
import numpy as np
from scipy.misc import imread, imsave

#Read in image
img = imread('noisyImage.jpg', flatten=True).astype(np.float32)

# Get platforms, both CPU and GPU
plat = cl.get_platforms()
CPU = plat[0].get_devices()
try:
    GPU = plat[1].get_devices()
except IndexError:
    GPU = "none"

#Create context for GPU/CPU
if GPU!= "none":
    ctx = cl.Context(GPU)
else:
    ctx = cl.Context(CPU)

# Create queue for each kernel execution
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
