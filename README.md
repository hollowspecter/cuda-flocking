# cuda-flocking
A simple OpenGL application to simulate a flocking species using GPU programming with Nvidia's CUDA with the interopability between CUDA and OpenGL.

### OpenGL and CUDA
Using the CUDA Interopability, you can use CUDA and OpenGL together. To learn more about the interop, please read this
https://www.3dgep.com/opengl-interoperability-with-cuda/

The steps for the interopability that this project uses:
    1. Create an empty vertex buffer object (VBO)
    2. Register the VBO with Cuda
    3. Map the VBO for writing from Cuda
    4. Run Cuda kernel to modify the vertex positions
    5. Unmap the VBO
    6. Render the results using OpenGL
    

## The files
Currently in this project there are just a few files:
1. flocking.h
2. kernel.h
3. defs.h

### flocking.h
This file contains the main function and initializes all the OpenGL functions.

### kernel.h
Here's all the CUDA stuff is stored in.

* ``void init_kernel();``: Allocates the neccessary memory on the CUDA device and makes any kernel calls to initialize
* ``void launch_update_kernel();``: This launches the kernel ``update_kernel`` and basically represents one simulation step. This function is called once every frame.
* ``void launch_vbo_kernel(float2 *pos);``: This launches the kernel ``copy_pos_kernel``. It's also called once per frame and copies the the boid data into the vbo. Afterwards OpenGL renders it.


## Glossary
**kernel**: "CUDA C extends C by allowing the programmer to define C functions, called *kernels*, that, when called, are executed N times in parallel by N different CUDA threads, as opposed to only once like regular C functions." [source](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
**vbo**: "Vertex Buffer Object" [wiki](https://en.wikipedia.org/wiki/Vertex_buffer_object)

defs.h is just a bunch of defines.