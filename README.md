# cuda-flocking
A simple OpenGL application to simulate a flocking species using GPU programming with Nvidia's CUDA using the interopability between CUDA and OpenGL.

[current state gif](https://twitter.com/HollowSpecter/status/989494293009231872)

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
1. core/kernel.cu
2. core/window.cpp
3. core/gui.cpp

### kernel.h
Here's all the CUDA stuff is stored in.
There are three GPU passes.
1. Sorting pass: This is used to sort the data structure of the boids for optimization
2. Simulation pass: simulates the boids with all the different behaviours, applies, acceleration and velocity using euler integration
3. VBO pass: generates the VBO vertex data using the boids position and rotation

All the other functions are helper functions.

### window.cpp
Contains all the OpenGL code.
This example is just a simple OpenGL application with only one VBO and a single draw call.

### gui.cpp
Manages the imgui window and holds all the configurations.

## Flocking Behaviours
All the flocking simulation is done in ``kernel.h``. For each boid, following information is stored in seperate arrays, allocated on the device (GPU):
* float2 position
* float2 velocity
* float2 acceleration
* float rotation
* float angularVelocity
* float angularAcceleration

For simulation, all the boids are initialized in a random position, with a random direction (this is done in ``init_kernels()``). Each frame, via the ``launch_update_kernel`` function, a kernel is launched with one thread per boid on the GPU (the ``update_kernel``).
Here, the following steps are calculated for each boid:

1. Calculate the acceleration of the boid using [Craig Reynolds flocking behaviors](https://www.red3d.com/cwr/boids/). I've used [this tutorial](https://gamedevelopment.tutsplus.com/tutorials/3-simple-rules-of-flocking-behaviors-alignment-cohesion-and-separation--gamedev-3444).
2. Apply the acceleration on the velocity using simple explicit euler (``applyAcceleration()``) with ``velocity = velocity + acceleration * deltatime``. Make sure to cap acceleration and velocity!
3. Calculate the desired rotation with the function ``lookWhereYourGoing()``, done by some trigonometry
4. Apply the velocity on the position in ``applyVelocity()`` with ``position = position + velocity * deltatime``. Because I didn't want to loose my boids out of view and deal with scaling the camera or whatever, I also mapped them around in this function. Once I have Obstacle Avoidance, I could try to let them turn away from the screen-borders.

## Glossary
**kernel**: "CUDA C extends C by allowing the programmer to define C functions, called *kernels*, that, when called, are executed N times in parallel by N different CUDA threads, as opposed to only once like regular C functions." [source](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

**vbo**: "Vertex Buffer Object" [wiki](https://en.wikipedia.org/wiki/Vertex_buffer_object)

# Questions?
Shoot me an email: [baguiov[at]posteo.net](mailto:baguiov@posteo.net)

# Screenshots
