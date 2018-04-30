#pragma once

#include "GL\glew.h"
#include "GL\freeglut.h"

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <curand_kernel.h>

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop
#include "device_launch_parameters.h" // syncThreads

// methods
void init_kernel();
void launch_update_kernel();
void launch_vbo_kernel(float2 *pos);
void cleanupKernel();
void copy_host_to_device();
void update_configs(float *configs);

__global__ void copy_pos_kernel(float2 *pos, float2 *newpos, float *rot);
__global__ void update_kernel(float2 *pos, float2 *velo, float2  *accel, float *rot,
	float *wanderAngle, float *wanderAngularVelo, curandState_t *states, float *configs);
__device__ float length2(float2 p);
__device__ float sqrLength2(float2 p);
__device__ void applyAcceleration(unsigned int index, float2 *velo, float2 *accel, float *configs);
__device__ void lookWhereYourGoing(unsigned int index, float2 *pos, float2 *velo, float *rot);
__device__ void applyVelocity(unsigned int index, float2 *pos, float2 *velo, float *configs);
__global__ void init_states_kernel(unsigned int seed, curandState_t *states);
__device__ void wanderBehavior(unsigned int index, float2 *pos, float2 *accel, float2 *velo, float *rot, float *wanderAngle, float *wanderAngularVelo, curandState_t *states);
__device__ void wanderBehavior2(unsigned int index, float2 *pos, float2 *accel, float2 *velo, float *rot, float *wanderAngle, float *wanderAngularVelo, curandState_t *states, float *configs);
__device__ void flockingBehavior(unsigned int index, float2 *pos, float2 *velo, float2 *accel, float *configs);
__device__ float2 normalize2(float2 p);
