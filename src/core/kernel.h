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

struct boidAttrib {
	float2 velo;
	float2 accel;
	float rot;
	float wanderAngle;
	float wanderAngularVelo;
	float2 resultFlocking;//todo rausrefactorn
	float2 resultWander;
	float2 resultSeek;
	float2 resultCohesion;
	float2 resultAlignement;
	float2 resultSeperation;
	bool useDefaultColor = true;
	float4 color = make_float4(1.f, 0, 0, 1.f);
	bool useGoal = false;
	float2 goal;
};

// host functions
void init_kernel();
void launch_update_kernel();
void launch_vbo_kernel(float2 *pos, float4 *col);
void cleanupKernel();
void copy_host_to_device();
void update_configs(float *configs);
void initMatrices();
void launch_sorting_kernel();
void sortHostPosMatrix();
void scenarioDefault();
void scenarioFaceToFace();
void uploadSortedScenario();

// Kernel Functions
__global__ void init_states_kernel(unsigned int seed, curandState_t *states);
__global__ void vbo_pass(float2 *pos, float4 *col, float2 *posMat, boidAttrib *attribMat, float *configs);
__global__ void sorting_pass(float2 *posMat, boidAttrib *attribMat);
__global__ void simulation_pass(float2 *posMat, boidAttrib *attribMat, curandState_t *states, float *configs);

// Behaviour Functions
__device__ void seekBehaviour(unsigned int index, float2 *posMat, boidAttrib *attribMat, float *configs);
__device__ void flockingBehavior(unsigned int index, float2 *posMat, boidAttrib *attribMat, float *configs);
__device__ void wanderBehavior(unsigned int index, float2 *posMat, boidAttrib *attribMat, curandState_t *states, float *configs);

// Physics Functions
__device__ void applyVelocity(unsigned int index, float2 *posMat, boidAttrib *attribMat, float *configs);
__device__ void applyAcceleration(unsigned int index, boidAttrib *attribMat, float *configs);
__device__ void lookWhereYourGoing(unsigned int index, float2 *posMat, boidAttrib *attribMat);

// Helper Functions
__device__ float2 normalize2(float2 p);
__device__ float sqrLength2(float2 p);
__device__ float length2(float2 p);
__device__ int getGlobalIdx_3D_1D();
__device__ float2 limit(float2 v, float max);