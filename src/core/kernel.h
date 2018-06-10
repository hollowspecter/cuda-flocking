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
	float2 resultAvoidance;
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
void scenarioDefault(bool randomColor);
void scenarioFaceToFace();
void scenarioCross();

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
__device__ float2 avoidanceForce(float2 pa, float2 pb, float2 va, float2 vb);

// Helper Functions
__device__ float2 normalize2(float2 p);
__device__ float sqrLength2(float2 p);
__device__ float length2(float2 p);
__device__ int getGlobalIdx_3D_1D();
__device__ float2 limit(float2 v, float max);
__device__ float dot(float2 v1, float2 v2);


// SOURCE: https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		char c;
		std::cin >> c;
		exit(-1);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		char c;
		std::cin >> c;
		exit(-1);
	}
#endif

	return;
}