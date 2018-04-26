/* Probleme / Fragen
* GUI?!
* 1. Ich krieg keine ordentlichen Random Numbers mit CuRand raus, was is das Problem?
* 2. Obstacle Avoidance
* 3. Goal Seeking
* 4. Predator Fleeing*/

#include "GL\glew.h"
#include "GL\freeglut.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <helper_gl.h>
#include <Windows.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop
#include "device_launch_parameters.h" // syncThreads

#include <vector_types.h>

// CUDA random numbers on GPU
#include <curand.h>
#include <curand_kernel.h>

// DEFINES
#ifndef M_PI
# define M_PI 3.14159265358979323846
#endif
#define REFRESH_DELAY (10) //ms
#define SQR_LOOK_DISTANCE 1000 // for flocking behaviour
#define MAX_VELOCITY 100.0
#define MAX_ACCELERATION 50.0
#define NUMBER_OF_BOIDS 1024
#define DELTA_TIME 0.0166
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define DEG_TO_RAD(a) (a * M_PI / 180.0)
#define RAD_TO_DEG(a) (a * 180.0 / M_PI) 
#define BOID_SIZE 10
#define EPSILON (0.000001)
#define CLAMP(a,b,c) { b = ((b) < (a))? (a) : (((b) > (c))? (c): (b));}

// wander
#define CENTER_OFFSET (150)
#define WANDER_RADIUS (70)
#define MAX_WANDER_VELO (0.2)


////////////////////////////////////////////////////////////////////////////////
// DECLARATIONS OF VARIABLES
////////////////////////////////////////////////////////////////////////////////
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource; // A handle to the registered object is returned as resource.
float timer = 0.0; // sort of
const unsigned int window_width = WINDOW_WIDTH;
const unsigned int window_height = WINDOW_HEIGHT;
unsigned int numVertices = 0; // is set in vbo create method
float2 goal;			// goal, alignment, cohesion, seperation
float4 weights = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
bool reset = false;

// device variables
float2 *d_pos, *d_velo, *d_accel;
float *d_rot, *d_angVelo, *d_angAccel;
float *d_wanderAngle, *d_wanderAngularVelo;
curandState_t *d_states;

////////////////////////////////////////////////////////////////////////////////
// DECLARATIONS OF METHODS
////////////////////////////////////////////////////////////////////////////////
void renderScene(void);
int main(int argc, char **argv);
bool initGL(int *argc, char **argv);
void cleanup();
void createVBO();
void deleteVBO();
void runCuda();
void launch_vbo_kernel(float2 *pos);
void timerEvent(int value);
void mouse(int button, int state, int x, int y);
void setTitle();
__device__ float2 normalize2(float2 p);
void keyboard(unsigned char key, int /*x*/, int /*y*/);
void init_kernel();
__global__ void copy_pos_kernel(float2 *pos, float2 *newpos, float *rot);
__global__ void update_kernel(float2 *pos, float2 *velo, float2  *accel, float *rot,
	float *wanderAngle, float *wanderAngularVelo, curandState_t *states);
void launch_update_kernel();
__device__ float length2(float2 p);
__device__ float sqrLength2(float2 p);
__device__ void applyAcceleration(unsigned int index, float2 *velo, float2 *accel);
__device__ void lookWhereYourGoing(unsigned int index, float2 *pos, float2 *velo, float *rot);
__device__ void applyVelocity(unsigned int index, float2 *pos, float2 *velo);
__global__ void init_states_kernel(unsigned int seed, curandState_t *states);
__device__ void wanderBehavior(unsigned int index, float2 *pos, float2 *accel, float2 *velo, float *rot, float *wanderAngle, float *wanderAngularVelo, curandState_t *states);
__device__ void wanderBehavior2(unsigned int index, float2 *pos, float2 *accel, float2 *velo, float *rot, float *wanderAngle, float *wanderAngularVelo, curandState_t *states);
__device__ void flockingBehavior(unsigned int index, float2 *pos, float2 *velo, float2 *accel);

////////////////////////////////////////////////////////////////////////////////
// CUDA METHODS
////////////////////////////////////////////////////////////////////////////////

__global__ void init_states_kernel(unsigned int seed, curandState_t *states) {

	/* we have to initialize the state */
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
		threadIdx.x, /* the sequence number should be different for each core (unless you want all
					cores to get the same sequence of numbers for some reason - use thread id! */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&states[threadIdx.x]);
}

__global__ void copy_pos_kernel(float2 *pos, float2 *newpos, float *rot)
{
	unsigned int boidIndex = threadIdx.x;
	unsigned int pointIndex = boidIndex * 6;
	float rot1 = -rot[boidIndex]+90, rot2 = rot1 - 140, rot3 = rot1 + 140;

	// first triangle
	pos[pointIndex].x     = newpos[boidIndex].x;
	pos[pointIndex].y     = newpos[boidIndex].y;
	pos[pointIndex + 1].x = newpos[boidIndex].x + cosf(DEG_TO_RAD(rot1)) * BOID_SIZE;
	pos[pointIndex + 1].y = newpos[boidIndex].y + sinf(DEG_TO_RAD(rot1)) * BOID_SIZE;
	pos[pointIndex + 2].x = newpos[boidIndex].x + cosf(DEG_TO_RAD(rot2)) * BOID_SIZE;
	pos[pointIndex + 2].y = newpos[boidIndex].y + sinf(DEG_TO_RAD(rot2)) * BOID_SIZE;
											  					 
	// second triangle						  					 
	pos[pointIndex + 3].x = newpos[boidIndex].x;				 
	pos[pointIndex + 3].y = newpos[boidIndex].y;				 
	pos[pointIndex + 4].x = newpos[boidIndex].x + cosf(DEG_TO_RAD(rot1)) * BOID_SIZE;
	pos[pointIndex + 4].y = newpos[boidIndex].y + sinf(DEG_TO_RAD(rot1)) * BOID_SIZE;
	pos[pointIndex + 5].x = newpos[boidIndex].x + cosf(DEG_TO_RAD(rot3)) * BOID_SIZE;
	pos[pointIndex + 5].y = newpos[boidIndex].y + sinf(DEG_TO_RAD(rot3)) * BOID_SIZE;
}

__global__ void update_kernel(float2 *pos, float2 *velo, float2  *accel, float *rot,
	float *wanderAngle, float *wanderAngularVelo, curandState_t *states)
{
	unsigned int index = threadIdx.x;

	//wanderBehavior2(index, pos, accel, velo, rot, wanderAngle, wanderAngularVelo, states);
	flockingBehavior(index, pos, velo, accel);
	
	///////////////physics
	applyAcceleration(index, velo, accel);
	lookWhereYourGoing(index, pos, velo, rot);
	applyVelocity(index, pos, velo);

	// curand test http://cs.umw.edu/~finlayson/class/fall16/cpsc425/notes/cuda-random.html
	//curand_init(673,0,0,&state);
	/*pos[index].x = curand(&states[index]) % window_width;
	pos[index].y = curand(&states[index]) % window_height;*/
}

__device__ void flockingBehavior(unsigned int index, float2 *pos, float2 *velo, float2 *accel) {
	// store the positions in a shared buffer
	__shared__ float2 posBuffer[1024]; // see cuda1 p.34
	__shared__ float2 veloBuffer[1024]; // see cuda1 p.34
	posBuffer[index].x = pos[index].x;
	posBuffer[index].y = pos[index].y;
	veloBuffer[index].x = velo[index].x;
	veloBuffer[index].y = velo[index].y;

	__syncthreads();

	// implement alignment, cohesion and seperation vectors
	float2 alignment = make_float2(0.f, 0.f);
	float2 cohesion = make_float2(0.f, 0.f);
	float2 seperation = make_float2(0.f, 0.f);

	int numNeighbors = 0;
	for (int i = 0; i < 1024; ++i) {
		// skip yourself
		if (i == index)
			continue;

		// calculate squareDistance
		float dx = posBuffer[index].x - posBuffer[i].x;
		float dy = posBuffer[index].y - posBuffer[i].y;
		float sqrDistance = dx * dx + dy * dy;

		// for every close neighbor
		if (sqrDistance < SQR_LOOK_DISTANCE) {
			numNeighbors++;

			alignment.x += veloBuffer[i].x;
			alignment.y += veloBuffer[i].y;
			cohesion.x += posBuffer[i].x;
			cohesion.y += posBuffer[i].y;
			seperation.x += dx;
			seperation.y += dy;
		}
	}//endfor

	 // no neighbors found?
	if (numNeighbors == 0) {
		alignment.x = 0.f;
		alignment.y = 0.f;
		cohesion.x = 0.f;
		cohesion.y = 0.f;
		seperation.x = 0.f;
		seperation.y = 0.f;
	}
	else {
		/*alignment.x /= numNeighbors;
		alignment.y /= numNeighbors;*/
		cohesion.x /= numNeighbors;
		cohesion.y /= numNeighbors;
		cohesion = make_float2(cohesion.x - posBuffer[index].x, cohesion.y - posBuffer[index].y);
		/*seperation.x /= numNeighbors;
		seperation.y /= numNeighbors;*/

		alignment = normalize2(alignment);
		cohesion = normalize2(cohesion);
		seperation = normalize2(seperation);
	}

	float2 desiredVelo;
	desiredVelo.x = alignment.x + cohesion.x + 1.5f*seperation.x;
	desiredVelo.y = alignment.y + cohesion.y + 1.5f*seperation.y;
	desiredVelo = normalize2(desiredVelo);
	desiredVelo.x *= MAX_VELOCITY;
	desiredVelo.y *= MAX_VELOCITY;
	accel[index].x += (desiredVelo.x - veloBuffer[index].x);
	accel[index].y += (desiredVelo.y - veloBuffer[index].y);
}
// doesnt work, always aligns on diagonal line
__device__ void wanderBehavior(unsigned int index, float2 *pos, float2 *accel, float2 *velo, float *rot, float *wanderAngle, float *wanderAngularVelo, curandState_t *states) {
	// wander behaviour
	float2 circleCenter = make_float2(0.0f, 0.0f),
		circlePoint = make_float2(0.0f, 0.0f),
		desiredPos = make_float2(0.0f, 0.0f),
		desiredVelo = make_float2(0.0f, 0.0f);
	// calculate circle center
	float currentRotation = DEG_TO_RAD(rot[index]);
	circleCenter.x = pos[index].x + CENTER_OFFSET * cosf(currentRotation);
	circleCenter.y = pos[index].y + CENTER_OFFSET * sinf(currentRotation);
	float circleAngle = wanderAngle[index] + DEG_TO_RAD(rot[index]);
	// calculate desired position
	circlePoint.x = WANDER_RADIUS * cos(circleAngle);
	circlePoint.y = WANDER_RADIUS * sin(circleAngle);
	desiredPos.x = circleCenter.x + circlePoint.x;
	desiredPos.y = circleCenter.y + circlePoint.y;
	// calculate desired velo and resulting acceleration
	desiredVelo.x = desiredPos.x - pos[index].x;
	desiredVelo.y = desiredPos.y - pos[index].y;
	accel[index].x = desiredVelo.x - velo[index].x;
	accel[index].y = desiredVelo.y - velo[index].y;

	// move the circle point randomly on the circular path
	// calculate a randomized acceleration for the circle point
	float wanderAngularAccel = (0.2*double(curand(&states[index])) / double(RAND_MAX) - 0.1);
	wanderAngularVelo[index] += 0.5f * wanderAngularAccel;
	CLAMP(-MAX_WANDER_VELO, wanderAngularVelo[index], MAX_WANDER_VELO);
	wanderAngle[index] += 0.5f * wanderAngularVelo[index];
}
// still problem with random numbers
__device__ void wanderBehavior2(unsigned int index, float2 *pos, float2 *accel, float2 *velo, float *rot, float *wanderAngle, float *wanderAngularVelo, curandState_t *states) {
	// wander behaviour from here: https://gamedevelopment.tutsplus.com/tutorials/understanding-steering-behaviors-wander--gamedev-1624
	float2 circleCenter = make_float2(0.0f, 0.0f),
		displacement = make_float2(0.0f, -1.f);
	// calculate circle center
	circleCenter.x = velo[index].x;
	circleCenter.y = velo[index].y;
	circleCenter = normalize2(circleCenter);
	circleCenter.x *= CENTER_OFFSET;
	circleCenter.y *= CENTER_OFFSET;

	// calculate displacement force
	displacement.x = cosf(wanderAngle[index]) * WANDER_RADIUS;
	displacement.y = sinf(wanderAngle[index]) * WANDER_RADIUS;

	accel[index].x = circleCenter.x + displacement.x;
	accel[index].y = circleCenter.y + displacement.y;

	// move the circle point randomly on the circular path by changing the wanderAngle
	float wanderAngularAccel = (0.2*double(curand(&states[index])) / double(RAND_MAX) - 0.1);
	wanderAngularVelo[index] += 0.5f * wanderAngularAccel;
	CLAMP(-MAX_WANDER_VELO, wanderAngularVelo[index], MAX_WANDER_VELO);
	wanderAngle[index] += 0.5f * wanderAngularVelo[index];
}

__device__ void applyVelocity(unsigned int index, float2 *pos, float2 *velo) {
	// apply velocity
	pos[index].x += DELTA_TIME * velo[index].x;
	pos[index].y += DELTA_TIME * velo[index].y;

	// cap the position
	if (pos[index].x > WINDOW_WIDTH) {
		pos[index].x -= WINDOW_WIDTH;
	}
	if (pos[index].x < 0) {
		pos[index].x += WINDOW_WIDTH;
	}
	if (pos[index].y > WINDOW_HEIGHT) {
		pos[index].y -= WINDOW_HEIGHT;
	}
	if (pos[index].y < 0) {
		pos[index].y += WINDOW_HEIGHT;
	}
}
__device__ void lookWhereYourGoing(unsigned int index, float2 *pos, float2 *velo, float *rot) {
	if (length2(velo[index]) > EPSILON) {
		rot[index] = RAD_TO_DEG(atan2(velo[index].x, velo[index].y));
	}
}
__device__ void applyAcceleration(unsigned int index, float2 *velo, float2 *accel) {
	// cap acceleration
	if (length2(accel[index]) > MAX_ACCELERATION) {
		velo[index] = normalize2(velo[index]);
		velo[index].x *= MAX_ACCELERATION;
		velo[index].y *= MAX_ACCELERATION;
	}
	
	// apply acceleration
	velo[index].x += DELTA_TIME * accel[index].x;
	velo[index].y += DELTA_TIME * accel[index].y;

	// cap velocity
	if (length2(velo[index]) > MAX_VELOCITY) {
		velo[index] = normalize2(velo[index]);
		velo[index].x *= MAX_VELOCITY;
		velo[index].y *= MAX_VELOCITY;
	}
}
__device__ float2 normalize2(float2 p)
{
	float length = sqrt(p.x * p.x + p.y * p.y);
	float px = p.x;
	float py = p.y;
	if (length <= FLT_EPSILON) {
		px = 0.f;
		py = 0.f;
	}
	else {
		px /= length;
		py /= length;
	}
	return make_float2(px,py);
}
__device__ float length2(float2 p) {
	return sqrt(p.x * p.x + p.y * p.y);
}
__device__ float sqrLength2(float2 p) {
	return p.x * p.x + p.y * p.y;
}

void init_kernel() {
	// speicher anfordern für 1024 objekte
	// position, velocity, acceleration
	// rotation, angular velo, angular accell
	float2 *h_pos, *h_velo, *h_accel;
	float *h_rot;// , *d_angVelo, *d_angAccel;
	float *h_wanderAngle, *h_wanderAngularVelo;
	cudaHostAlloc(&h_pos, sizeof(float2) * NUMBER_OF_BOIDS, cudaHostAllocDefault);
	cudaHostAlloc(&h_velo, sizeof(float2) * NUMBER_OF_BOIDS, cudaHostAllocDefault);
	cudaHostAlloc(&h_accel, sizeof(float2) * NUMBER_OF_BOIDS, cudaHostAllocDefault);
	cudaHostAlloc(&h_rot, sizeof(float) * NUMBER_OF_BOIDS, cudaHostAllocDefault);
	cudaHostAlloc(&h_wanderAngle, sizeof(float) * NUMBER_OF_BOIDS, cudaHostAllocDefault);
	cudaHostAlloc(&h_wanderAngularVelo, sizeof(float) * NUMBER_OF_BOIDS, cudaHostAllocDefault);
	checkCudaErrors(cudaMalloc(&d_pos, sizeof(float2) * NUMBER_OF_BOIDS));
	checkCudaErrors(cudaMalloc(&d_velo, sizeof(float2) * NUMBER_OF_BOIDS));
	checkCudaErrors(cudaMalloc(&d_accel, sizeof(float2) * NUMBER_OF_BOIDS));
	checkCudaErrors(cudaMalloc(&d_rot, sizeof(float) * NUMBER_OF_BOIDS));
	checkCudaErrors(cudaMalloc(&d_wanderAngle, sizeof(float) * NUMBER_OF_BOIDS));
	checkCudaErrors(cudaMalloc(&d_wanderAngularVelo, sizeof(float) * NUMBER_OF_BOIDS));

	// init host array
	for (int i = 0; i < NUMBER_OF_BOIDS; ++i) {
		h_pos[i].x = rand() % window_width;
		h_pos[i].y = rand() % window_height;
		h_velo[i].x = 0.f;
		h_velo[i].y = 0.f;
		h_accel[i].x = (2.0*float(rand()) / float(RAND_MAX) - 1.0f) * MAX_ACCELERATION;
		h_accel[i].y = (2.0*float(rand()) / float(RAND_MAX) - 1.0f) * MAX_ACCELERATION;
		h_rot[i] = rand() % 360;
		h_wanderAngle[i] = (rand() % 100)/100.f * 2 * M_PI;
		h_wanderAngularVelo[i] = 0.1*(2.0f*double(rand()+i) / double(RAND_MAX) - 1.0f);
	}
	
	// copy to device
	checkCudaErrors(cudaMemcpy(d_pos, h_pos, sizeof(float2) * NUMBER_OF_BOIDS, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_velo, h_velo, sizeof(float2) * NUMBER_OF_BOIDS, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_accel, h_accel, sizeof(float2) * NUMBER_OF_BOIDS, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_rot, h_rot, sizeof(float) * NUMBER_OF_BOIDS, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wanderAngle, h_wanderAngle, sizeof(float) * NUMBER_OF_BOIDS, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wanderAngularVelo, h_wanderAngularVelo, sizeof(float) * NUMBER_OF_BOIDS, cudaMemcpyHostToDevice));

	// free host
	cudaFreeHost(h_pos);
	cudaFreeHost(h_velo);
	cudaFreeHost(h_accel);
	cudaFreeHost(h_rot);
	cudaFreeHost(h_wanderAngle);
	cudaFreeHost(h_wanderAngularVelo);

	// allocate space for random states
	checkCudaErrors(cudaMalloc(&d_states, sizeof(curandState_t) * NUMBER_OF_BOIDS));
	init_states_kernel<<<1, 1024>>>(time(0), d_states);
}

void launch_update_kernel() {
	update_kernel<<<1,1024>>>(d_pos, d_velo, d_accel, d_rot, d_wanderAngle,
		d_wanderAngularVelo, d_states);
}

void launch_vbo_kernel(float2 *pos)
{							
	//simple_vbo_kernel<<<1,1024>>>(pos, goal, weights);
	copy_pos_kernel<<<1,1024>>>(pos, d_pos, d_rot);
}

////////////////////////////////////////////////////////////////////////////////
// METHODS
////////////////////////////////////////////////////////////////////////////////

// is called once in main
bool initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("OpenGL First Window");
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

	glewInit();
	std::cout << "\tOpen GL Version: " << glGetString(GL_VERSION) << std::endl; //4.6

	// register callbacks
	glutDisplayFunc(renderScene);
	glutKeyboardFunc(keyboard);
	glutCloseFunc(cleanup);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	glutMouseFunc(mouse);
	SDK_CHECK_ERROR_GL(); // cuda_helper

	// more init stuff
	glClearColor(0.0, 0.0, 0.0, 1.0);

	glViewport(0, 0, window_width, window_height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, window_width, 0, window_height, -500, 500);
	//gluOrtho2D(0.0f, 0.0f, (double)window_height, (double)window_width);
	//glMatrixMode(GL_MODELVIEW);

	return true;
}

// Is called from the Main Loop / DisplayFunc
void renderScene(void)
{
	// run cuda to modify vbo
	runCuda();

	// clear buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// bind vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 0, 0);

	// draw
	glEnableClientState(GL_VERTEX_ARRAY); // enables vertex array
	glColor3f(1.0, 1.0, 1.0);
	glDrawArrays(GL_TRIANGLES, 0, 3 * NUMBER_OF_BOIDS);
	glDisableClientState(GL_VERTEX_ARRAY);

	// swap buffers
	glutSwapBuffers();

	// some more stuff per frame
	timer += 0.1f;
}

// Is called On Close
void cleanup()
{
	std::cout << "Cleanup" << std::endl;
	deleteVBO();
	std::cout << "\tFreeing Cuda Memory" << std::endl;
	cudaFree(d_pos);
	cudaFree(d_velo);
	cudaFree(d_accel);
	cudaFree(d_rot);
	cudaFree(d_wanderAngle);
	cudaFree(d_wanderAngularVelo);
	cudaFree(d_states);
	std::cout << "\tGLUT: Finished" << std::endl;
	glutLeaveMainLoop();
}

// creates the vertex data and the vertex buffer object on gpu
void createVBO()
{
	assert(&vbo);
	std::cout << "\tCreating Buffer Object" << std::endl;

	// create buffer object
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	// uploading nothing but creating lots of vertices in vertex buffer
	std::cout << "\tUploading Data" << std::endl;
	unsigned int size = 6 * NUMBER_OF_BOIDS * sizeof(float2);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	
	// unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__TYPES_g2c3c3a69caaf333d29d0b38b75de5ffd.html#gg2c3c3a69caaf333d29d0b38b75de5ffd3d4fa7699e964ffc201daac20d2ecd6b
	std::cout << "\tRegister Buffer Object to CUDA" << std::endl;
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));

	SDK_CHECK_ERROR_GL();
}

// gets called in cleanup
void deleteVBO()
{
	std::cout << "\tDeleting VBO" << std::endl;

	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

	// delete vbo
	glBindBuffer(1, vbo);
	glDeleteBuffers(1, &vbo);

	vbo = 0;
}

// gets called every frame from renderScene
void runCuda()
{
	// update the boids
	launch_update_kernel();

	// map OpenGL buffer object for writing from CUDA
	float2 *dptr; // pointer on the positions
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cuda_vbo_resource));

	// launch cuda kernel to update VBO
	launch_vbo_kernel(dptr);

	// unmap buffer object (waits for all previous GPU activity to complete)
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}

// for refreshing the window
void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}

// called by mouse motion events
void mouse(int button, int state, int x, int y) {
	goal.x = x;
	goal.y = window_height - y;
	setTitle();
}

// sets the title for debug
void setTitle() {
	char title[256];
	sprintf(title, "HPC Abgabe (%4.2f|%4.2f)", goal.x, goal.y);
	glutSetWindowTitle(title);
}

// keyboard callback
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case (27): // escape
		glutDestroyWindow(glutGetWindow());
		return;
	case('r'): // R Button
		reset = true;
		return;
	}
}

////////////////////////////////////////////////////////////////////////////////
// MAIN METHOD
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
	// initialize Open GL
	std::cout << "Initializing GL Context" << std::endl;
	initGL(&argc, argv);
	
	// set the Cuda Device
	std::cout << "Setting Cuda Device" << std::endl;
	checkCudaErrors(cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId()));

	// init my positions of my boids
	std::cout << "Init Boids" << std::endl;
	init_kernel();

	// create VBO
	std::cout << "Creating the VBO" << std::endl;
	createVBO();

	// init goal
	goal.x = window_width - 50.0f;
	goal.y = window_height - 50.0f;
	setTitle();

	// run the main loop
	std::cout << "Start the Main Loop" << std::endl;
	glutMainLoop();

	// exit
	std::cout << "Input anything to exit..." << std::endl;
	char c;
	std::cin >> c;
	return 0;
}