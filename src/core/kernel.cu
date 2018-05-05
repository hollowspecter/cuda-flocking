#include "kernel.h"
#include "defs.h"

// device variables
float2 *d_pos, *d_velo, *d_accel;
float *d_rot, *d_angVelo, *d_angAccel;
float *d_wanderAngle, *d_wanderAngularVelo;
curandState_t *d_states;
float *d_configs;

// host variables
float2 *h_pos, *h_velo, *h_accel;
float *h_rot;// , *d_angVelo, *d_angAccel;
float *h_wanderAngle, *h_wanderAngularVelo;
float *h_configs;

const unsigned int threadsPerBlock = 512;
const unsigned int numBlocks = 8;

////////////////////////////////////////////////////////////////////////////////
// CUDA KERNEL FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

__global__ void init_states_kernel(unsigned int seed, curandState_t *states) {

	/* we have to initialize the state */
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
		threadIdx.x, /* the sequence number should be different for each core (unless you want all
					 cores to get the same sequence of numbers for some reason - use thread id! */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&states[threadIdx.x]);
}

__global__ void copy_pos_kernel(float2 *pos, float2 *newpos, float *rot, float *configs)
{
	//unsigned int boidIndex = threadIdx.x;
	unsigned int boidIndex = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pointIndex = boidIndex * 6;
	float rot1 = -rot[boidIndex] + 90, rot2 = rot1 - 140, rot3 = rot1 + 140;
	float size = configs[BOID_SIZE];

	// first triangle
	pos[pointIndex].x = newpos[boidIndex].x;
	pos[pointIndex].y = newpos[boidIndex].y;
	pos[pointIndex + 1].x = newpos[boidIndex].x + cosf(DEG_TO_RAD(rot1)) * size;
	pos[pointIndex + 1].y = newpos[boidIndex].y + sinf(DEG_TO_RAD(rot1)) * size;
	pos[pointIndex + 2].x = newpos[boidIndex].x + cosf(DEG_TO_RAD(rot2)) * size;
	pos[pointIndex + 2].y = newpos[boidIndex].y + sinf(DEG_TO_RAD(rot2)) * size;

	// second triangle						  					 
	pos[pointIndex + 3].x = newpos[boidIndex].x;
	pos[pointIndex + 3].y = newpos[boidIndex].y;
	pos[pointIndex + 4].x = newpos[boidIndex].x + cosf(DEG_TO_RAD(rot1)) * size;
	pos[pointIndex + 4].y = newpos[boidIndex].y + sinf(DEG_TO_RAD(rot1)) * size;
	pos[pointIndex + 5].x = newpos[boidIndex].x + cosf(DEG_TO_RAD(rot3)) * size;
	pos[pointIndex + 5].y = newpos[boidIndex].y + sinf(DEG_TO_RAD(rot3)) * size;
}

__global__ void update_kernel(float2 *pos, float2 *velo, float2  *accel, float *rot,
	float *wanderAngle, float *wanderAngularVelo, curandState_t *states, float *configs)
{
	//unsigned int index = threadIdx.x;
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

	///////////////behaviour pass
	if (configs[ENABLE_WANDER] > 0.f)
		wanderBehavior2(index, pos, accel, velo, rot, wanderAngle, wanderAngularVelo, states, configs);
	if (configs[ENABLE_FLOCKING] > 0.f)
		flockingBehavior(index, pos, velo, accel, configs);
	if (configs[ENABLE_SEEK] > 0.f)
		seekBehaviour(index, pos, accel, velo, configs);

	///////////////simulation pass
	applyAcceleration(index, velo, accel, configs);
	lookWhereYourGoing(index, pos, velo, rot);
	applyVelocity(index, pos, velo, configs);
}

////////////////////////////////////////////////////////////////////////////////
// BEHAVIOUR FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

__device__ void flockingBehavior(unsigned int index, float2 *pos, float2 *velo, float2 *accel, float *configs) {
	// store the positions in a shared buffer
	// does not work with several blocks bc shared buffer only works in one block!
	/*__shared__ float2 posBuffer[NUMBER_OF_BOIDS];
	__shared__ float2 veloBuffer[NUMBER_OF_BOIDS];
	posBuffer[index].x = pos[index].x;
	posBuffer[index].y = pos[index].y;
	veloBuffer[index].x = velo[index].x;
	veloBuffer[index].y = velo[index].y;*/

	//__syncthreads(); // all the threads must be synced here, so the buffers are filled!

	// implement alignment, cohesion and seperation vectors
	float2 alignment = make_float2(0.f, 0.f);
	float2 cohesion = make_float2(0.f, 0.f);
	float2 seperation = make_float2(0.f, 0.f);

	int numNeighborsAlignement = 0, numNeighborsCohesion = 0, numNeighborsSeperation = 0;
	for (int i = 0; i < NUMBER_OF_BOIDS; ++i) {
		// skip yourself
		if (i == index)
			continue;

		// calculate squareDistance
		float dx = pos[index].x - pos[i].x;
		float dy = pos[index].y - pos[i].y;
		float sqrDistance = dx * dx + dy * dy;

		// for every close neighbor, alignement
		float sqrThreshold = configs[DISTANCE_ALIGNEMENT];
		sqrThreshold *= sqrThreshold;
		if (sqrDistance < sqrThreshold) {
			numNeighborsAlignement++;
			alignment.x += velo[i].x;
			alignment.y += velo[i].y;
		}
		sqrThreshold = configs[DISTANCE_COHESION];
		sqrThreshold *= sqrThreshold;
		if (sqrDistance < sqrThreshold) {
			numNeighborsCohesion++;
			cohesion.x += pos[i].x;
			cohesion.y += pos[i].y;
		}
		sqrThreshold = configs[DISTANCE_SEPERATION];
		sqrThreshold *= sqrThreshold;
		if (sqrDistance < sqrThreshold) {
			numNeighborsSeperation++;
			seperation.x += dx;
			seperation.y += dy;
		}
	}//endfor

	 // no neighbors found?
	if (numNeighborsAlignement == 0) {
		alignment.x = 0.f;
		alignment.y = 0.f;
	}
	if (numNeighborsCohesion == 0) {
		cohesion.x = 0.f;
		cohesion.y = 0.f;
	}
	else {
		cohesion.x /= numNeighborsCohesion;
		cohesion.y /= numNeighborsCohesion;
		cohesion = make_float2(cohesion.x - pos[index].x, cohesion.y - pos[index].y);
	}
	if (numNeighborsSeperation == 0) {
		seperation.x = 0.f;
		seperation.y = 0.f;
	}
	alignment = normalize2(alignment);
	cohesion = normalize2(cohesion);
	seperation = normalize2(seperation);

	float2 desiredVelo;
	desiredVelo.x = configs[WEIGHT_ALIGNEMENT] * alignment.x + configs[WEIGHT_COHESION] * cohesion.x + configs[WEIGHT_SEPERATION] * seperation.x;
	desiredVelo.y = configs[WEIGHT_ALIGNEMENT] * alignment.y + configs[WEIGHT_COHESION] * cohesion.y + configs[WEIGHT_SEPERATION] * seperation.y;
	desiredVelo = normalize2(desiredVelo);
	desiredVelo.x *= configs[BOID_MAX_VELOCITY];
	desiredVelo.y *= configs[BOID_MAX_VELOCITY];
	accel[index].x += (desiredVelo.x - velo[index].x) * configs[WEIGHT_FLOCKING];
	accel[index].y += (desiredVelo.y - velo[index].y) * configs[WEIGHT_FLOCKING];
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

__device__ void wanderBehavior2(unsigned int index, float2 *pos, float2 *accel, float2 *velo, float *rot, float *wanderAngle, float *wanderAngularVelo, curandState_t *states, float *configs) {
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

	accel[index].x = (circleCenter.x + displacement.x) * configs[WEIGHT_WANDER];
	accel[index].y = (circleCenter.y + displacement.y) * configs[WEIGHT_WANDER];

	// move the circle point randomly on the circular path by changing the wanderAngle
	float wanderAngularAccel = (0.2*double(curand(&states[index])) / double(RAND_MAX) - 0.1);
	wanderAngularVelo[index] += 0.5f * wanderAngularAccel;
	CLAMP(-MAX_WANDER_VELO, wanderAngularVelo[index], MAX_WANDER_VELO);
	wanderAngle[index] += 0.5f * wanderAngularVelo[index];
}

__device__ void seekBehaviour(unsigned int index, float2 *pos, float2 *accel, float2 *velo, float *configs) {
	//desired_velocity = normalize(target - position) * max_velocity
	//acceleration = desired_velocity - velocity
	float2 desired_velo;
	desired_velo.x = configs[GOAL_1_x] - pos[index].x;
	desired_velo.y = configs[GOAL_1_y] - pos[index].y;
	normalize2(desired_velo);
	desired_velo.x *= configs[BOID_MAX_VELOCITY];
	desired_velo.y *= configs[BOID_MAX_VELOCITY];
	accel[index].x = (desired_velo.x - velo[index].x) * configs[WEIGHT_SEEK];
	accel[index].y = (desired_velo.y - velo[index].y) * configs[WEIGHT_SEEK];
}

////////////////////////////////////////////////////////////////////////////////
// PHYSICS FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

__device__ void applyVelocity(unsigned int index, float2 *pos, float2 *velo, float *configs) {
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
__device__ void applyAcceleration(unsigned int index, float2 *velo, float2 *accel, float *configs) {
	const float maxaccel = configs[BOID_MAX_ACCEL];
	const float maxvelo = configs[BOID_MAX_VELOCITY];

	// cap acceleration
	if (length2(accel[index]) > maxaccel) {
		velo[index] = normalize2(velo[index]);
		velo[index].x *= maxaccel;
		velo[index].y *= maxaccel;
	}

	// apply acceleration
	velo[index].x += DELTA_TIME * accel[index].x;
	velo[index].y += DELTA_TIME * accel[index].y;

	// cap velocity
	if (length2(velo[index]) > maxvelo) {
		velo[index] = normalize2(velo[index]);
		velo[index].x *= maxvelo;
		velo[index].y *= maxvelo;
	}
}

////////////////////////////////////////////////////////////////////////////////
// HELPER FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

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
	return make_float2(px, py);
}
__device__ float length2(float2 p) {
	return sqrt(p.x * p.x + p.y * p.y);
}
__device__ float sqrLength2(float2 p) {
	return p.x * p.x + p.y * p.y;
}

////////////////////////////////////////////////////////////////////////////////
// HOST FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

// called once, allocates all the memory on the cuda device
void init_kernel() {
	// allocate host arrays
	cudaHostAlloc(&h_pos, sizeof(float2) * NUMBER_OF_BOIDS, cudaHostAllocDefault);
	cudaHostAlloc(&h_velo, sizeof(float2) * NUMBER_OF_BOIDS, cudaHostAllocDefault);
	cudaHostAlloc(&h_accel, sizeof(float2) * NUMBER_OF_BOIDS, cudaHostAllocDefault);
	cudaHostAlloc(&h_rot, sizeof(float) * NUMBER_OF_BOIDS, cudaHostAllocDefault);
	cudaHostAlloc(&h_wanderAngle, sizeof(float) * NUMBER_OF_BOIDS, cudaHostAllocDefault);
	cudaHostAlloc(&h_wanderAngularVelo, sizeof(float) * NUMBER_OF_BOIDS, cudaHostAllocDefault);

	// init host array
	for (int i = 0; i < NUMBER_OF_BOIDS; ++i) {
		h_pos[i].x = (float)(rand() % WINDOW_WIDTH);
		h_pos[i].y = (float)(rand() % WINDOW_HEIGHT);
		h_velo[i].x = 0.f;
		h_velo[i].y = 0.f;
		h_accel[i].x = (2.0*float(rand()) / float(RAND_MAX) - 1.0f) * (float)MAX_ACCELERATION;
		h_accel[i].y = (2.0*float(rand()) / float(RAND_MAX) - 1.0f) * (float)MAX_ACCELERATION;
		h_rot[i] = (float)(rand() % 360);
		h_wanderAngle[i] = (rand() % 100) / 100.f * 2.f * (float)M_PI;
		h_wanderAngularVelo[i] = 0.1f*(2.0f*double(rand() + i) / double(RAND_MAX) - 1.0f);
	}

	// allocate device arrays
	checkCudaErrors(cudaMalloc(&d_pos, sizeof(float2) * NUMBER_OF_BOIDS));
	checkCudaErrors(cudaMalloc(&d_velo, sizeof(float2) * NUMBER_OF_BOIDS));
	checkCudaErrors(cudaMalloc(&d_accel, sizeof(float2) * NUMBER_OF_BOIDS));
	checkCudaErrors(cudaMalloc(&d_rot, sizeof(float) * NUMBER_OF_BOIDS));
	checkCudaErrors(cudaMalloc(&d_wanderAngle, sizeof(float) * NUMBER_OF_BOIDS));
	checkCudaErrors(cudaMalloc(&d_wanderAngularVelo, sizeof(float) * NUMBER_OF_BOIDS));

	copy_host_to_device();

	// allocate space for random states
	checkCudaErrors(cudaMalloc(&d_states, sizeof(curandState_t) * NUMBER_OF_BOIDS));
	init_states_kernel << <numBlocks, threadsPerBlock >> >(time(0), d_states);

	// allocate and init configuration stuff
	cudaHostAlloc(&h_configs, sizeof(float) * NUM_OF_CONFIG_VARS, cudaHostAllocDefault);
	for (int i = 0; i < NUM_OF_CONFIG_VARS; ++i) {
		h_configs[i] = 1.f;
	}
	checkCudaErrors(cudaMalloc(&d_configs, sizeof(float) * NUM_OF_CONFIG_VARS));
	checkCudaErrors(cudaMemcpy(d_configs, h_configs, sizeof(float) * NUM_OF_CONFIG_VARS, cudaMemcpyHostToDevice));
}

// used to reset the positions
void copy_host_to_device() {
	// copy to device
	checkCudaErrors(cudaMemcpy(d_pos, h_pos, sizeof(float2) * NUMBER_OF_BOIDS, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_velo, h_velo, sizeof(float2) * NUMBER_OF_BOIDS, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_accel, h_accel, sizeof(float2) * NUMBER_OF_BOIDS, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_rot, h_rot, sizeof(float) * NUMBER_OF_BOIDS, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wanderAngle, h_wanderAngle, sizeof(float) * NUMBER_OF_BOIDS, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_wanderAngularVelo, h_wanderAngularVelo, sizeof(float) * NUMBER_OF_BOIDS, cudaMemcpyHostToDevice));
}

void update_configs(float *configs) {
	for (int i = 0; i < NUM_OF_CONFIG_VARS; ++i) {
		h_configs[i] = configs[i];
	}
	checkCudaErrors(cudaMemcpy(d_configs, h_configs, sizeof(float) * NUM_OF_CONFIG_VARS, cudaMemcpyHostToDevice));
}

// launches the kernel that is doing the simulation step
void launch_update_kernel() {
	update_kernel << <numBlocks, threadsPerBlock >> >(d_pos, d_velo, d_accel, d_rot, d_wanderAngle,
		d_wanderAngularVelo, d_states, d_configs);
}

// gets called to update the vbo
void launch_vbo_kernel(float2 *pos)
{
	//simple_vbo_kernel<<<1,1024>>>(pos, goal, weights);
	copy_pos_kernel << <numBlocks, threadsPerBlock >> >(pos, d_pos, d_rot, d_configs);
}

// cleans up all the allocated memory on the device
void cleanupKernel() {
	cudaFreeHost(h_pos);
	cudaFreeHost(h_velo);
	cudaFreeHost(h_accel);
	cudaFreeHost(h_rot);
	cudaFreeHost(h_wanderAngle);
	cudaFreeHost(h_wanderAngularVelo);
	cudaFreeHost(h_configs);

	cudaFree(d_pos);
	cudaFree(d_velo);
	cudaFree(d_accel);
	cudaFree(d_rot);
	cudaFree(d_wanderAngle);
	cudaFree(d_wanderAngularVelo);
	cudaFree(d_states);
	cudaFree(d_configs);
}