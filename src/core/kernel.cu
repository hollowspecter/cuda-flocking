#include "kernel.h"
#include "defs.h"

// device variables
curandState_t *d_states;
float *d_configs;
float2 *d_mat_pos; 
boidAttrib *d_mat_attribs; 

// host variables
float *h_configs;
float2 *h_mat_pos; // the position matrix, basically sorting key
boidAttrib *h_mat_attribs; // the attribute matrix, that will get sorted as well with the same

const unsigned int threadsPerBlock = 1024;
const dim3 grid = dim3(4,4,1);

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
__global__ void vbo_pass(float2 *pos, float4 *col, float2 *posMat, boidAttrib *attribMat, float *configs) {
	//unsigned int boidIndex = threadIdx.x;
	unsigned int boidIndex = getGlobalIdx_3D_1D();
	unsigned int pointIndex = boidIndex * 6;
	//float rot1 = -rot[boidIndex] + 90, rot2 = rot1 - 140, rot3 = rot1 + 140;
	float rot1 = -1 * attribMat[boidIndex].rot + 90, rot2 = rot1 - 140, rot3 = rot1 + 140;
	float size = configs[BOID_SIZE];

	// first triangle
	pos[pointIndex].x = posMat[boidIndex].x;
	pos[pointIndex].y = posMat[boidIndex].y;
	pos[pointIndex + 1].x = posMat[boidIndex].x + cosf(DEG_TO_RAD(rot1)) * size;
	pos[pointIndex + 1].y = posMat[boidIndex].y + sinf(DEG_TO_RAD(rot1)) * size;
	pos[pointIndex + 2].x = posMat[boidIndex].x + cosf(DEG_TO_RAD(rot2)) * size;
	pos[pointIndex + 2].y = posMat[boidIndex].y + sinf(DEG_TO_RAD(rot2)) * size;

	// second triangle						  					 
	pos[pointIndex + 3].x = posMat[boidIndex].x;
	pos[pointIndex + 3].y = posMat[boidIndex].y;
	pos[pointIndex + 4].x = posMat[boidIndex].x + cosf(DEG_TO_RAD(rot1)) * size;
	pos[pointIndex + 4].y = posMat[boidIndex].y + sinf(DEG_TO_RAD(rot1)) * size;
	pos[pointIndex + 5].x = posMat[boidIndex].x + cosf(DEG_TO_RAD(rot3)) * size;
	pos[pointIndex + 5].y = posMat[boidIndex].y + sinf(DEG_TO_RAD(rot3)) * size;

	// Color first triangle
	float4 color = (attribMat[boidIndex].useDefaultColor)
		? make_float4(configs[DEFAULT_COLOR_R], configs[DEFAULT_COLOR_G], configs[DEFAULT_COLOR_B], 1.f)
		: attribMat[boidIndex].color;

	col[pointIndex] = color;
	col[pointIndex + 1] = color;
	col[pointIndex + 2] = color;
	col[pointIndex + 3] = color;
	col[pointIndex + 4] = color;
	col[pointIndex + 5] = color;
}
__global__ void sorting_pass(float2 *posMat, boidAttrib *attribMat) {
	unsigned int index = getGlobalIdx_3D_1D();
	unsigned int i;
	float2 ftmp;
	boidAttrib btmp;
	float2 left, right;
	
	// even columns
	i = 2 * index;
	left = posMat[i];
	right = posMat[i + 1];
	if ((left.x >  right.x) || // eq.1
		(left.x >= right.x - EPSILON && left.y < right.y)) { // eq.2

		ftmp = posMat[i];
		posMat[i] = posMat[i + 1];
		posMat[i + 1] = ftmp;

		btmp = attribMat[i];
		attribMat[i] = attribMat[i + 1];
		attribMat[i + 1] = btmp;
	}

	__syncthreads();
	

	// odd columns
	i = 2 * index + 1;
	left = posMat[i];
	right = posMat[i + 1];
	if ((i+1) % MAT_SIZE != 0 && // jump the end of one row bc its odd
		(left.x >  right.x || // eq.1
		(left.x >= right.x - EPSILON && left.y < right.y))) { // eq.2

		ftmp = posMat[i];
		posMat[i] = posMat[i + 1];
		posMat[i + 1] = ftmp;

		btmp = attribMat[i];
		attribMat[i] = attribMat[i + 1];
		attribMat[i + 1] = btmp;
	}

	__syncthreads();
	

	// even lines
		// x				 // y              *2x row-length
	i = (index % MAT_SIZE) + (index/MAT_SIZE) * 2 * MAT_SIZE;
	if (posMat[i].y < posMat[i + MAT_SIZE].y || //eq.3
		(posMat[i].y <= posMat[i + MAT_SIZE].y - EPSILON && posMat[i].x > posMat[i + MAT_SIZE].x)) { // eq.4

		ftmp = posMat[i];
		posMat[i] = posMat[i + MAT_SIZE];
		posMat[i + MAT_SIZE] = ftmp;

		btmp = attribMat[i];
		attribMat[i] = attribMat[i + MAT_SIZE];
		attribMat[i + MAT_SIZE] = btmp;
	}

	__syncthreads();
	

	// odd lines
	i = (index % MAT_SIZE) + (index/MAT_SIZE) * 2 * MAT_SIZE + MAT_SIZE;
	if (((i + MAT_SIZE) < NUMBER_OF_BOIDS) &&
		(posMat[i].y < posMat[i + MAT_SIZE].y || // eq.3
		(posMat[i].y <= posMat[i + MAT_SIZE].y - EPSILON && posMat[i].x > posMat[i + MAT_SIZE].x))) { // eq.4
		
		ftmp = posMat[i];
		posMat[i] = posMat[i + MAT_SIZE];
		posMat[i + MAT_SIZE] = ftmp;
	
		btmp = attribMat[i];
		attribMat[i] = attribMat[i + MAT_SIZE];
		attribMat[i + MAT_SIZE] = btmp;
	}
}
__global__ void simulation_pass(float2 *posMat, boidAttrib *attribMat, curandState_t *states, float *configs) {
	//unsigned int index = threadIdx.x;
	unsigned int index = getGlobalIdx_3D_1D();

	// reset color
	attribMat[index].useDefaultColor = true;

	///////////////behaviour pass
	if (configs[ENABLE_WANDER] > 0.f)
		wanderBehavior(index, posMat, attribMat, states, configs);
	else {
		attribMat[index].resultWander.x = 0.f;
		attribMat[index].resultWander.y = 0.f;
	}
	if (configs[ENABLE_FLOCKING] > 0.f)
		flockingBehavior(index, posMat, attribMat, configs);
	else {
		attribMat[index].resultCohesion.x = 0.f;
		attribMat[index].resultCohesion.y = 0.f;
		attribMat[index].resultSeperation.x = 0.f;
		attribMat[index].resultSeperation.y = 0.f;
		attribMat[index].resultAlignement.x = 0.f;
		attribMat[index].resultAlignement.y = 0.f;
	}
	if (configs[ENABLE_SEEK] > 0.f)
		seekBehaviour(index, posMat, attribMat, configs);
	else {
		attribMat[index].resultSeek.x = 0.f;
		attribMat[index].resultSeek.y = 0.f;
	}
	// weighted sum over all the behaviours
	//attribMat[index].accel.x =
	//	attribMat[index].resultWander.x * configs[WEIGHT_WANDER]
	//	+ attribMat[index].resultFlocking.x * configs[WEIGHT_FLOCKING]
	//	+ attribMat[index].resultSeek.x * configs[WEIGHT_SEEK];
	//attribMat[index].accel.y =
	//	attribMat[index].resultWander.y * configs[WEIGHT_WANDER]
	//	+ attribMat[index].resultFlocking.y * configs[WEIGHT_FLOCKING]
	//	+ attribMat[index].resultSeek.y * configs[WEIGHT_SEEK];

	// try again weighted sum
	float2 desiredVelo;
	desiredVelo.x = attribMat[index].resultWander.x * configs[WEIGHT_WANDER]
		+ attribMat[index].resultCohesion.x * configs[WEIGHT_COHESION]
		+ attribMat[index].resultAlignement.x * configs[WEIGHT_ALIGNEMENT]
		+ attribMat[index].resultSeperation.x * configs[WEIGHT_SEPERATION]
		+ attribMat[index].resultSeek.x * configs[WEIGHT_SEEK];
	desiredVelo.y = attribMat[index].resultWander.y * configs[WEIGHT_WANDER]
		+ attribMat[index].resultCohesion.y * configs[WEIGHT_COHESION]
		+ attribMat[index].resultAlignement.y * configs[WEIGHT_ALIGNEMENT]
		+ attribMat[index].resultSeperation.y * configs[WEIGHT_SEPERATION]
		+ attribMat[index].resultSeek.y * configs[WEIGHT_SEEK];
	desiredVelo = normalize2(desiredVelo);
	desiredVelo.x *= configs[BOID_MAX_VELOCITY];
	desiredVelo.y *= configs[BOID_MAX_VELOCITY];

	// write acceleration
	attribMat[index].accel.x += (desiredVelo.x - attribMat[index].velo.x);
	attribMat[index].accel.y += (desiredVelo.y - attribMat[index].velo.y);


	///////////////simulation pass
	applyAcceleration(index, attribMat, configs);
	lookWhereYourGoing(index, posMat, attribMat);
	applyVelocity(index, posMat, attribMat, configs);
}

////////////////////////////////////////////////////////////////////////////////
// BEHAVIOUR FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

__device__ void seekBehaviour(unsigned int index, float2 *posMat, boidAttrib *attribMat, float *configs) {
	//desired_velocity = normalize(target - position) * max_velocity
	//acceleration = desired_velocity - velocity
	float2 desired_velo;
	desired_velo.x = configs[GOAL_1_x] - posMat[index].x;
	desired_velo.y = configs[GOAL_1_y] - posMat[index].y;
	normalize2(desired_velo);
	desired_velo.x *= configs[BOID_MAX_VELOCITY];
	desired_velo.y *= configs[BOID_MAX_VELOCITY];

	// write it to the boid
	attribMat[index].resultSeek.x = (desired_velo.x - attribMat[index].velo.x);
	attribMat[index].resultSeek.y = (desired_velo.y - attribMat[index].velo.y);
}
__device__ void flockingBehavior(unsigned int index, float2 *posMat, boidAttrib *attribMat, float *configs) {

	// calculate the extended moore neighborhood
	int radius = configs[NEIGHBORHOOD_RADIUS];
	int startX = (index % MAT_SIZE) - radius;
	int startY = (index / MAT_SIZE) - radius;
	int endX = startX + 2 * radius;
	int endY = startY + 2 * radius;
	if (startX < 0) startX = 0;
	if (startY < 0) startY = 0;
	if (endX > MAT_SIZE) endX = MAT_SIZE;
	if (endY > MAT_SIZE) endY = MAT_SIZE;
	
	// implement alignment, cohesion and seperation vectors
	float2 alignment = make_float2(0.f, 0.f);
	float2 cohesion = make_float2(0.f, 0.f);
	float2 seperation = make_float2(0.f, 0.f);
	int numNeighborsAlignment = 0, numNeighborsCohesion = 0, numNeighborsSeperation = 0;

	for (int x = startX; x<endX; ++x)
		for (int y = startY; y < endY; ++y) {
			// calc index
			int i = x + y * MAT_SIZE; 

			// skip yourself
			if (i == index)
				continue;
			
			// debug color
			if (index == 100)
				attribMat[i].useDefaultColor = false;

			// calculate squareDistance
			float dx = posMat[index].x - posMat[i].x;
			float dy = posMat[index].y - posMat[i].y;
			float sqrDistance = dx * dx + dy * dy;

			// for every close neighbor, alignement
			float sqrThreshold = configs[DISTANCE_ALIGNEMENT];
			sqrThreshold *= sqrThreshold;
			if (sqrDistance < sqrThreshold) {
				numNeighborsAlignment++;
				alignment.x += attribMat[i].velo.x;
				alignment.y += attribMat[i].velo.y;
			}
			sqrThreshold = configs[DISTANCE_COHESION];
			sqrThreshold *= sqrThreshold;
			if (sqrDistance < sqrThreshold) {
				numNeighborsCohesion++;
				cohesion.x += posMat[i].x;
				cohesion.y += posMat[i].y;
			}
			sqrThreshold = configs[DISTANCE_SEPERATION];
			sqrThreshold *= sqrThreshold;
			if (sqrDistance < sqrThreshold) {
				float2 diff = make_float2(dx, dy);
				normalize2(diff);
				float distance = sqrt(sqrDistance);
				diff.x /= distance;
				diff.y /= distance;
				numNeighborsSeperation++;
				seperation.x += dx;
				seperation.y += dy;
			}
		}//endfor

	__syncthreads();

	 // no neighbors found?

	// alignement
	if (numNeighborsAlignment == 0) {
		alignment.x = 0.f;
		alignment.y = 0.f;
	}
	else {
		alignment.x /= (float)numNeighborsAlignment;
		alignment.x /= (float)numNeighborsAlignment;
		alignment = normalize2(alignment);
		alignment.x *= configs[BOID_MAX_VELOCITY];
		alignment.y *= configs[BOID_MAX_VELOCITY];
		alignment.x = (alignment.x - attribMat[index].velo.x);
		alignment.y = (alignment.y - attribMat[index].velo.y);
		alignment = limit(alignment, configs[BOID_MAX_ACCEL]);
	}

	// cohesion
	if (numNeighborsCohesion == 0) {
		cohesion.x = 0.f;
		cohesion.y = 0.f;
	}
	else {
		cohesion.x /= (float)numNeighborsCohesion;
		cohesion.y /= (float)numNeighborsCohesion;
		cohesion = make_float2(cohesion.x - posMat[index].x, cohesion.y - posMat[index].y);
		cohesion = normalize2(cohesion);
		cohesion.x *= configs[BOID_MAX_VELOCITY];
		cohesion.y *= configs[BOID_MAX_VELOCITY];
		cohesion.x = (cohesion.x - attribMat[index].velo.x);
		cohesion.y = (cohesion.y - attribMat[index].velo.y);
		cohesion = limit(cohesion, configs[BOID_MAX_ACCEL]);
	}

	// seperation
	if (numNeighborsSeperation > 0) {
		seperation.x /= numNeighborsSeperation;
		seperation.y /= numNeighborsSeperation;
	}
	if (length2(seperation) > EPSILON) {
		seperation = normalize2(seperation);
		seperation.x *= configs[BOID_MAX_VELOCITY];
		seperation.y *= configs[BOID_MAX_VELOCITY];
		seperation.x = (seperation.x - attribMat[index].velo.x);
		seperation.y = (seperation.y - attribMat[index].velo.y);
		seperation = limit(seperation, configs[BOID_MAX_ACCEL]);
	}

	attribMat[index].resultCohesion.x = cohesion.x;
	attribMat[index].resultCohesion.y = cohesion.y;
	attribMat[index].resultAlignement.x = alignment.x;
	attribMat[index].resultAlignement.y = alignment.y;
	attribMat[index].resultSeperation.x = seperation.x;
	attribMat[index].resultSeperation.y = seperation.y;
}
__device__ void wanderBehavior(unsigned int index, float2 *posMat, boidAttrib *attribMat, curandState_t *states, float *configs) {
	// wander behaviour from here: https://gamedevelopment.tutsplus.com/tutorials/understanding-steering-behaviors-wander--gamedev-1624
	float2 circleCenter = make_float2(0.0f, 0.0f),
		displacement = make_float2(0.0f, -1.f);
	// calculate circle center
	circleCenter.x = attribMat[index].velo.x;
	circleCenter.y = attribMat[index].velo.y;
	circleCenter = normalize2(circleCenter);
	circleCenter.x *= CENTER_OFFSET;
	circleCenter.y *= CENTER_OFFSET;

	// calculate displacement force
	displacement.x = cosf(attribMat[index].wanderAngle) * WANDER_RADIUS;
	displacement.y = sinf(attribMat[index].wanderAngle) * WANDER_RADIUS;

	// write it to the boid
	attribMat[index].resultWander.x = (circleCenter.x + displacement.x);
	attribMat[index].resultWander.y = (circleCenter.y + displacement.y);

	//attribMat[index].accel.x = (circleCenter.x + displacement.x) * configs[WEIGHT_WANDER];
	//attribMat[index].accel.y = (circleCenter.y + displacement.y) * configs[WEIGHT_WANDER];

	// move the circle point randomly on the circular path by changing the wanderAngle
	float wanderAngularAccel = (0.2*double(curand(&states[index])) / double(RAND_MAX) - 0.1);
	attribMat[index].wanderAngularVelo += 0.5f * wanderAngularAccel;
	CLAMP(-MAX_WANDER_VELO, attribMat[index].wanderAngularVelo, MAX_WANDER_VELO);
	attribMat[index].wanderAngle += 0.5f * attribMat[index].wanderAngularVelo;
}

////////////////////////////////////////////////////////////////////////////////
// PHYSICS FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

__device__ void applyVelocity(unsigned int index, float2 *posMat, boidAttrib *attribMat, float *configs) {
	// apply velocity
	posMat[index].x += DELTA_TIME * attribMat[index].velo.x;
	posMat[index].y += DELTA_TIME * attribMat[index].velo.y;

	// cap the position
	if (posMat[index].x > WINDOW_WIDTH) {
		posMat[index].x -= WINDOW_WIDTH;
	}
	if (posMat[index].x < 0) {
		posMat[index].x += WINDOW_WIDTH;
	}	  
	if (posMat[index].y > WINDOW_HEIGHT) {
		posMat[index].y -= WINDOW_HEIGHT;
	}	  
	if (posMat[index].y < 0) {
		posMat[index].y += WINDOW_HEIGHT;
	}
}
__device__ void lookWhereYourGoing(unsigned int index, float2 *posMat, boidAttrib *attribMat) {
	if (length2(attribMat[index].velo) > EPSILON) {
		attribMat[index].rot = RAD_TO_DEG(atan2(attribMat[index].velo.x, attribMat[index].velo.y));
	}
}
__device__ void applyAcceleration(unsigned int index, boidAttrib *attribMat, float *configs) {
	const float maxaccel = configs[BOID_MAX_ACCEL];
	const float maxvelo = configs[BOID_MAX_VELOCITY];

	// cap acceleration
	attribMat[index].accel = limit(attribMat[index].accel, maxaccel);

	// apply acceleration
	attribMat[index].velo.x += DELTA_TIME * attribMat[index].accel.x;
	attribMat[index].velo.y += DELTA_TIME * attribMat[index].accel.y;

	// cap velocity
	attribMat[index].velo = limit(attribMat[index].velo, maxvelo);
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
__device__ int getGlobalIdx_3D_1D()
{
	int blockId = blockIdx.x
		+ blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * blockDim.x + threadIdx.x;
	return threadId;
}
__device__ float2 limit(float2 v, float max) {
	float2 result = make_float2(v.x, v.y);
	if (length2(v) > max) {
		result = normalize2(result);
		result.x *= max;
		result.y *= max;
	}
	return result;
}

////////////////////////////////////////////////////////////////////////////////
// HOST FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

// called once, allocates all the memory on the cuda device
void init_kernel() {
	// check if the boid number and the grid size work out
	assert(NUMBER_OF_BOIDS == grid.x*grid.y*grid.z*threadsPerBlock);

	// allocate space for random states
	checkCudaErrors(cudaMalloc(&d_states, sizeof(curandState_t) * NUMBER_OF_BOIDS));
	init_states_kernel << <grid, threadsPerBlock >> >((unsigned int)time(0), d_states);

	// allocate and init configuration stuff
	cudaHostAlloc(&h_configs, sizeof(float) * NUM_OF_CONFIG_VARS, cudaHostAllocDefault);
	for (int i = 0; i < NUM_OF_CONFIG_VARS; ++i) {
		h_configs[i] = 1.f;
	}
	checkCudaErrors(cudaMalloc(&d_configs, sizeof(float) * NUM_OF_CONFIG_VARS));
	checkCudaErrors(cudaMemcpy(d_configs, h_configs, sizeof(float) * NUM_OF_CONFIG_VARS, cudaMemcpyHostToDevice));

	// initialise the boids in the matrices
	initMatrices();

	// copy all the host arrays to device
	copy_host_to_device();
}

// initialize the boid matrices
void initMatrices() {
	// allocate host matrices
	cudaHostAlloc(&h_mat_pos, sizeof(float2) * NUMBER_OF_BOIDS, cudaHostAllocDefault);
	cudaHostAlloc(&h_mat_attribs, sizeof(boidAttrib) * NUMBER_OF_BOIDS, cudaHostAllocDefault);

	// init host matrices
	for (int i = 0; i < NUMBER_OF_BOIDS; ++i) {
		h_mat_pos[i].x = (float)(rand() % WINDOW_WIDTH);
		h_mat_pos[i].y = (float)(rand() % WINDOW_HEIGHT);
		h_mat_attribs[i].velo.x = 0.f;
		h_mat_attribs[i].velo.y = 0.f;
		h_mat_attribs[i].accel.x = (float)(2.0*float(rand()) / float(RAND_MAX) - 1.0f) * 50.f;
		h_mat_attribs[i].accel.y = (float)(2.0*float(rand()) / float(RAND_MAX) - 1.0f) * 50.f;
		h_mat_attribs[i].rot = (float)(rand() % 360);
		h_mat_attribs[i].wanderAngle = (rand() % 100) / 100.f * 2.f * (float)M_PI;
		h_mat_attribs[i].wanderAngularVelo = (float)(0.1f*(2.0f*double(rand() + i) / double(RAND_MAX) - 1.0f));
		h_mat_attribs[i].resultFlocking.x = 0.f;
		h_mat_attribs[i].resultFlocking.y = 0.f;
		h_mat_attribs[i].resultSeek.x = 0.f;
		h_mat_attribs[i].resultSeek.y = 0.f;
		h_mat_attribs[i].resultWander.x = 0.f;
		h_mat_attribs[i].resultWander.y = 0.f;
		h_mat_attribs[i].color = make_float4(1.f, 0.f, 0.f, 1.f);
	}

	// sort host  matrix before uploading to device
	sortHostPosMatrix();

	// upload matrices data
	checkCudaErrors(cudaMalloc(&d_mat_pos, sizeof(float2) * NUMBER_OF_BOIDS));
	checkCudaErrors(cudaMalloc(&d_mat_attribs, sizeof(boidAttrib) * NUMBER_OF_BOIDS));
}

// sort the pos matrix on host
void sortHostPosMatrix() {
	// print 4x4 matrix not sorted
	std::cout << std::endl << "Unsortiert" << std::endl;
	for (int y = 0; y < 4; ++y) {
		for (int x = 0; x < 4; ++x) {
			std::cout << round(h_mat_pos[x + MAT_SIZE * y].x) << "|" << round(h_mat_pos[x + MAT_SIZE * y].y) << " & ";
		}
		std::cout << std::endl;
	}

	int swapCount = 0, passCount = 0;
	do {
		swapCount = 0;

		// bubble sort rows by pos.x
		for (int y = 0; y < MAT_SIZE; ++y)
			for (int n = MAT_SIZE; n > 1; --n)
				for (int x = 0; x < n - 1; ++x) {
					int index1 = x + MAT_SIZE*y,
						index2 = (x + 1) + MAT_SIZE*y;
					if (h_mat_pos[index1].x > h_mat_pos[index2].x || // eq. 1
						(h_mat_pos[index1].x >= h_mat_pos[index2].x - EPSILON && h_mat_pos[index1].y < h_mat_pos[index2].y) // eq.2
						) {

						float2 temp = h_mat_pos[index1];
						h_mat_pos[index1] = h_mat_pos[index2];
						h_mat_pos[index2] = temp;

						swapCount++;
					}

				}

		// bubble sort columns by pos.y
		for (int x = 0; x < MAT_SIZE; ++x)
			for (int n = MAT_SIZE; n > 1; --n)
				for (int y = 0; y < n - 1; ++y) {
					int index1 = x + MAT_SIZE*y,
						index2 = x + MAT_SIZE*y + MAT_SIZE;
					if (h_mat_pos[index1].y < h_mat_pos[index2].y || // eq. 3
						(h_mat_pos[index1].y <= h_mat_pos[index2].y - EPSILON && h_mat_pos[index1].x > h_mat_pos[index2].x) // eq. 4
						) {

						float2 temp = h_mat_pos[index1];
						h_mat_pos[index1] = h_mat_pos[index2];
						h_mat_pos[index2] = temp;

						swapCount++;
					}
				}

		std::cout << "Pass " << passCount++ << ": " << swapCount << " Swaps" << std::endl;
	} while (swapCount > 0);


	// print 4x4 matrix not sorted
	std::cout << std::endl << "Sortiert" << std::endl;
	for (int y = 0; y < 4; ++y) {
		for (int x = 0; x < 4; ++x) {
			std::cout << round(h_mat_pos[x + MAT_SIZE * y].x) << "|" << round(h_mat_pos[x + MAT_SIZE * y].y) << " & ";
		}
		std::cout << std::endl;
	}
}

// used to reset the positions
void copy_host_to_device() {
	// copy to device
	checkCudaErrors(cudaMemcpy(d_mat_pos, h_mat_pos, sizeof(float2) * NUMBER_OF_BOIDS, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_attribs, h_mat_attribs, sizeof(boidAttrib) * NUMBER_OF_BOIDS, cudaMemcpyHostToDevice));
}

void update_configs(float *configs) {
	for (int i = 0; i < NUM_OF_CONFIG_VARS; ++i) {
		h_configs[i] = configs[i];
	}
	checkCudaErrors(cudaMemcpy(d_configs, h_configs, sizeof(float) * NUM_OF_CONFIG_VARS, cudaMemcpyHostToDevice));
}

// launches the kernel that is doing the simulation step
void launch_update_kernel() {
	//update_kernel << <numBlocks, threadsPerBlock >> >(d_pos, d_velo, d_accel, d_rot, d_wanderAngle,
		//d_wanderAngularVelo, d_states, d_configs);

	simulation_pass << <grid, threadsPerBlock >> > (d_mat_pos, d_mat_attribs, d_states, d_configs);
}

// gets called to update the vbo
void launch_vbo_kernel(float2 *pos, float4 *col)
{
	vbo_pass << <grid, threadsPerBlock >> > (pos, col, d_mat_pos, d_mat_attribs, d_configs);
}

// launches kernel to sort the position matrix
void launch_sorting_kernel() {
	dim3 halfgrid = dim3(grid.x / 2, grid.y, grid.z);
	sorting_pass << < halfgrid, threadsPerBlock >> > (d_mat_pos, d_mat_attribs);
	//sorting_pass << < numBlocks / 2, threadsPerBlock >> > (d_mat_pos, d_mat_attribs);
	//sorting_pass << < numBlocks / 2, threadsPerBlock >> > (d_mat_pos, d_mat_attribs);
	//sorting_pass << < numBlocks / 2, threadsPerBlock >> > (d_mat_pos, d_mat_attribs);
}

// cleans up all the allocated memory on the device
void cleanupKernel() {
	// copy down from device to host
	checkCudaErrors(cudaMemcpy(h_mat_pos, d_mat_pos, sizeof(float2) * NUMBER_OF_BOIDS, cudaMemcpyDeviceToHost));
	
	// sort again to check
	sortHostPosMatrix();

	cudaFreeHost(h_mat_pos);
	cudaFreeHost(h_mat_attribs);

	cudaFree(d_mat_pos);
	cudaFree(d_mat_attribs);
}