#pragma once

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