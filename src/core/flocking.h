#pragma once
#include "defs.h"
#include "kernel.h"

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
void timerEvent(int value);
void mouse(int button, int state, int x, int y);
void setTitle();
void keyboard(unsigned char key, int /*x*/, int /*y*/);
void initIMGUI();
void imguiWindow();