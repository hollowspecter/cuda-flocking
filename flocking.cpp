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
	cleanupKernel();
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