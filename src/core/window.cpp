#include "window.h"
#include "gui.h"

Window *Window::pWindow = 0;

void renderSceneRedirect(void) { Window::pWindow->renderScene(); }
void closeCallbackRedirect() { Window::pWindow->closeCallback(); }
void timerCallbackRedirect(int value) { Window::pWindow->timerCallback(value); }
void mouseCallbackRedirect(int button, int state, int x, int y) { Window::pWindow->mouseCallback(button,state,x,y); }
void mouseMoveCallbackRedirect(int x, int y) { Window::pWindow->mouseMoveCallback(x,y); }
void mouseDragCallbackRedirect(int x, int y) { Window::pWindow->mouseDragCallback(x, y); }
void keyboardCallbackRedirect(unsigned char key, int x, int y) { Window::pWindow->keyboardCallback(key,x,y); }
void mouseWheelRedirect(int button, int state, int x, int y) { Window::pWindow->mouseWheel(button, state, x, y); }


////////////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS AND DESTRUCTORS
////////////////////////////////////////////////////////////////////////////////

//inits the opengl window
Window::Window(int *argc, char **argv) {
	Window::pWindow = this;

	// Init GUI
	std::cout << "Init DearImGUI" << std::endl;
	pGui = new Gui();

	// Init GL
	std::cout << "Init GL" << std::endl;
	initGL(argc, argv);

	// set the Cuda Device
	std::cout << "Setting Cuda Device" << std::endl;
	checkCudaErrors(cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId()));

	// init my positions of my boids
	std::cout << "Init Boids" << std::endl;
	init_kernel();

	// create VBO
	std::cout << "Creating the VBO" << std::endl;
	createVBO();

	// run the main loop
	std::cout << "Start the Main Loop" << std::endl;
	glutMainLoop();
}

Window::~Window() {
	delete(pGui);
}

void Window::initGL(int *argc, char **argv) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutCreateWindow("OpenGL First Window");
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutFullScreen();

	glewInit();
	std::cout << "\tOpen GL Version: " << glGetString(GL_VERSION) << std::endl; //4.6

	// register callbacks
	glutDisplayFunc(renderSceneRedirect);
	glutKeyboardFunc(keyboardCallbackRedirect);
	glutCloseFunc(closeCallbackRedirect);
	glutTimerFunc(REFRESH_DELAY, timerCallbackRedirect, 0);
	glutMouseFunc(mouseCallbackRedirect);
	glutMotionFunc(mouseDragCallbackRedirect);
	glutPassiveMotionFunc(mouseMoveCallbackRedirect);
	glutMouseWheelFunc(mouseWheelRedirect);
	SDK_CHECK_ERROR_GL(); // cuda_helper

	// more init stuff
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1);
}

////////////////////////////////////////////////////////////////////////////////
// CALLBACK FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

void Window::renderScene(void)
{
	// fetch updates from gui
	if (pGui->gui_reset_boids) {
		copy_host_to_device();
		pGui->gui_reset_boids = false;
	}
	update_configs(pGui->getConfiguration());

	// run cuda to modify vbo
	runCuda();

	// clear buffer
	float* color = pGui->getBackgroundColor();
	glClearColor(color[0], color[1], color[2], 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// redefine scaling matrix
	glLoadIdentity();
	glOrtho(0, WINDOW_WIDTH * zoom, 0, WINDOW_HEIGHT * zoom, -1, 1);

	// bind vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 0, 0);

	// draw
	glEnableClientState(GL_VERTEX_ARRAY); // enables vertex array
	color = pGui->getBoidColor();
	glColor3f(color[0], color[1], color[2]);
	glDrawArrays(GL_TRIANGLES, 0, 3 * NUMBER_OF_BOIDS);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// gui rendering
	pGui->renderImgui();

	// swap buffers
	glutSwapBuffers();
	glutPostRedisplay();
}

void Window::closeCallback() {
	std::cout << "Cleanup" << std::endl;
	deleteVBO();
	std::cout << "\tFreeing Cuda Memory" << std::endl;
	cleanupKernel();
	std::cout << "\tGLUT: Finished" << std::endl;
	glutLeaveMainLoop();
}

void Window::createVBO() {
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

void Window::deleteVBO()
{
	std::cout << "\tDeleting VBO" << std::endl;

	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

	// delete vbo
	glBindBuffer(1, vbo);
	glDeleteBuffers(1, &vbo);

	vbo = 0;
}

void Window::runCuda() {
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

void Window::timerCallback(int value) {
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerCallbackRedirect, 0);
	}
}

void Window::mouseCallback(int button, int state, int x, int y) {
	pGui->guiMouse(button, state, x, y);
	glutPostRedisplay();
}

void Window::mouseMoveCallback(int x, int y) {
	pGui->guiMousePos(x, y);
	glutPostRedisplay();
}

void Window::mouseDragCallback(int x, int y) {
	pGui->guiMousePos(x, y);
	glutPostRedisplay();
}

void Window::keyboardCallback(unsigned char key, int /*x*/, int /*y*/) {
	switch (key)
	{
	case (27): // escape
		glutDestroyWindow(glutGetWindow());
		return;
	}

	pGui->guiKeyboard(key, 0, 0);
}

void Window::mouseWheel(int button, int dir, int x, int y)
{
	/*if (dir > 0)
	{
		zoom -= 1.f/(1 << 4);
	}
	else if (dir < 0)
	{
		zoom += 1.f/(1 << 4);
	}*/
	//pGui->guiMouseWheel(button, dir, x, y);
	glutPostRedisplay();
}
