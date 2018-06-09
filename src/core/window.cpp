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
	frames = 0;
	lastTime = glutGet(GLUT_ELAPSED_TIME);
}

////////////////////////////////////////////////////////////////////////////////
// CALLBACK FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

void Window::renderScene(void)
{
	// deal with gui
	gui();

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
	glBindBuffer(GL_ARRAY_BUFFER, vboPos);
	glVertexPointer(2, GL_FLOAT, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, vboCol);
	glColorPointer(4, GL_FLOAT, 0, 0);

	// draw
	glEnableClientState(GL_VERTEX_ARRAY); // enables vertex array
	glEnableClientState(GL_COLOR_ARRAY); // enables vertex array
	color = pGui->getBoidColor();
	glColor3f(color[0], color[1], color[2]);
	glDrawArrays(GL_TRIANGLES, 0, 6 * NUMBER_OF_BOIDS);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// gui rendering
	pGui->renderImgui();

	// swap buffers
	glutSwapBuffers();
	glutPostRedisplay();
}

void Window::gui() {
	// calculate framerate
	int currentTime = glutGet(GLUT_ELAPSED_TIME);
	frames++;
	if (currentTime - lastTime >= 1000)
	{
		sprintf(framerate, "FPS:%4.2f", frames*1000.0 / (currentTime - lastTime));
		lastTime = currentTime;
		frames = 0;
	}
	strncpy(pGui->framerate, framerate, 100);

	// fetch updates from gui
	update_configs(pGui->getConfiguration());

	// switch scenarios
	if (pGui->gui_scenario_faceToFace) {
		scenarioFaceToFace();
		copy_host_to_device();
		pGui->gui_scenario_faceToFace = false;
	}
	if (pGui->gui_scenario_cross) {
		scenarioCross();
		copy_host_to_device();
		pGui->gui_scenario_cross = false;
	}
	if (pGui->gui_scenario_default) {
		scenarioDefault();
		copy_host_to_device();
		pGui->gui_scenario_default = false;
	}

	// reset
	if (pGui->gui_reset_boids) {
		copy_host_to_device();
		pGui->gui_reset_boids = false;
	}
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
	assert(&vboPos);
	std::cout << "\tCreating Buffer Object" << std::endl;

	/* Position VBO */

	// create buffer object
	glGenBuffers(1, &vboPos);
	glBindBuffer(GL_ARRAY_BUFFER, vboPos);
	// uploading nothing but creating lots of vertices in vertex buffer
	std::cout << "\tUploading Data Position" << std::endl;
	unsigned int size = 6 * NUMBER_OF_BOIDS * sizeof(float2);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__TYPES_g2c3c3a69caaf333d29d0b38b75de5ffd.html#gg2c3c3a69caaf333d29d0b38b75de5ffd3d4fa7699e964ffc201daac20d2ecd6b
	std::cout << "\tRegister Position Buffer Object to CUDA" << std::endl;
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource_pos, vboPos, cudaGraphicsMapFlagsNone));

	SDK_CHECK_ERROR_GL();

	/* Color VBO */
	glGenBuffers(1, &vboCol);
	glBindBuffer(GL_ARRAY_BUFFER, vboCol);
	// uploading nothing but creating lots of vertices in vertex buffer
	std::cout << "\tUploading Color Data" << std::endl;
	size = 6 * NUMBER_OF_BOIDS * sizeof(float4);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__TYPES_g2c3c3a69caaf333d29d0b38b75de5ffd.html#gg2c3c3a69caaf333d29d0b38b75de5ffd3d4fa7699e964ffc201daac20d2ecd6b
	std::cout << "\tRegister Color Buffer Object to CUDA" << std::endl;
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource_col, vboCol, cudaGraphicsMapFlagsNone));

	SDK_CHECK_ERROR_GL();
}

void Window::deleteVBO()
{
	std::cout << "\tDeleting VBO" << std::endl;

	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource_pos));
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource_col));

	// delete vbo
	glBindBuffer(1, vboPos);
	glDeleteBuffers(1, &vboPos);
	glBindBuffer(1, vboCol);
	glDeleteBuffers(1, &vboCol);

	vboPos = 0;
	vboCol = 0;
}

void Window::runCuda() {
	// sorting pass
	launch_sorting_kernel();

	// update the boids
	launch_update_kernel();

	// map OpenGL buffer object for writing from CUDA
	float2 *dptrPos;
	float4 *dptrCol; // pointer on the positions
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource_pos, 0));
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource_col, 0));
	size_t num_bytes_pos, num_bytes_col;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptrPos, &num_bytes_pos, cuda_vbo_resource_pos));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptrCol, &num_bytes_col, cuda_vbo_resource_col));

	// launch cuda kernel to update VBO
	launch_vbo_kernel(dptrPos, dptrCol);

	// unmap buffer object (waits for all previous GPU activity to complete)
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource_pos, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource_col, 0));
}

void Window::timerCallback(int value) {
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerCallbackRedirect, 0);
	}
}

void Window::mouseCallback(int button, int state, int x, int y) {
	if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN))
	{
		pGui->setPrimaryGoal(x, WINDOW_HEIGHT - y);
	}
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
