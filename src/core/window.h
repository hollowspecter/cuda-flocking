#include "defs.h"
#include "kernel.h"
#ifndef WINDOW_H
#define WINDOW_H

class Gui;

class Window {
private:
	GLuint vbo;
	struct cudaGraphicsResource *cuda_vbo_resource;
	Gui *pGui = 0;
public:
	static Window *pWindow;

	Window(int *argc, char **argv);
	~Window();

	void initGL(int *argc, char **argv);
	void renderScene(void);
	void closeCallback();
	void createVBO();
	void deleteVBO();
	void runCuda();
	void timerCallback(int value);
	void mouseCallback(int button, int state, int x, int y);
	void mouseMoveCallback(int x, int y);
	void mouseDragCallback(int x, int y);
	void keyboardCallback(unsigned char key, int /*x*/, int /*y*/);
};
#endif //GUI_H