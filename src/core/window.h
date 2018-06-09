#include "defs.h"
#include "kernel.h"
#ifndef WINDOW_H
#define WINDOW_H

class Gui;

class Window {
private:
	GLuint vboPos, vboCol;
	struct cudaGraphicsResource *cuda_vbo_resource_pos, *cuda_vbo_resource_col;
	Gui *pGui = 0;
	float zoom = 1.0f;
	int lastTime;
	int frames;
	char framerate[100];

	void initGL(int *argc, char **argv);
	void createVBO();
	void deleteVBO();
	void runCuda();
	
public:
	static Window *pWindow;

	Window(int *argc, char **argv);
	~Window();

	void renderScene(void);
	void gui();
	void closeCallback();
	void timerCallback(int value);
	void mouseCallback(int button, int state, int x, int y);
	void mouseMoveCallback(int x, int y);
	void mouseDragCallback(int x, int y);
	void keyboardCallback(unsigned char key, int /*x*/, int /*y*/);
	void mouseWheel(int button, int dir, int x, int y);
};
#endif //GUI_H