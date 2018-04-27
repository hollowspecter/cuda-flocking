#pragma once

//class CExample {
//private:
//	bool gui_reset_boids;
//public:
//	CExample();
//	~CExample();
//	string getName();
//};
//#endif
//
////--End of Header--

void initIMGUI();
void renderImgui();
void cleanupGui();

// callback functions
void guiMouse(int button, int state, int x, int y);
void guiKeyboard(unsigned char key, int /*x*/, int /*y*/);
void guiMousePos(int x, int y);