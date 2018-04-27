#pragma once
#ifndef GUI_H
#define GUI_H

class Gui {
private:
	bool gui_reset_boids;
	bool show_test_window;
	bool show_another_window;
public:
	Gui();
	~Gui();
	void renderImgui();
	void guiMouse(int button, int state, int x, int y);
	void guiKeyboard(unsigned char key, int /*x*/, int /*y*/);
	void guiMousePos(int x, int y);
};
#endif //GUI_H