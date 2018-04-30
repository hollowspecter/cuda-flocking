#ifndef GUI_H
#define GUI_H

class Gui {
private:
	bool show_debug_window;
	float *configs;
public:
	bool gui_reset_boids;

	Gui();
	~Gui();
	void renderImgui();
	void guiMouse(int button, int state, int x, int y);
	void guiKeyboard(unsigned char key, int /*x*/, int /*y*/);
	void guiMousePos(int x, int y);
	float* getConfiguration();
};
#endif //GUI_H