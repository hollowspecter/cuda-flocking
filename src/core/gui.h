#ifndef GUI_H
#define GUI_H

class Gui {
private:
	bool show_debug_window;
	float *configs;
	float color_bg[3];
	float color_boid[3];
public:
	bool gui_reset_boids;
	bool gui_enable_flocking;
	bool gui_enable_wander;

	Gui();
	~Gui();
	void fillConfigs();
	void renderImgui();
	void guiMouse(int button, int state, int x, int y);
	void guiKeyboard(unsigned char key, int /*x*/, int /*y*/);
	void guiMousePos(int x, int y);
	void guiMouseWheel(int button, int dir, int x, int y);

	float* getConfiguration();
	float* getBackgroundColor();
	float* getBoidColor();
};
#endif //GUI_H