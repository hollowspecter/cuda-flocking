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
	bool gui_enable_seek;
	bool gui_scenario_default = false;
	bool gui_scenario_faceToFace = false;
	bool gui_scenario_cross = false;
	bool gui_random_color = true;
	char framerate[100];

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
	void setPrimaryGoal(float x, float y);
};
#endif //GUI_H