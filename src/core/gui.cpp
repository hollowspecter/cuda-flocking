#include "gui.h"
#include "..\imgui\imgui.h"
#include "imgui_impl_glut.h"
#include "GL\glew.h"
#include "GL\freeglut.h"
#include "defs.h"

////////////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS AND DESTRUCTORS, INIT FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

Gui::Gui()
{
	ImGui::CreateContext();
	ImGui_ImplGLUT_Init();

	show_debug_window = false;
	gui_reset_boids = false;
	gui_enable_flocking = true;
	gui_enable_wander = true;

	fillConfigs();
}

Gui::~Gui()
{
	delete[]configs;
	ImGui_ImplGLUT_Shutdown();
}

void Gui::fillConfigs() {
	// init configs
	configs = new float[NUM_OF_CONFIG_VARS];
	for (int i = 0; i < NUM_OF_CONFIG_VARS; ++i) {
		configs[i] = 1.0f;
	}
	configs[WEIGHT_ALIGNEMENT] = 0.7f;
	configs[WEIGHT_COHESION] = 0.89f;
	configs[BOID_MAX_VELOCITY] = MAX_VELOCITY;
	configs[BOID_MAX_ACCEL] = MAX_ACCELERATION;
	configs[BOID_SIZE] = 10.f;
	configs[DISTANCE_ALIGNEMENT] = 100.f;
	configs[DISTANCE_COHESION] = 100.f;
	configs[DISTANCE_SEPERATION] = 50.f;
	configs[GOAL_1_x] = 300.f;
	configs[GOAL_1_y] = 300.f;

	// init colors
	color_bg[0] = 24.f / 255.f;
	color_bg[1] = 26.f / 255.f;
	color_bg[2] = 33.f/255.f;
	color_boid[0] = 180.f / 255.f;
	color_boid[1] = 128.f / 255.f;
	color_boid[2] = 213.f / 255.f;
}

////////////////////////////////////////////////////////////////////////////////
// WINDOW FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

void Gui::renderImgui()
{
	ImGui_ImplGLUT_NewFrame(WINDOW_WIDTH, WINDOW_HEIGHT);

	ImGui::Begin("ImGui Demo", &show_debug_window);

	if (ImGui::Button("Reset Boids")) gui_reset_boids ^= 1;
	ImGui::Checkbox("Flocking Behaviour", &gui_enable_flocking);
	configs[ENABLE_FLOCKING] = (gui_enable_flocking) ? 1.0f : -1.0f;
	ImGui::Checkbox("Wander Behaviour", &gui_enable_wander);
	configs[ENABLE_WANDER] = (gui_enable_flocking) ? 1.0f : -1.0f;

	if (ImGui::CollapsingHeader("Environment Attributes")) {
		ImGui::ColorEdit3("background color", color_bg);
	}
	
	if (ImGui::CollapsingHeader("Boid Attributes"))
	{
		ImGui::SliderFloat("size", &configs[BOID_SIZE], 1.0f, 50.f);
		ImGui::ColorEdit3("background boid", color_boid);
		ImGui::SliderFloat("max velocity", &configs[BOID_MAX_VELOCITY], 0.0f, (float)MAX_VELOCITY * 2.f);
		ImGui::SliderFloat("max acceleration", &configs[BOID_MAX_ACCEL], 0.0f, (float)MAX_ACCELERATION * 2.f);
	}

	if (ImGui::CollapsingHeader("Flocking Behaviour"))
	{
		ImGui::Text("Alignement");
		ImGui::SliderFloat("weight: alignement", &configs[WEIGHT_ALIGNEMENT], 0.0f, 1.0f);
		ImGui::SliderFloat("distance: alignement", &configs[DISTANCE_ALIGNEMENT], 1.0f, 300.0f);
		ImGui::Text("Cohesion");
		ImGui::SliderFloat("weight: cohesion", &configs[WEIGHT_COHESION], 0.0f, 1.0f);
		ImGui::SliderFloat("distance: cohesion", &configs[DISTANCE_COHESION], 1.0f, 300.0f);
		ImGui::Text("Seperation");
		ImGui::SliderFloat("weight: seperation", &configs[WEIGHT_SEPERATION], 0.0f, 1.0f);
		ImGui::SliderFloat("distance: seperation", &configs[DISTANCE_SEPERATION], 1.0f, 300.0f);
	}

	if (ImGui::CollapsingHeader("Wander Behaviour"))
	{
		ImGui::SliderFloat("weight: wander", &configs[WEIGHT_WANDER], 0.0f, 1.0f);
	}

	ImGui::End();

	//static bool b = true;
	//ImGui::ShowDemoWindow(&b);

	ImGui::Render();
}

////////////////////////////////////////////////////////////////////////////////
// CALLBACK FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

void Gui::guiMouse(int button, int state, int x, int y) {
	ImGuiIO& io = ImGui::GetIO();
	guiMousePos(x, y);

	if (state == GLUT_DOWN && (button == GLUT_LEFT_BUTTON))
		io.MouseDown[0] = true;
	else
		io.MouseDown[0] = false;

	if (state == GLUT_DOWN && (button == GLUT_RIGHT_BUTTON))
		io.MouseDown[1] = true;
	else
		io.MouseDown[1] = false;
}
void Gui::guiKeyboard(unsigned char key, int /*x*/, int /*y*/) {
	ImGuiIO& io = ImGui::GetIO();
	io.AddInputCharacter(key);
}
void Gui::guiMousePos(int x, int y) {
	ImGuiIO& io = ImGui::GetIO();
	io.MousePos = ImVec2((float)x, (float)y + 227.f);
}
void Gui::guiMouseWheel(int button, int dir, int x, int y) {
	ImGuiIO& io = ImGui::GetIO();
	io.MousePos = ImVec2((float)x, (float)y);
	if (dir > 0)
	{
		io.MouseWheel = 1.0;
	}
	else if (dir < 0)
	{
		io.MouseWheel = -1.0;
	}
}

////////////////////////////////////////////////////////////////////////////////
// GETTER
////////////////////////////////////////////////////////////////////////////////

float* Gui::getConfiguration() {

	float result[NUM_OF_CONFIG_VARS];
	for (int i = 0; i < NUM_OF_CONFIG_VARS; ++i) {
		result[i] = configs[i];
	}
	return &result[0];
}

float* Gui::getBackgroundColor() {
	float result[3];
	memcpy(result, color_bg, sizeof(color_bg));
	return result;
}

float* Gui::getBoidColor() {
	float result[3];
	memcpy(result, color_boid, sizeof(color_boid));
	return result;
}