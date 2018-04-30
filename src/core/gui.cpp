#include "gui.h"
#include "..\imgui\imgui.h"
#include "imgui_impl_glut.h"
#include "GL\glew.h"
#include "GL\freeglut.h"
#include "defs.h"

////////////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS AND DESTRUCTORS
////////////////////////////////////////////////////////////////////////////////

Gui::Gui()
{
	ImGui::CreateContext();
	ImGui_ImplGLUT_Init();

	show_debug_window = false;
	gui_reset_boids = false;

	configs = new float[NUM_OF_CONFIG_VARS];
	for (int i = 0; i < NUM_OF_CONFIG_VARS; ++i) {
		configs[i] = 1.0f;
	}
	configs[BOID_MAX_VELOCITY] = MAX_VELOCITY;
	configs[BOID_MAX_ACCEL] = MAX_ACCELERATION;
	configs[BOID_SIZE] = 10.f;
}

Gui::~Gui()
{
	delete[]configs;
	ImGui_ImplGLUT_Shutdown();
}

////////////////////////////////////////////////////////////////////////////////
// WINDOW FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

void Gui::renderImgui()
{
	ImGui_ImplGLUT_NewFrame(WINDOW_WIDTH, WINDOW_HEIGHT);

	ImGui::Begin("ImGui Demo", &show_debug_window);

	if (ImGui::Button("Reset Boids")) gui_reset_boids ^= 1;
		
	if (ImGui::CollapsingHeader("Boid Attributes"))
	{
		ImGui::SliderFloat("size", &configs[BOID_SIZE], 1.0f, 50.f);
		ImGui::SliderFloat("max velocity", &configs[BOID_MAX_VELOCITY], 0.0f, (float)MAX_VELOCITY * 2.f);
		ImGui::SliderFloat("max acceleration", &configs[BOID_MAX_ACCEL], 0.0f, (float)MAX_ACCELERATION * 2.f);
	}

	if (ImGui::CollapsingHeader("Flocking Behaviour"))
	{
		ImGui::SliderFloat("weight: alignement", &configs[WEIGHT_ALIGNEMENT], 0.0f, 1.0f);
		ImGui::SliderFloat("weight: cohesion", &configs[WEIGHT_COHESION], 0.0f, 1.0f);
		ImGui::SliderFloat("weight: seperation", &configs[WEIGHT_SEPERATION], 0.0f, 1.0f);
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