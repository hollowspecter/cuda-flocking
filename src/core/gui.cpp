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

	show_test_window = false;
	show_another_window = false;
	gui_reset_boids = false;
	weight_alignement = 1.0f;
	weight_cohesion = 1.0f;
	weight_seperation = 1.0f;
}

Gui::~Gui()
{
	ImGui_ImplGLUT_Shutdown();
}

////////////////////////////////////////////////////////////////////////////////
// WINDOW FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

void Gui::renderImgui()
{
	ImGui_ImplGLUT_NewFrame(WINDOW_WIDTH, WINDOW_HEIGHT);

	{
		if (ImGui::Button("Reset")) gui_reset_boids ^= 1;
		ImGui::SliderFloat("weight: alignement", &weight_alignement, 0.0f, 1.0f);
		ImGui::SliderFloat("weight: cohesion", &weight_cohesion, 0.0f, 1.0f);
		ImGui::SliderFloat("weight: seperation", &weight_seperation, 0.0f, 1.0f);
	}

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

////////////////////////////////////////////////////////////////////////////////
// GETTER
////////////////////////////////////////////////////////////////////////////////

float* Gui::getConfiguration() {
	float result[3] = { weight_alignement , weight_cohesion , weight_seperation };
	return &result[0];
}