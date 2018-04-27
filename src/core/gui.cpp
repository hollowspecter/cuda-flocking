#include "gui.h"
#include "..\imgui\imgui.h"
#include "imgui_impl_glut.h"
#include "GL\glew.h"
#include "GL\freeglut.h"
#include "defs.h"

////////////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS AND DESTRUCTORS
////////////////////////////////////////////////////////////////////////////////

// constructor
Gui::Gui()
{
	ImGui::CreateContext();
	ImGui_ImplGLUT_Init();
}

// destructor
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

		static float f = 0.0f;
		ImGui::Text("Hello, world!");
		ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
		if (ImGui::Button("Test Window")) show_test_window ^= 1;
		if (ImGui::Button("Another Window")) show_another_window ^= 1;
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
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
	io.MousePos = ImVec2((float)x, (float)y + 16.f);
}