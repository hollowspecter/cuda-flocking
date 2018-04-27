#pragma once

void initIMGUI();
void renderImgui();
void cleanupGui();

// callback functions
void guiMouse(int button, int state, int x, int y);
void guiKeyboard(unsigned char key, int /*x*/, int /*y*/);
void guiMousePos(int x, int y);