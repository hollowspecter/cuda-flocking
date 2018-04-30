#include "core\window.h"

////////////////////////////////////////////////////////////////////////////////
// MAIN METHOD
////////////////////////////////////////////////////////////////////////////////

Window *g_pWindow = 0;

int main(int argc, char **argv)
{
	// initialize Open GL, Cuda etc.
	g_pWindow = new Window(&argc, argv);

	// exit
	delete(g_pWindow);
	std::cout << "Input anything to exit..." << std::endl;
	char c;
	std::cin >> c;
	return 0;
}