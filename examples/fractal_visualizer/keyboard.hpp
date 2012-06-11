#include "header.hpp"

void processNormalKeys(unsigned char key, int x, int y) {
    float scaler = 10;
	switch (key) 
	{
	case 27:		//ESC
		PostQuitMessage(0);
		break;
	case 'a':		
		Camera.RotateY(5.0);
		break;
	case 'd':		
		Camera.RotateY(-5.0);
		break;
	case 'w':		
		Camera.MoveForward( -0.1*scaler ) ;
		break;
	case 's':		
		Camera.MoveForward( 0.1*scaler ) ;
		break;
	case 'x':		
		Camera.RotateX(5.0);
		break;
	case 'y':		
		Camera.RotateX(-5.0);
		break;
	case 'c':		
		Camera.StrafeRight(-0.1*scaler);
		break;
	case 'v':		
		Camera.StrafeRight(0.1*scaler);
		break;
	case 'f':
		Camera.MoveUpward(-0.3*scaler);
		break;
	case 'r':
		Camera.MoveUpward(0.3*scaler);
		break;

	case 'm':
		Camera.RotateZ(-5.0);
		break;
	case 'n':
		Camera.RotateZ(5.0);
		break;

	}
}