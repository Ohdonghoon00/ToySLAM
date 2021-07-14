#include "GL/freeglut.h" 



void initialize_window()
{
    int mode = GLUT_RGB | GLUT_SINGLE;
    glutInitDisplayMode(mode);              // Set drawing surface property
    glutInitWindowPosition(200, 200);       // Set window Position at Screen
    glutInitWindowSize(1000,1000);          // Set window size. Set printed working area size. Bigger than this size
    glutCreateWindow("Trajectory");         // Generate window. argument is window's name

    glClearColor(1.0, 1.0, 1.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT); 
}

void show_trajectory(const GLfloat &x, const GLfloat &y, const GLfloat &z, const double r, const double g, const double b, const double size)
{
    glColor3f(r,g,b);
    glPointSize(size);
    glBegin(GL_POINTS);
    glVertex3f(x*0.001, z*0.001, y*-0.001);
    glEnd();
}
    
    
