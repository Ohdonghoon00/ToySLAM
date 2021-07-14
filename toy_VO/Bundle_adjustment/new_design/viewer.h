#include "GL/freeglut.h" 
// #include <GL/gl.h>


void initialize_window()
{
    int mode = GLUT_RGB | GLUT_SINGLE;
    glutInitDisplayMode(mode);              // Set drawing surface property
    glutInitWindowPosition(0, 0);       // Set window Position at Screen
    glutInitWindowSize(1000,1000);          // Set window size. Set printed working area size. Bigger than this size
    glutCreateWindow("GT and before BA trajectory");         // Generate window. argument is window's name

    glClearColor(1.0, 1.0, 1.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT); 
}

void initialize_window_for_BA()
{
    int mode = GLUT_RGB | GLUT_SINGLE;
    glutInitDisplayMode(mode);              // Set drawing surface property
    glutInitWindowPosition(1000, 0);       // Set window Position at Screen
    glutInitWindowSize(500,500);          // Set window size. Set printed working area size. Bigger than this size
    glutCreateWindow("GT and after BA trajectory");         // Generate window. argument is window's name

    glClearColor(1.0, 1.0, 1.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT); 
}

void show_trajectory(const GLdouble &x, const GLdouble &y, const GLdouble &z, const double r, const double g, const double b, const double size)
{
    glColor3f(r,g,b);
    glPointSize(size);
    glBegin(GL_POINTS);
    glVertex3d(x*0.001, z*0.001, y*-0.001);
    glEnd();
}

void show_trajectory_mini(const GLdouble &x, const GLdouble &y, const GLdouble &z, const double r, const double g, const double b, const double size)
{
    glColor3f(r,g,b);
    glPointSize(size);
    glBegin(GL_POINTS);
    glVertex3d(x*0.001 + 0.5, z*0.001, y*-0.0001);
    glEnd();
}

    
