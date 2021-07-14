#pragma once
#include "GL/freeglut.h" 
// #include <GL/gl.h>
#include "opencv2/opencv.hpp"


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
    glVertex3d(x*0.0011, z*0.0011 - 0.8, y*-0.0011);
    glEnd();
}

void show_trajectory_keyframe(cv::Mat rbt, const double r, const double g, const double b, const double size)
{
    glColor3f(r,g,b);
    glLineWidth(size);
    glBegin(GL_LINE_LOOP);
    for(int i = 0 ; i < 3; i++)
    {
        GLdouble x(rbt.at<double>(0, i)), y(rbt.at<double>(1, i)), z(rbt.at<double>(2, i));
        glVertex3d(x*0.0011, z*0.0011 - 0.8, y*-0.0011);
    }    
    
    // glVertex3d(x*0.003, z*0.003 - 0.8, y*-0.003);
    // glVertex3d(x*0.003, z*0.003 - 0.8, y*-0.003);
    glEnd();
}

void show_trajectory_left_keyframe_mini(const GLdouble &x, const GLdouble &y, const GLdouble &z, const double r, const double g, const double b, const double size)
{
    glColor3f(r,g,b);
    glLineWidth(size);
    glBegin(GL_LINE_LOOP);
    glVertex3d(x*0.001 - 0.01 - 0.5, z*0.001 - 0.01 + 0.5, y*-0.001 - 0.01);
    glVertex3d(x*0.001 + 0.01 - 0.5, z*0.001 - 0.01 + 0.5, y*-0.001 - 0.01);
    glVertex3d(x*0.001 - 0.5, z*0.001 + 0.01 + 0.5, y*-0.001);
    glEnd();
}

void show_trajectory_right_mini(const GLdouble &x, const GLdouble &y, const GLdouble &z, const double r, const double g, const double b, const double size)
{
    glColor3f(r,g,b);
    glPointSize(size);
    glBegin(GL_POINTS);
    glVertex3d(x*0.0007 + 0.5, z*0.0007+ 0.4, y*-0.0007);
    glEnd();
}
    
void show_trajectory_left_mini(const GLdouble &x, const GLdouble &y, const GLdouble &z, const double r, const double g, const double b, const double size)
{
    glColor3f(r,g,b);
    glPointSize(size);
    glBegin(GL_POINTS);
    glVertex3d(x*0.0007 - 0.5, z*0.0007 + 0.4, y*-0.0007);
    glEnd();
}