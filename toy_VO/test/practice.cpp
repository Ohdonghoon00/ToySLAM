#include "opencv2/opencv.hpp"
#include "GL/freeglut.h" 
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unistd.h>
#include <cmath>
#include "opencv2/sfm/triangulation.hpp"
#include "opencv2/sfm/projection.hpp"
#include "../TOY_VO.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{       
    string input_image_path ("../../../dataset/sequences/");
    string cam_calib_path ("../../../dataset/sequences/");
    // string GT_data_path ("../../../dataset/poses/");
    
    string final_input_image_path = input_image_path + argv[1] + "/image_0/%06d.png";
    string final_came_calib_path = cam_calib_path + argv[1] + "/calib.txt";
    // string final_GT_data_path = GT_data_path + argv[1] + ".txt";

    ifstream calib_data(final_came_calib_path);

    double intrinsic[3][4];
    string buffer2[13];
    string tmp;
    
    getline(calib_data,tmp);
    istringstream tmp_arr(tmp);
    
    for (int i =0; i < tmp.size(); i++) tmp_arr >> buffer2[i];
     
    for(int i= 0; i < 3; i++)
        for(int j = 0; j < 4; j++)
            intrinsic[i][j] = stod(buffer2[4*i+j+1]);
            
    double f = intrinsic[0][0];
    cv::Point2d c(intrinsic[0][2],intrinsic[1][2]);
    const cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, c.x, 0, f, c.y, 0, 0, 1);
    std::vector<double> dist_coeff = { -0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 0.0001149909868437517, -0.01803978785585194 };
            
    std::cout << K << endl;
        
    // int mode = GLUT_RGB | GLUT_SINGLE;
    // glutInit(&argc, argv);
    // glutInitDisplayMode(mode);    // Set drawing surface property
    // glutInitWindowPosition(200, 200);    // Set window Position at Screen
    // glutInitWindowSize(1000,1000);    // Set window size. Set printed working area size. Bigger than this size
    // glutCreateWindow("Trajectory");    // Generate window. argument is window's name

    // glClearColor(1.0, 1.0, 1.0, 0.0);
    // glClear(GL_COLOR_BUFFER_BIT);    
        
    int min_inlier_num = 100;
    int inlier_num;
    // receive data
    cv::VideoCapture video;
    if (!video.open(final_input_image_path)) return -1;
    int times = 1;
    Frame previous_image(0);
    // for(int i = 0; i < 3; i++) previous_image.cam_pose[i] = 0;
    video >> previous_image.frame;

    // if (previous_image.empty()) break;
    if (previous_image.frame.channels() > 1) cv::cvtColor(previous_image.frame, previous_image.frame, cv::COLOR_RGB2GRAY);
    
    // ifstream GT_data(final_GT_data_path);
    
    // cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);
    // cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);
    // cv::Mat cam_pose_initialize = cv::Mat::eye(4, 4, CV_64F);
    // cv::Mat _cam_pose_ = cv::Mat::zeros(4, 1, CV_64F);
    // cv::Mat Iden = Mat::eye(4, 4, CV_64F);
    // std::vector<Point3f> map_point;
    // cv::Mat R, t, Rt;
    // cv::Mat World_R = cv::Mat::eye(3, 3, CV_64F), World_t = cv::Mat::zeros(3, 1, CV_64F), World_Rt;
    // std::vector<uchar> status;
    // cv::Mat err;
    // std::vector<cv::Point2f> projectionpoints;
    // std::vector<cv::Mat> normal, R_, t_;
    // cv::Mat X, map_change;
    // // Storage Storage_frame;
    // std::vector<Frame> frame_storage;

    while(true)
    {
        cv::Mat image;
        video >> image;
        if (image.empty()) break;
        if (image.channels() > 1) cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);

        cv::imshow("previous_image" , previous_image.frame);
        cv::imshow("image" , image);
        cv::waitKey();

        
    }
    return 0;
}