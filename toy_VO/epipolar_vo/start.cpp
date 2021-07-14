#include "opencv2/opencv.hpp"
#include "GL/freeglut.h" 
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unistd.h>
#include <cmath>
#include <sstream>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{       
    string input_image_path ("../../../dataset/sequences/");
    string cam_calib_path ("../../../dataset/sequences/");
    string GT_data_path ("../../../dataset/poses/");
    
    string final_input_image_path = input_image_path + argv[1] + "/image_0/%06d.png";
    string final_came_calib_path = cam_calib_path + argv[1] + "/calib.txt";
    string final_GT_data_path = GT_data_path + argv[1] + ".txt";

    ifstream calib_data(final_came_calib_path);

    double intrinsic[3][4];
    string buffer2[13];
    string tmp;
    
    getline(calib_data,tmp);
    istringstream tmp_arr(tmp);
    
    for (int i =0; i < tmp.size(); i++) tmp_arr >> buffer2[i];
     
    for(int i= 0; i < 3; i++){
        for(int j = 0; j < 4; j++){
            cout << buffer2[4*i+j+1];
            intrinsic[i][j] = stod(buffer2[4*i+j+1]);
            cout << intrinsic[i][j] << " ";
        }
        cout<<endl;
    }
    // calib_data.close();


    double f = intrinsic[0][0];
    // double cx = intrinsic[0][2];
    // double cy = intrinsic[1][2];
    cv::Point2d c(intrinsic[0][2],intrinsic[1][2]);
    // Point2d image_center(center_x, center_y);
    // cv::Mat K = (cv::Mat <double>(3, 3) << f, 0, c.x, 0, f, c.y, 0, 0, 1);
    cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, c.x, 0, f, c.y, 0, 0, 1);
    std::vector<double> dist_coeff = { -0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 0.0001149909868437517, -0.01803978785585194 };

    // int f_select = 2;
    // const char* input_image = "/home/donghoon/KITTI_Odom_dataset/data_odometry_gray/dataset/sequences/00/image_0/%06d.png";
    // double f = 718.856;
    // cv::Point2d c(607.1928, 185.2157);
    int min_inlier_num = 100;
    
    // receive data
    cv::VideoCapture video;
    if (!video.open(final_input_image_path)) return -1;

    cv::Mat previous_image;
    video >> previous_image;
    // if (previous_image.empty()) break;
    if (previous_image.channels() > 1) cv::cvtColor(previous_image, previous_image, cv::COLOR_RGB2GRAY);

    int mode = GLUT_RGB | GLUT_SINGLE;
    glutInit(&argc, argv);
    glutInitDisplayMode(mode);    // Set drawing surface property
    glutInitWindowPosition(200, 200);    // Set window Position at Screen
    glutInitWindowSize(1000,1000);    // Set window size. Set printed working area size. Bigger than this size
    glutCreateWindow("Trajectory");    // Generate window. argument is window's name

    glClearColor(1.0, 1.0, 1.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT); 

    // cv::Mat camera_pose = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);
    // ifstream input_gt(argv[1]);
    ifstream GT_data(final_GT_data_path);
    cv::Mat R, t;
    int times = 2;
    std::vector<cv::Point3d> obj_points;
    cv::Mat rvec, tvec;
    while(true)
    {

        // if (times == 2) 
        // if(f_select = 1)
        // {
            // // Feature Detect / Extract Descriptor 
            // cv::Mat image;
            // video >> image;
            // if (image.empty()) break;

            // // if (previous_image.empty()) previous_image = image;
            

            
            // cv::Ptr<Feature2D> fdetector = cv::ORB::create();
            // cv::Mat current_image;
            // current_image = previous_image;
            // previous_image = image;

            // std::vector<cv::KeyPoint> keypoint1, keypoint2;
            // cv::Mat descriptor1, descriptor2;
        
            // fdetector->detectAndCompute(current_image, cv::Mat(), keypoint1, descriptor1);
            // fdetector->detectAndCompute(previous_image, cv::Mat(), keypoint2, descriptor2);

            // // matching
            // cv::Ptr<DescriptorMatcher> fmatcher = cv::BFMatcher::create(NORM_HAMMING);

            // std::vector<cv::DMatch> match;
            // fmatcher->match(descriptor1, descriptor2, match);

            // // save the matched point
            // vector<Point2f> pts1, pts2;
            // for ( size_t i = 0; i < match.size(); i++)
            // {
            //     pts1.push_back(keypoint1[match[i].queryIdx].pt);
            //     pts2.push_back(keypoint2[match[i].trainIdx].pt);
            // }

            

        
        // }
            cout << times <<endl;
        // if(f_select = 2)
        // {
            cv::Mat image, current_image;
            video >> image; 
            if (image.empty()) break;
            if (image.channels() > 1) cv::cvtColor(image, current_image, cv::COLOR_RGB2GRAY);
            else current_image = image.clone();

            
            
            
            // Extract optical flow
            vector<Point2f> pts1, pts2;
            cv::goodFeaturesToTrack(previous_image, pts1, 2000, 0.01, 10);
            std::vector<uchar> status;
            cv::Mat err;
            cv::calcOpticalFlowPyrLK(previous_image, current_image, pts1, pts2, status, err);
            
            previous_image = current_image;

        // }
            



        // Calculate Essential Matrix
        // for (int i = 3; i > times; i--)
        if( times == 2)
        {
        cv::Mat E, inlier_mask;
        E = cv::findEssentialMat(pts1, pts2, f, c, cv::RANSAC, 0.999, 1, inlier_mask);

        // Caculate R,t matrix
        
        int inlier_num = cv::recoverPose(E, pts1, pts2, R, t, f, c, inlier_mask);
        
        }
        // cout << pts2.size() << endl;
        // int threshold_pose_update = 100 * inlier_num / pts2.size(); 
        // cout << threshold_pose_update << endl;
        // if (threshold_pose_update > 20)
        // // if (inlier_num > min_inlier_num)
        // {dd
            // cv::Mat T = cv::Mat::eye(4, 4, R.type());
            // T(cv::Rect(0, 0, 3, 3)) = R * 1.0;
            // T.col(3).rowRange(0, 3) = t * 1.0;
            // camera_pose = camera_pose * T.inv();
            cout << obj_points.size() << endl;
            cout << pts2.size() << endl;
            rvec = R;
            tvec = t;
            
            if ( times > 2)
            {
            cv::solvePnP(obj_points, pts2, K, dist_coeff, rvec, tvec);
            cv::Rodrigues(rvec, rvec);
            }
            cout << times <<endl;
            
            // cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);
            cv::Mat Rt, X;
            // cv::Mat Iden = Mat::eye(4, 4, CV_64F);
            cv::hconcat(rvec, tvec, Rt);
            // cv::vconcat(Rt, Iden.row(3), Rt);
            // cv::vconcat(P0, Iden.row(3), P0);
            cv::Mat P1 = K * Rt;
            // P0.resize(3);
            // P1.resize(3);
            cv::triangulatePoints(P0, P1, pts1, pts2, X);
            
            X.row(0) = X.row(0) / X.row(3);
            X.row(1) = X.row(1) / X.row(3);
            X.row(2) = X.row(2) / X.row(3);
            X.row(3) = 1; 
        // }
            P0 = P1.clone();
            X.resize(3);
            // cout << times << endl;
            cout << X.col(0) << endl;
            
            for (int c = 0; c < pts1.size() ; c++ ) obj_points.push_back(cv::Point3f(X.col(c)));
            cv::Mat p = -rvec.t() * tvec;
            


            
            
        // show image and matching point
        // cv::String info = cv::format("Inliers: %d (%d%%),  XYZ: [%.3f, %.3f, %.3f]", inlier_num, 100 * inlier_num / point.size(), camera_pose.at<double>(0, 3), camera_pose.at<double>(1, 3), camera_pose.at<double>(2, 3));
        cv::Mat show_image;
        show_image = previous_image;
        if (show_image.channels() < 3) cv::cvtColor(show_image, show_image, cv::COLOR_GRAY2RGB);
        for(int a = 0; a < pts1.size(); a++) circle(show_image, pts1[a], 7, Scalar(0,255,0), 1);

        // cv::String info = cv::format("Inliers: %d ,  XYZ: [%.3f, %.3f, %.3f]", inlier_num, camera_pose.at<double>(0, 3), camera_pose.at<double>(1, 3), camera_pose.at<double>(2, 3));
        cv::String info = cv::format("Inliers:  ,  XYZ: [%.3f, %.3f, %.3f]", p.at<double>(0, 0), p.at<double>(1, 0), p.at<double>(2, 0));
        
        cv::putText(show_image, info, cv::Point(5, 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Vec3b(0, 0, 255));
        cv::imshow("image_pic", show_image);
        // cout << p << endl;
        // cv::waitKey();
        // OPENGL 
        GLfloat inputbuffer[12];
        for(int j=0;j<12;j++) GT_data >> inputbuffer[j];
        

        // show gt trajectory (red line)
        GLfloat x_gt(inputbuffer[3]), y_gt(inputbuffer[7]), z_gt(inputbuffer[11]);

        glColor3f(1.0,0.0,0.0);
        glPointSize(3.0);
        glBegin(GL_POINTS);
        glVertex3f(x_gt*0.001, z_gt*0.001, y_gt*-0.001);

        //show camera estimate pose trajectory (green line)
        GLfloat x_cam_pose(p.at<double>(0, 0)), y_cam_pose(p.at<double>(1, 0)), z_cam_pose(p.at<double>(2, 0));
        // GLfloat x_cam_pose(p.x), y_cam_pose(camera_pose.at<double>(1, 3)), z_cam_pose(camera_pose.at<double>(2, 3));

        glColor3f(0.0,1.0,0.0);
        glPointSize(3.0);
        glBegin(GL_POINTS);
        glVertex3f(x_cam_pose*0.001, z_cam_pose*0.001, y_cam_pose*-0.001);
        

        glEnd();
        glFlush();
        
        

        // double fps = video.set(CAP_PROP_FPS, 30);
        double delay = cvRound(1000/30);
        cv::waitKey(delay);
        // cv::waitKey();
        times++;
        
    }

    return 0;
}

        


        

