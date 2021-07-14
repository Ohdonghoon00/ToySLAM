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
#include "TOY_VO.h"

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
     
    for(int i= 0; i < 3; i++)
        for(int j = 0; j < 4; j++)
            intrinsic[i][j] = stod(buffer2[4*i+j+1]);
            
    double f = intrinsic[0][0];
    cv::Point2d c(intrinsic[0][2],intrinsic[1][2]);
    const cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, c.x, 0, f, c.y, 0, 0, 1);
    std::vector<double> dist_coeff = { -0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 0.0001149909868437517, -0.01803978785585194 };
            
    cout << K << endl;
        
    int mode = GLUT_RGB | GLUT_SINGLE;
    glutInit(&argc, argv);
    glutInitDisplayMode(mode);    // Set drawing surface property
    glutInitWindowPosition(200, 200);    // Set window Position at Screen
    glutInitWindowSize(1000,1000);    // Set window size. Set printed working area size. Bigger than this size
    glutCreateWindow("Trajectory");    // Generate window. argument is window's name

    glClearColor(1.0, 1.0, 1.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);    
        
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
    
    ifstream GT_data(final_GT_data_path);
    
    cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat cam_pose_ = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat _cam_pose_ = cv::Mat::zeros(4, 1, CV_64F);
    cv::Mat Iden = Mat::eye(4, 4, CV_64F);
    std::vector<Point3f> map_point;
    cv::Mat R, t, Rt;
    cv::Mat World_R = cv::Mat::eye(3, 3, CV_64F), World_t = cv::Mat::zeros(3, 1, CV_64F), World_Rt;
    std::vector<uchar> status;
    cv::Mat err;
    std::vector<cv::Point2f> projectionpoints;
    std::vector<cv::Mat> normal, R_, t_;
    cv::Mat X, map_change;
    // Storage storage_frame;
    std::vector<Frame> abs;

    
    while(true)
    {
        // count image num
        cout << times << endl;
        
        // push new image
        Frame current_image(times);
        cv::Mat image;
        video >> image;
        if (image.empty()) break;
        if (image.channels() > 1) cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
        
        if (times > 2)
        {
            cout << "optical flow" << endl;
            // cv::projectPoints(map_point, World_R, World_t, K, dist_coeff, projectionpoints);
            std::vector<Point2f> corresponding_points;
            cv::calcOpticalFlowPyrLK(previous_image.frame, image, previous_image.image_pts, corresponding_points, status, err);
            cout << previous_image.image_pts.size() << "    " << corresponding_points.size() << endl;
            // remove err point
            int indexCorrection_ = 0;
            for( int i=0; i<status.size(); i++)
            {
                Point2f pt = corresponding_points.at(i- indexCorrection_);
                if ((status[i] == 0)||(pt.x<0)||(pt.y<0))	
                    {
                        if((pt.x<0)||(pt.y<0)) status[i] = 0;             
                        previous_image.image_pts.erase ( previous_image.image_pts.begin() + i - indexCorrection_);
                        corresponding_points.erase (corresponding_points.begin() + i - indexCorrection_);
                        map_point.erase (map_point.begin() + i - indexCorrection_);
                        indexCorrection_++;
                    }
            }    
            cout << "pnpsolve" << endl;
            cout << previous_image.image_pts.size() << "    " << corresponding_points.size() << "   " << map_point.size() << endl;
            cv::Rodrigues(World_R,World_R);
            cv::solvePnPRansac(map_point, corresponding_points, K, dist_coeff, World_R, World_t, true);
            cout << World_R << "    " << World_t << endl;
            cv::Rodrigues(World_R, World_R);
            cout << " caculate pose" << endl;
            _cam_pose_ = -World_R.inv() * World_t;
            cout << _cam_pose_ << endl;
            cout << "push cam pose at database" << endl;
            current_image.cam_pose = {{_cam_pose_.at<double>(0, 0),_cam_pose_.at<double>(0, 1),_cam_pose_.at<double>(0, 2)}};
                        
        }
        current_image.frame = image.clone();
       
        
        
        cout << " feature track " << endl;
        cv::goodFeaturesToTrack(previous_image.frame, previous_image.image_pts, 2000, 0.01, 10);
        cv::calcOpticalFlowPyrLK(previous_image.frame, current_image.frame, previous_image.image_pts, current_image.image_pts, status, err);
        for(int i = 0; i < current_image.image_pts.size(); i++) current_image.frame_pts_id.push_back(i);
       cout << previous_image.image_pts.size() << "    " << current_image.image_pts.size() << endl;
        // cout << previous_image.image_pts << endl;
    
        
        // remove err point
        int indexCorrection = 0;
        for( int i=0; i<status.size(); i++)
        {
            Point2f pt = current_image.image_pts.at(i- indexCorrection);
            if ((status[i] == 0)||(pt.x<0)||(pt.y<0))	
            {
                if((pt.x<0)||(pt.y<0)) status[i] = 0;
                previous_image.image_pts.erase ( previous_image.image_pts.begin() + i - indexCorrection);
                current_image.image_pts.erase (current_image.image_pts.begin() + i - indexCorrection);
                indexCorrection++;
            }
        }        

        if ( times < 3)
        {
            cout << "essential matrix" << endl;
            // Caculate relatevi R,t using Essential Matrix 
            cv::Mat E, inlier_mask;
            E = cv::findEssentialMat(previous_image.image_pts, current_image.image_pts, f, c, cv::RANSAC, 0.99, 1, inlier_mask);
            inlier_num = cv::recoverPose(E, previous_image.image_pts, current_image.image_pts, R, t, f, c, inlier_mask);
            World_R = R;
            World_t = t;
            _cam_pose_ = _cam_pose_ * Rt.inv();
            // cv::hconcat(R, t, Rt);
            // cv::vconcat(Rt, Iden.row(3), Rt);
            // if ( 100 * inlier_num/ previous_image.image_pts.size() > 35) cam_pose_ = cam_pose_ * Rt.inv();
            

            // caculate relative R,t using homography
            // else
            // {
            //     cv::Mat H = cv::findHomography(previous_image.image_pts, current_image.image_pts, inlier_mask, cv::RANSAC);
            //     int solution = cv::decomposeHomographyMat(H, K, R_, t_, normal);
            //     R = R_[0];
            //     t = t_[0];
            //     cv::hconcat(R, t, Rt);
            //     cv::vconcat(Rt, Iden.row(3), Rt);
            //     cam_pose_ = cam_pose_ * Rt.inv();
            // }
        }

        
        // cv::sfm::projectionFromKRt(K, R, t, P1);
        // Rt.resize(3);
        // P1 = Rt;
        cout << "projection matrix " << endl;
        cout << P0 << endl;
        cout << P1 << endl;
        cv::sfm::projectionFromKRt(K, World_R, World_t, P1);
        cout << "triangulation" << endl;
        cv::triangulatePoints(P0, P1, previous_image.image_pts, current_image.image_pts, X);
        cout << " P0 " << "     " << P0 << endl;
        cout << " P1 " << "     " << P1 << endl;
        for (int i = 0 ; i < X.cols; i++)
        {
            X.col(i).row(0) = X.col(i).row(0) / X.col(i).row(3);
            X.col(i).row(1) = X.col(i).row(1) / X.col(i).row(3);
            X.col(i).row(2) = X.col(i).row(2) / X.col(i).row(3);
            X.col(i).row(3) = 1;
        }
        // cout << " X " << X << endl;
        X.convertTo(X, CV_64F);
        map_change = K.inv() * P0;
        cout << K.inv() * K << endl;
        cout << map_change << endl;
        cv::vconcat(map_change, Iden.row(3), map_change);
        cout << map_change << endl;
        X = map_change.inv() * X;
        // X.resize(3);
        // cout << " X " << X << endl;
        P0 = P1.clone();
        
        cout << " map_point " << endl;
        // for(int i = 0 ; i < 1000 ; i++) cout << X.at<double>(0,i) << endl;
        map_point.clear();
        for (int c = 0; c < previous_image.image_pts.size() ; c++ ) map_point.push_back(Point3f(X.at<double>(0, c), X.at<double>(1, c), X.at<double>(2, c)));    
        // cout << map_point << endl;

        cout << "reprojection 3D to 2D " << endl;
        // cv::projectPoints(map_point, World_R, World_t, K, cv::noArray(), projectionpoints);
        

        

        

        
        
        
        cout << "show image" << endl;
        // show image and matching point
        cv::Mat show_image = current_image.frame;
        if (show_image.channels() < 3) cv::cvtColor(show_image, show_image, cv::COLOR_GRAY2RGB);
        for(int a = 0; a < previous_image.image_pts.size(); a++) circle(show_image, previous_image.image_pts[a], 3, Scalar(0,255,0), 1);
        // for(int a = 0; a < previous_image.image_pts.size(); a++) circle(show_image, projectionpoints[a], 3, Scalar(0,0,255), 1);
        
        cv::String info = cv::format("Inliers: %d (%d%%) XYZ: [%.3lf, %.3lf, %.3lf]",inlier_num, 100 * inlier_num/ previous_image.image_pts.size(), _cam_pose_.at<double>(0, 0),_cam_pose_.at<double>(0, 1),_cam_pose_.at<double>(0, 2));
        cv::putText(show_image, info, cv::Point(5, 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Vec3b(255, 0, 0));
        cv::imshow("image_pic", show_image);
        
        // OPENGL 
        GLfloat inputbuffer[12];
        for(int j=0;j<12;j++) GT_data >> inputbuffer[j];
        

        // show gt trajectory (red line)
        GLfloat x_gt(inputbuffer[3]), y_gt(inputbuffer[7]), z_gt(inputbuffer[11]);

        glColor3f(1.0,0.0,0.0);
        glPointSize(3.0);
        glBegin(GL_POINTS);
        glVertex3f(x_gt*0.001, z_gt*0.001, y_gt*-0.001);
        glEnd();
        
        //show camera estimate pose trajectory (green line)
        GLfloat x_cam_pose(_cam_pose_.at<double>(0, 0)), y_cam_pose(_cam_pose_.at<double>(0, 1)), z_cam_pose(_cam_pose_.at<double>(0, 2));

        glColor3f(0.0,1.0,0.0);
        glPointSize(3.0);
        glBegin(GL_POINTS);
        glVertex3f(x_cam_pose*0.001, z_cam_pose*0.001, y_cam_pose*-0.001);
        glEnd();
        
        // show 3d map point
        if ( times % 20 == 0)
        {
            for( int i = 0 ; i < map_point.size(); i++)
            {
                GLfloat X_map(map_point[i].x), Y_map(map_point[i].y), Z_map(map_point[i].z);
                glColor3f(0.0,0.0,1.0);
                glPointSize(0.5);
                glBegin(GL_POINTS);
                glVertex3f(X_map*0.001, Z_map*0.001, Y_map*-0.001);
                glEnd();
            }        
        
        }
        
        glFlush();

        // storage_frame 
        
        
        abs.push_back(current_image);
        for (int i =0; i < times + 1 ; i ++) cout << abs[0].Frame_id << endl;
        Frame previous_image(current_image);

        ++times;
        // double fps = video.set(CAP_PROP_FPS, 30);
        double delay = cvRound(1000/30);
        // cv::waitKey(delay);
        cv::waitKey();
    }

    return 0;
}



               

        
        
        

            
            
                            

                            



        

        
            
         
        
        