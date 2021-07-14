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
    string input_image_path ("../../../../dataset/sequences/");
    string cam_calib_path ("../../../../dataset/sequences/");
    string GT_data_path ("../../../../dataset/poses/");
    
    string final_input_image_path = input_image_path + argv[1] + "/image_0/%06d.png";
    string final_came_calib_path = cam_calib_path + argv[1] + "/calib.txt";
    string final_GT_data_path = GT_data_path + argv[1] + ".txt";

    ifstream calib_data(final_came_calib_path);
    ifstream GT_data(final_GT_data_path);
    


    cv::VideoCapture video;
    if (!video.open(final_input_image_path)) return -1;

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
    

    glutInit(&argc, argv);
    initialize_window();

    
        
    int min_inlier_num = 100;
    int inlier_num;
    // receive data

    int times = 1;
    Frame previous_image(0);
    // for(int i = 0; i < 3; i++) previous_image.cam_pose[i] = 0;
    video >> previous_image.frame;
    const int image_x_size = previous_image.frame.cols;
    const int image_y_size = previous_image.frame.rows;
    // if (previous_image.empty()) break;
    if (previous_image.frame.channels() > 1) cv::cvtColor(previous_image.frame, previous_image.frame, cv::COLOR_RGB2GRAY);
    
    
    cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat P1;
    cv::Mat cam_pose_initialize = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat cam_pose = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat Iden = Mat::eye(4, 4, CV_64F);
    std::vector<Point3d> map_point;
    cv::Mat R, t, Rt;
    
    cv::Mat World_R, World_t, World_Rt;
    std::vector<uchar> status;
    cv::Mat err;
    std::vector<cv::Point2f> projectionpoints;
    std::vector<cv::Mat> normal, R_, t_;
    cv::Mat X, map_change;
    // Storage Storage_frame;
    std::vector<Frame> frame_storage;

    
    while(true)
    {
        // count image num
        std::cout << times << " Frame !!!!!!!!1" << endl;
        
        // push new image
        Frame current_image(times);
        if(times > 1) 
        {
            cout << previous_image.pts_id.size() << endl;
            cout << current_image.pts_id.size();
        }
        // for (int i =0; i < times - 1  ; i ++)
        // { 
        //     std::cout << " Frame_Id :       " << frame_storage[i].Frame_id << "   " << endl;
        //     std::cout << " Frame_point_id : " << frame_storage[i].pts_id.size() << "   " << endl;
        //     std::cout << " cam_pose : !!!!!!!!!!!" << i << endl << frame_storage[i].cam_pose_[0] << "   " << endl;
        // }
        cv::Mat image;
        video >> image;
        if (image.empty()) break;
        if (image.channels() > 1) cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
        current_image.frame = image.clone();
        if (times > 1)
        {
            std::cout << "optical flow" << endl;
            // cv::projectPoints(map_point, World_R, World_t, K, dist_coeff, projectionpoints);
            std::vector<Point2f> corresponding_points;
            cv::calcOpticalFlowPyrLK(previous_image.frame, current_image.frame, previous_image.pts, current_image.pts, status, err);
            std::cout << previous_image.pts.size() << "    " << current_image.pts.size() << endl;
            
            // remove err point
            int indexCorrection_ = 0;
            for( int i=0; i<status.size(); i++)
            {
                Point2f pt = current_image.pts.at(i- indexCorrection_);
                if ((status[i] == 0)||(pt.x < 0)||(pt.y < 0)||(pt.x > image_x_size)||(pt.y > image_y_size))	
                    {
                        if((pt.x < 0)||(pt.y < 0)||(pt.x > image_x_size)||(pt.y > image_y_size)) status[i] = 0;             
                        previous_image.pts.erase ( previous_image.pts.begin() + i - indexCorrection_);
                        current_image.pts.erase (current_image.pts.begin() + i - indexCorrection_);
                        map_point.erase (map_point.begin() + i - indexCorrection_);
                        indexCorrection_++;
                    }
            }    
            std::cout << "push id" << endl;
            std::cout << current_image.pts.size() << endl;
            for(int i = 0; i < current_image.pts.size(); i++) current_image.pts_id.push_back(previous_image.pts_id[i]);

            std::cout << "pnpsolve" << endl;
            std::cout << previous_image.pts.size() <<  "   " << map_point.size() << endl;
            cv::Rodrigues(World_R,World_R);
            cv::solvePnPRansac(map_point, current_image.pts, K, dist_coeff, World_R, World_t, true);
            std::cout << World_R << "    " << World_t << endl;
            
            cv::Rodrigues(World_R, World_R);
            World_Rt = R_t_to_homogenous(World_R, World_t);

            std::cout << " caculate pose" << endl;
            // cam_pose = -World_R.inv() * World_t;
            cam_pose = World_Rt.inv();
            cv::Mat cam_pose_ = cam_pose.clone();
            std::cout << cam_pose << endl;
            std::cout << "push cam pose at database" << endl;
            // current_image.cam_pose_ = {{cam_pose.at<double>(0, 3),cam_pose.at<double>(1, 3),cam_pose.at<double>(2, 3)}};
            current_image.cam_pose_.push_back(cam_pose_);
            for(int i = 0; i < times-1; i++) std::cout << " cam_pose : !!!!!!!!!!!    " << frame_storage[i].Frame_id << endl << frame_storage[i].cam_pose_ << "   " << endl;
            World_Rt.resize(3);
            P1 = K * World_Rt;
            frame_storage.push_back(current_image);
                        
        }
        
       
        
        
    //     previous_image.pts.clear();
    //     cout << previous_image.pts.size() << endl;
    //     std::cout << " feature track " << endl;
    //     cv::goodFeaturesToTrack(previous_image.frame, previous_image.pts, 2000, 0.01, 10);
    //     cout << previous_image.pts.size() << endl;
    //     cv::calcOpticalFlowPyrLK(previous_image.frame, current_image.frame, previous_image.pts, current_image.pts, status, err);
        
    //    std::cout << previous_image.pts.size() << "    " << current_image.pts.size() << endl;
    //     // std::cout << previous_image.pts << endl;
    
        
    //     // remove err point
    //     int indexCorrection = 0;
    //     for( int i=0; i<status.size(); i++)
    //     {
    //         Point2f pt = current_image.pts.at(i- indexCorrection);
    //         if ((status[i] == 0)||(pt.x<0)||(pt.y<0)||(pt.x>image_x_size)||(pt.y>image_y_size))	
    //         {
    //             if((pt.x<0)||(pt.y<0)||(pt.x>image_x_size)||(pt.y>image_y_size)) status[i] = 0;
    //             previous_image.pts.erase ( previous_image.pts.begin() + i - indexCorrection);
    //             current_image.pts.erase (current_image.pts.begin() + i - indexCorrection);
    //             indexCorrection++;
    //         }
    //     }     
           
        std::cout << previous_image.pts.size() << "    " << current_image.pts.size() << endl;
        if ( times < 2)
        {
            previous_image.pts.clear();
            cout << previous_image.pts.size() << endl;
            std::cout << " feature track " << endl;
            cv::goodFeaturesToTrack(previous_image.frame, previous_image.pts, 2000, 0.01, 10);
            cout << previous_image.pts.size() << endl;
            cv::calcOpticalFlowPyrLK(previous_image.frame, current_image.frame, previous_image.pts, current_image.pts, status, err);
        
            std::cout << previous_image.pts.size() << "    " << current_image.pts.size() << endl;
            // std::cout << previous_image.pts << endl;
    
        
            // remove err point
            int indexCorrection = 0;
            for( int i=0; i<status.size(); i++)
            {
                Point2f pt = current_image.pts.at(i- indexCorrection);
                if ((status[i] == 0)||(pt.x<0)||(pt.y<0)||(pt.x>image_x_size)||(pt.y>image_y_size))	
                {
                    if((pt.x<0)||(pt.y<0)||(pt.x>image_x_size)||(pt.y>image_y_size)) status[i] = 0;
                    previous_image.pts.erase ( previous_image.pts.begin() + i - indexCorrection);
                    current_image.pts.erase (current_image.pts.begin() + i - indexCorrection);
                    indexCorrection++;
                }
            }     
            
            for(int i = 0; i < current_image.pts.size(); i++) current_image.pts_id.push_back(i);
            std::cout << "essential matrix" << endl;
            // Caculate relatevi R,t using Essential Matrix 
            cv::Mat E, inlier_mask;
            E = cv::findEssentialMat(previous_image.pts, current_image.pts, f, c, cv::RANSAC, 0.99, 1, inlier_mask);
            inlier_num = cv::recoverPose(E, previous_image.pts, current_image.pts, R, t, f, c, inlier_mask);
            World_R = R.clone();
            World_t = t.clone();
            std::cout << World_R << "    " << World_t << endl;
            // cv::hconcat(R, t, Rt);
            // cv::vconcat(Rt, Iden.row(3), Rt);

            Rt = R_t_to_homogenous(R, t);

            cv::Mat cam_pose_initialize_;
            cam_pose_initialize_ = cam_pose_initialize.clone();
            cam_pose_initialize_ = cam_pose_initialize_ * Rt.inv();
            cam_pose_initialize = cam_pose_initialize_.clone();

            // current_image.cam_pose_ = {{cam_pose_initialize.at<double>(0, 3),cam_pose_initialize.at<double>(1, 3),cam_pose_initialize.at<double>(2, 3)}};
            cout << "@@@@@@@@@@" << current_image.cam_pose_ << endl;
            current_image.cam_pose_.push_back(cam_pose_initialize);
            for(int i = 0; i < times-1; i++) std::cout << " cam_pose : !!!!!!!!!!!    " << frame_storage[i].Frame_id << endl << frame_storage[i].cam_pose_ << "   " << endl;
            // caculate relative R,t using homography
            // else
            // {
            //     cv::Mat H = cv::findHomography(previous_image.pts, current_image.pts, inlier_mask, cv::RANSAC);
            //     int solution = cv::decomposeHomographyMat(H, K, R_, t_, normal);
            //     R = R_[0];
            //     t = t_[0];
            //     cv::hconcat(R, t, Rt);
            //     cv::vconcat(Rt, Iden.row(3), Rt);
            //     cam_pose_ = cam_pose_ * Rt.inv();
            // }
            cv::Mat Rt_ = cam_pose_initialize.inv();
            cout << Rt_ << endl;
            Rt_.resize(3);
            std::cout << "projection matrix " << endl;
            P1 = K * Rt_;
            cam_pose = cam_pose_initialize.clone();
            frame_storage.push_back(current_image);
        }

        
        // cv::sfm::projectionFromKRt(K, World_R, World_t, P1);
        // cv::sfm::projectionFromKRt(K, R, t, P1);
        // Rt.resize(3);
        // P1 = Rt;
        
        std::cout << "triangulation" << endl;
        cv::triangulatePoints(P0, P1, previous_image.pts, current_image.pts, X);
        std::cout << " P0 " << "     " << P0 << endl;
        std::cout << " P1 " << "     " << P1 << endl;
        for (int i = 0 ; i < X.cols; i++)
        {
            X.col(i).row(0) = X.col(i).row(0) / X.col(i).row(3);
            X.col(i).row(1) = X.col(i).row(1) / X.col(i).row(3);
            X.col(i).row(2) = X.col(i).row(2) / X.col(i).row(3);
            X.col(i).row(3) = 1;
        }
        X.convertTo(X, CV_64F);
        // std::cout << " X " << X << endl;
        // map_change = K.inv() * P0;
        // std::cout << K.inv() * K << endl;
        // std::cout << map_change << endl;
        // cv::vconcat(map_change, Iden.row(3), map_change);
        // std::cout << map_change << endl;
        // X = map_change.inv() * X;
        // X.resize(3);
        // std::cout << " X " << X << endl;
        P0 = P1.clone();
        
        std::cout << " map_point " << endl;
        // for(int i = 0 ; i < 1000 ; i++) std::cout << X.at<double>(0,i) << endl;
        map_point.clear();
        for (int c = 0; c < previous_image.pts.size() ; c++ ) map_point.push_back(Point3d(X.at<double>(0, c), X.at<double>(1, c), X.at<double>(2, c)));    
        // std::cout << map_point << endl;

        std::cout << "reprojection 3D to 2D " << endl;
        // cv::projectPoints(map_point, World_R, World_t, K, cv::noArray(), projectionpoints);
        

        

        

        
        
        
        std::cout << "show image" << endl;
        // show image and matching point
        cv::Mat show_image = current_image.frame;
        if (show_image.channels() < 3) cv::cvtColor(show_image, show_image, cv::COLOR_GRAY2RGB);
        for(int a = 0; a < previous_image.pts.size(); a++) circle(show_image, previous_image.pts[a], 3, Scalar(0,255,0), 1);
        // for(int a = 0; a < previous_image.pts.size(); a++) circle(show_image, projectionpoints[a], 3, Scalar(0,0,255), 1);
        
        cv::String info = cv::format("Inliers: %d (%d%%) XYZ: [%.3lf, %.3lf, %.3lf]",inlier_num, 100 * inlier_num/ previous_image.pts.size(), current_image.cam_pose_.at<double>(0, 3),current_image.cam_pose_.at<double>(1, 3),current_image.cam_pose_.at<double>(2, 3));
        cv::putText(show_image, info, cv::Point(5, 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Vec3b(255, 0, 0));
        cv::imshow("image_pic", show_image);
        
        
        

        // show gt trajectory (red line)
        GLfloat inputbuffer[12];
        for(int j=0;j<12;j++) GT_data >> inputbuffer[j];
        GLfloat x_gt(inputbuffer[3]), y_gt(inputbuffer[7]), z_gt(inputbuffer[11]);
        show_trajectory(x_gt, y_gt, z_gt, 1.0, 0.0, 0.0, 3.0);

        
        //show camera estimate pose trajectory (green line)
        GLfloat x_cam_pose(current_image.cam_pose_.at<double>(0, 3)), y_cam_pose(current_image.cam_pose_.at<double>(1, 3)), z_cam_pose(current_image.cam_pose_.at<double>(2, 3));
        show_trajectory(x_cam_pose, y_cam_pose, z_cam_pose, 0.0, 1.0, 0.0, 3.0);

        
        // show 3d map point
        // if ( times % 20 == 0)
        // {
            // for( int i = 0 ; i < map_point.size(); i++)
            // {
            //     GLfloat X_map(map_point[i].x), Y_map(map_point[i].y), Z_map(map_point[i].z);
            //     show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 1.0, 0.5);
            // }        
        // }
        
        glFlush();
        

        // storage_frame 

        
        for (int i =0; i < times ; i ++)
        { 
            std::cout << " Frame_Id :       " << frame_storage[i].Frame_id << "   " << endl;
            std::cout << " Frame_point_id : " << frame_storage[i].pts_id.size() << "   " << endl;
            std::cout << " cam_pose : " << i + 1 << endl << frame_storage[i].cam_pose_ << "   " << endl;
        }
        
        previous_image = current_image;

        ++times;
        // double fps = video.set(CAP_PROP_FPS, 30);
        double delay = cvRound(1000/30);
        // cv::waitKey(delay);
        cv::waitKey();
    }

    return 0;
}



               

        
        
        

            
            
                            

                            



        

        
            
         
        
        