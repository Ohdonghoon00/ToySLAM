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
    int initialize_frame_num = 8;
    // receive data

    int times = 1;
    int keyframe_num = 1;
    
    
    cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat cam_pose_initialize = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat cam_pose = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat Iden = Mat::eye(4, 4, CV_64F);
    std::vector<Point3d> map_point;
    cv::Mat R, t, Rt;
    cv::Mat inliers;
    
    cv::Mat World_R, World_t, World_Rt;
    // std::vector<uchar> status;
    // cv::Mat err;
    std::vector<cv::Point2f> projectionpoints;
    cv::Mat R_, t_;
    cv::Mat X, map_change;
    
    // Storage Storage_frame;
    std::vector<Frame> frame_storage;
    std::vector<Keyframe> keyframe_storage;
    std::vector<Map> map_storage;

    Frame previous_image(0);
    // for(int i = 0; i < 3; i++) previous_image.cam_pose[i] = 0;
    video >> previous_image.frame;
    const int image_x_size = previous_image.frame.cols;
    const int image_y_size = previous_image.frame.rows;
    // if (previous_image.empty()) break;
    if (previous_image.frame.channels() > 1) cv::cvtColor(previous_image.frame, previous_image.frame, cv::COLOR_RGB2GRAY);
    
    
    while(true)
    {
        // count image num
        std::cout << times << " Frame !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        
        // push new image
        Frame current_image(times);
        cv::Mat image;
        video >> image;
        if (image.empty()) cv::waitKey();
        if (image.channels() > 1) cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
        current_image.frame = image.clone();

        // Caculate R, t using Essential Matrix or Homography 
        if ( times < initialize_frame_num + 1)
        {
            previous_image.pts.clear();
            
            // Find feature in previous_image 
            std::cout << " feature track " << endl;
            cv::goodFeaturesToTrack(previous_image.frame, previous_image.pts, 2000, 0.01, 10);

            // Matching prev_image and current_image using optical flow
            std::cout << " optical flow " << endl;
std::cout << previous_image.pts.size() << endl;            
            track_opticalflow_and_remove_err(previous_image.frame, current_image.frame, previous_image.pts, current_image.pts);
std::cout << previous_image.pts.size() << endl;             
            // Caculate relative R, t using Essential Matrix
            std::cout << "essential matrix" << endl;
            cv::Mat E, inlier_mask;
            E = cv::findEssentialMat(previous_image.pts, current_image.pts, f, c, cv::RANSAC, 0.99, 1, inlier_mask);
            inlier_num = cv::recoverPose(E, previous_image.pts, current_image.pts, R, t, f, c, inlier_mask);
            
            // Caculate relative R, t using homography
            // std:: cout << "homography" << endl;
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

            // Update cam_pose using R, t
            Rt = R_t_to_homogenous(R, t);
            cam_pose_initialize = cam_pose_initialize * Rt.inv();
            
            // Storage cam_pose
            cv::Mat cam_pose_initialize_;
            cam_pose_initialize_ = cam_pose_initialize.clone();
            current_image.cam_pose_.push_back(cam_pose_initialize_);
            
            // Storage pts_id
            for(int i = 0; i < previous_image.pts.size(); i++) previous_image.pts_id.push_back(i);
            for(int i = 0; i < current_image.pts.size(); i++) current_image.pts_id.push_back(i);

            // Caculate projection matrix
            P0 = P1.clone();
            cv::Mat Rt_ = cam_pose_initialize.inv();
            Rt_.resize(3);
            P1 = K * Rt_;

            // Triangulation keyframe_storage (id = 0) and current_image
            std::cout << " triangulation " << endl;
            cv::triangulatePoints(P0, P1, previous_image.pts, current_image.pts, X); 

            // map_point
            std::cout << " storage map point " << endl;
            world_xyz_point_to_homogenous(X);
            X.convertTo(X, CV_64F);
            map_point.clear();
            for (int c = 0; c < previous_image.pts.size() ; c++ ) map_point.push_back(Point3d(X.at<double>(0, c), X.at<double>(1, c), X.at<double>(2, c)));
            
            // Storage map point
            // if(times == 5)
            // {
            //     Map initialize_map(keyframe_storage[0], map_point);
            //     map_storage.push_back(initialize_map);
            // }

            // Reprojection 3D to 2D current image
            std::cout << "reprojection 3D to 2D initialize stage " << endl;
            cv::sfm::KRtFromProjection(P1, K, R_, t_);
            cv::Mat project_mat = Mat(map_point).clone();
            project_mat.convertTo(project_mat, CV_32F);
            cv::projectPoints(project_mat, R_, t_, K, cv::noArray(), projectionpoints);                
                
            // Storage current image 
            Frame copy_current_image = current_image;
            frame_storage.push_back(copy_current_image);

        }
            
            
        // SolvePnP + Triangulation after times > 5 
        if (times > initialize_frame_num)
        {
            // Matching prev_image and current_image using optical flow
            std::cout << "optical flow" << endl;
            float before_track_feature_num = previous_image.pts.size();
            track_opticalflow_and_remove_err_for_SolvePnP(previous_image.frame, current_image.frame, previous_image.pts, current_image.pts, map_point);
std::cout << "  track inlier ratio : " << 100 * previous_image.pts.size() / before_track_feature_num << endl;
            // Storage Id
            for(int i = 0; i < current_image.pts.size(); i++) current_image.pts_id.push_back(previous_image.pts_id[i]);
std::cout << "SolvePnP map_point num : " << "   " << map_point.size() << endl;
            // Calculate World R, t using SolvePnP
            std::cout << "SolvePnP" << endl;
            cv::solvePnPRansac(map_point, current_image.pts, K, cv::noArray(), World_R, World_t, false, 100, 3.0F, 0.99, inliers );
std::cout << inliers << endl;
std::cout << " Inlier ratio : " << 100 * inliers.rows / map_point.size() << " %" << endl;
            // Calculate camera_pose
            std::cout << "caculate pose" << endl;
            cv::Rodrigues(World_R, World_R);
            World_Rt = R_t_to_homogenous(World_R, World_t);
            // cam_pose = -World_R.inv() * World_t;
            cam_pose = World_Rt.inv();

            // Storage camera pose
            std::cout << "  storage cam pose   " << endl;
            cv::Mat cam_pose_ = cam_pose.clone();
            current_image.cam_pose_.push_back(cam_pose_);

            
            // Storage Frame 
            frame_storage.push_back(current_image);


            // cv::Mat difference_for_R, difference_for_t, difference_for_Rt;
            // cv::sfm::KRtFromProjection(P1, K, difference_for_R, difference_for_t);
            // cv::hconcat(difference_for_R, difference_for_t, difference_for_Rt);
            // World_Rt.resize(3);
std::cout << "previous cam pose : " << endl << previous_image.cam_pose_ << endl;
            cv::Mat pose_difference = cv::abs(cam_pose_) - cv::abs(previous_image.cam_pose_);
std::cout << pose_difference << endl;
            double translation_difference = std::abs(pose_difference.at<double>(0, 3)) + std::abs(pose_difference.at<double>(1, 3)) + std::abs(pose_difference.at<double>(2, 3));
std::cout << "translation difference : " << translation_difference << endl;

            // Find new feature in previous image 
            std::cout << "  find feature    " << endl;
            while (true)
            {
                // Determinate doing triangulation or not 
                // if ( 100 * previous_image.pts.size() / before_track_feature_num > 99) break;
                if (translation_difference < 0.06 and 100 * previous_image.pts.size() / before_track_feature_num > 95 ) break;
                // if ( translation_difference < 0.087 ) break;
                
                previous_image.pts.clear();
                cv::goodFeaturesToTrack(previous_image.frame, previous_image.pts, 2000, 0.01, 10);

                // Matching previous_image and current_image using optical flow
                std::cout << "  optical flow    " << endl;
std::cout << previous_image.pts.size() << "    " << current_image.pts.size() << endl;
                track_opticalflow_and_remove_err(previous_image.frame, current_image.frame, previous_image.pts, current_image.pts);
std::cout << previous_image.pts.size() << "    " << current_image.pts.size() << endl;
                // Caculate projection matrix for triangulation
                std::cout << "  projection matrix    " << endl;
                P0 = P1.clone();
                World_Rt.resize(3);
                P1 = K * World_Rt;
            
            

            
            

            
            
            
            
                // Triangulation previous_image and current_image
                std::cout << "  triangulation    " << endl;
                cv::triangulatePoints(P0, P1, previous_image.pts, current_image.pts,  X);

                // Map_point
                std::cout << "  storage map    " << endl;
                world_xyz_point_to_homogenous(X);
                X.convertTo(X, CV_64F);
                map_point.clear();
                for (int c = 0; c < previous_image.pts.size() ; c++ ) map_point.push_back(Point3d(X.at<double>(0, c), X.at<double>(1, c), X.at<double>(2, c)));
                break;
            }
            // Storage map_point
            // Map initialize_map(keyframe_storage[(times / 5) - 1], map_point);
            // map_storage.push_back(initialize_map);

            // Storage current image 
            Frame copy_current_image = current_image;
            frame_storage.push_back(copy_current_image);
    
        }

std::cout << "current cam pose  : " << endl <<  current_image.cam_pose_ << endl;
        // Reprojection 3D to 2D current image
        std::cout << "reprojection 3D to 2D at SolvePnP stage " << endl;
        if(times > initialize_frame_num) 
        {
            cv::Mat project_mat = Mat(map_point).clone();
            project_mat.convertTo(project_mat, CV_32F);
            cv::projectPoints(project_mat, World_R, World_t, K, cv::noArray(), projectionpoints); 
        }
        
        
        std::cout << "show image" << endl;
        // Show image and Matching points and reprojection points
        cv::Mat show_image = current_image.frame;
        if (show_image.channels() < 3) cv::cvtColor(show_image, show_image, cv::COLOR_GRAY2RGB);
        for(int a = 0; a < current_image.pts.size(); a++) circle(show_image, current_image.pts[a], 3, Scalar(0,255,0), 1);
        for(int a = 0; a < current_image.pts.size(); a++) circle(show_image, projectionpoints[a], 3, Scalar(0,0,255), 1);
        
        // Put text to image about camera_pose
        cv::String info = cv::format("Inliers: %d (%d%%) XYZ: [%.3lf, %.3lf, %.3lf]",inlier_num, 100 * inlier_num/ previous_image.pts.size(), current_image.cam_pose_.at<double>(0, 3),current_image.cam_pose_.at<double>(1, 3),current_image.cam_pose_.at<double>(2, 3));
        cv::putText(show_image, info, cv::Point(5, 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Vec3b(255, 0, 0));
        cv::imshow("image", show_image);
        

        // Show gt trajectory (red line)
        GLfloat inputbuffer[12];
        for(int j=0;j<12;j++) GT_data >> inputbuffer[j];
        GLfloat x_gt(inputbuffer[3]), y_gt(inputbuffer[7]), z_gt(inputbuffer[11]);
        show_trajectory(x_gt, y_gt, z_gt, 1.0, 0.0, 0.0, 3.0);

        // Show camera estimate pose trajectory ( green line )
        GLfloat x_cam_pose(current_image.cam_pose_.at<double>(0, 3)), y_cam_pose(current_image.cam_pose_.at<double>(1, 3)), z_cam_pose(current_image.cam_pose_.at<double>(2, 3));
        show_trajectory(x_cam_pose, y_cam_pose, z_cam_pose, 0.0, 1.0, 0.0, 3.0);
        
        // Show 3d map point ( blue dot )
        // if ( times  % 5 == 0)
        // {   
        //     cout << "   mapamapmpampa    " <<  map_point.size() << endl;
        //     for( int i = 0 ; i < map_point.size(); i++)
        //     {
        //         GLfloat X_map(map_storage[keyframe_num - 1].world_xyz[i].x), Y_map(map_storage[keyframe_num - 1].world_xyz[i].y), Z_map(map_storage[keyframe_num - 1].world_xyz[i].z);
        //         show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 1.0, 0.5);
        //     }        
        // }

        // Show 3d map point ( blue dot )
        // if ( times % 25 == 0)
        // {   
        //     cout << "   mapamapmpampa    " <<  map_point.size() << endl;
        //     for( int i = 0 ; i < map_point.size(); i++)
        //     {
        //         GLfloat X_map(map_point[i].x), Y_map(map_point[i].y), Z_map(map_point[i].z);
        //         show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 1.0, 0.5);
        //     }        
        // }
        
        glFlush();

        
        // Keyframe_storage.
        if (times == 1)
        {
            cv::Mat cam_pose_initialize_ = cv::Mat::eye(4, 4, CV_64F);
            previous_image.cam_pose_.push_back(cam_pose_initialize_);

            Keyframe keyframe_(0);
            keyframe_.keyframe = previous_image;

            keyframe_storage.push_back(keyframe_);
            // for (int i = 0; i < keyframe_num; i++) 
            // {
            //     std::cout <<    " keyframe id   :  " << keyframe_storage[i].keyframe_id << "   " << endl;
            //     std::cout <<    " keyframe         " << keyframe_storage[i].keyframe.Frame_id << "  " << endl;
            //     std::cout <<    " keyfrmae pts size" << keyframe_storage[i].keyframe.pts.size() << "   " << endl;
            //     cv::imshow("keyframe2", keyframe_storage[i].keyframe.frame);    
            // }            
        }
        if (times % 5 == 0)
        {
            Keyframe keyframe_(keyframe_num);
            keyframe_.keyframe = current_image;
            keyframe_num += 1;
            keyframe_storage.push_back(keyframe_);
            for (int i = 0; i < keyframe_num; i++) 
            {
            //     std::cout <<    " keyframe id   :  " << keyframe_storage[i].keyframe_id << "   " << endl;
            //     std::cout <<    " Frame id         " << keyframe_storage[i].keyframe.Frame_id << "  " << endl;
            //     std::cout <<    " keyframe cam pose" << endl << keyframe_storage[i].keyframe.cam_pose_ << "  " << endl;
                cv::imshow(" keyframe image ", keyframe_storage[i].keyframe.frame);      
            }
        }
        
        previous_image = current_image;

        ++times;
        
        // double fps = video.set(CAP_PROP_FPS, 30);
        double delay = cvRound(1000/30);
        // cv::waitKey(delay);
        // if(times > 540) 
        cv::waitKey();
    }

    return 0;
}
            
        

       

                
    


            


                        
        
            
            

                
            
                