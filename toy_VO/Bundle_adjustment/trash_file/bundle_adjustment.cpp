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
    int initialize_frame_num = 8;
    int track_inlier_ratio;
    bool new_keyframe_selection = false;
    bool caculate_triangulation = true;

    int times = 1;
    int keyframe_num = 0;
    int initial_bundle_frame_num = 0;
    
    
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
    // std::vector<Keyframe> keyframe_storage;
    std::vector<Map> map_storage;

    Frame previous_image(0);
    // for(int i = 0; i < 3; i++) previous_image.cam_pose[i] = 0;
    video >> previous_image.frame;
    
    Frame copy_previous_image = previous_image;
    frame_storage.push_back(copy_previous_image);    
    
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


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////        Caculate R, t using ESSENTIAL MATRIX or HOMOGRAPHY      //////////////////////////////////////////// 
        if ( times < initialize_frame_num + 1)
        {
            previous_image.pts.clear();
            
            // Find feature in previous_image 
            std::cout << " feature track " << endl;
            cv::goodFeaturesToTrack(previous_image.frame, previous_image.pts, 2000, 0.01, 10);

            // Matching prev_image and current_image using optical flow
            std::cout << " optical flow " << endl;
            std::vector<int> correspond_id = track_opticalflow_and_remove_err(previous_image.frame, current_image.frame, previous_image.pts, current_image.pts);
            
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
            


            // Reprojection 3D to 2D current image
            std::cout << "reprojection 3D to 2D initialize stage " << endl;
            cv::sfm::KRtFromProjection(P1, K, R_, t_);
            cv::Mat project_mat = Mat(map_point).clone();
            project_mat.convertTo(project_mat, CV_32F);
            cv::projectPoints(project_mat, R_, t_, K, cv::noArray(), projectionpoints);                
                


        }
            




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////            
//////////////////////////      SOLVEPNP + TRIANGULATION             ////////////////////////////////////////////////////// 
        if (times > initialize_frame_num)
        {
            // Matching prev_image and current_image using optical flow
            std::cout << "optical flow" << endl;
            float before_track_feature_num = previous_image.pts.size();
            std::vector<int> correspond_id = track_opticalflow_and_remove_err_for_SolvePnP(previous_image.frame, current_image.frame, previous_image.pts, current_image.pts, map_point);
            
                

            // Storage corresponding Id
            for(int i = 0; i < current_image.pts.size(); i++) current_image.pts_id.push_back(previous_image.pts_id[correspond_id[i]]);
            // for(int i = 0; i < current_image.pts_id.size(); i++) std::cout << current_image.pts_id[i] << "    ";           

std::cout << "SolvePnP map_point num : " << "   " << map_point.size() << endl;
            
            // Calculate World R, t using SolvePnP
            std::cout << "SolvePnP" << endl;
            cv::solvePnPRansac(map_point, current_image.pts, K, cv::noArray(), World_R, World_t, false, 100, 3.0F, 0.99, inliers );

std::cout << " Inlier ratio : " << 100 * inliers.rows / map_point.size() << " %" << endl;
            
            // Calculate camera_pose
            std::cout << "caculate pose" << endl;
            cv::Rodrigues(World_R, World_R);
            World_Rt = R_t_to_homogenous(World_R, World_t);
            // cam_pose = -World_R.inv() * World_t;
            cam_pose = World_Rt.inv();


            track_inlier_ratio = 100 * current_image.pts.size() / map_storage[keyframe_num].keyframe_[keyframe_num].pts.size();std::cout << "  track inlier ratio : " << track_inlier_ratio << endl;

            // Determinate select keyframe or not
            if (track_inlier_ratio < 60) new_keyframe_selection = true;
            if (new_keyframe_selection)
            {


                // Bundle Adjustment
                // Define CostFunction
                ceres::Problem ba;
                std::vector<cv::Vec6d> campose_for_bundle;
                for(int j = initial_bundle_frame_num; j < times - 1; j++) campose_for_bundle.push_back(homogenous_campose_to_vec6d(frame_storage[j].cam_pose_));

                for(int j = initial_bundle_frame_num; j < times - 1; j++)
                {
                    for ( int i = 0; i < frame_storage[j].pts.size(); i++)
                    {
                            ceres::CostFunction* cost_func = ReprojectionError::create(frame_storage[j].pts[i], f, cv::Point2d(c.x, c.y));
                            
                            double* camera = (double*)(&campose_for_bundle[j-initial_bundle_frame_num]);
                            int id_ = frame_storage[j].pts_id[i];
                            double* X_ = (double*)(&(map_storage[keyframe_num].world_xyz[id_][id_]));
                            ba.AddResidualBlock(cost_func, NULL, camera, X_); 
                            
                    }            
                }

                //////////////////////////////////////////
                cv::Vec6d ab = homogenous_campose_to_vec6d(cam_pose);
                for ( int i = 0; i < current_image.pts.size(); i++)
                {
                            ceres::CostFunction* cost_func = ReprojectionError::create(current_image.pts[i], f, cv::Point2d(c.x, c.y));
                            // cv::Vec6d ab = homogenous_campose_to_vec6d(cam_pose);
                            double* camera = (double*)(&ab);
                            int id_ = current_image.pts_id[i];
                            double* X_ = (double*)(&(map_storage[keyframe_num].world_xyz[id_][id_]));
                            ba.AddResidualBlock(cost_func, NULL, camera, X_); 
                            
                }    
                //////////////////////////////////////////
                // ceres option       
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::ITERATIVE_SCHUR;
                options.num_threads = 8;
                options.minimizer_progress_to_stdout = true;
                ceres::Solver::Summary summary;
                
                // Camera pose and map_point before BA
                std::cout <<" camera pose before BA " << endl;
                for(int j = initial_bundle_frame_num; j < times - 1; j++) std::cout << " Frame num : " << j << endl << frame_storage[j].cam_pose_ << endl;
                std::cout << " first map point value before BA " << endl;
                std::cout << map_storage[keyframe_num].world_xyz[0][0] << endl;


                // solve
                ceres::Solve(options, &ba, &summary);                
                
                for(int j = initial_bundle_frame_num; j < times - 1; j++) frame_storage[j].cam_pose_ = vec6d_to_homogenous_campose(campose_for_bundle[j-initial_bundle_frame_num]);
                ////////////////////
                cam_pose = vec6d_to_homogenous_campose(ab);
                ///////////////////
                // Camera pose and map_point after BA
                std::cout <<" camera pose after BA " << endl;
                for(int j = initial_bundle_frame_num; j < times - 1; j++) std::cout << " Frame num : " << j << endl << frame_storage[j].cam_pose_ << endl;                
                std::cout << " first map point value after BA " << endl;
                std::cout << map_storage[keyframe_num].world_xyz[0][0] << endl;           

                previous_image.pts.clear();
                cv::goodFeaturesToTrack(previous_image.frame, previous_image.pts, 2000, 0.01, 10);

                track_opticalflow_and_remove_err(previous_image.frame, current_image.frame, previous_image.pts, current_image.pts);

                current_image.pts_id.clear();
                for(int i = 0; i < current_image.pts.size(); i++) current_image.pts_id.push_back(i);
            }
            




std::cout << "previous cam pose : " << endl << frame_storage[times - 1].cam_pose_ << endl;
std::cout << "current cam pose : " << endl << cam_pose << endl;
            cv::Mat pose_difference = cv::abs(cam_pose) - cv::abs(frame_storage[times - 1].cam_pose_);
std::cout << " pose difference   : " << endl << pose_difference << endl;
            double translation_difference = std::abs(pose_difference.at<double>(0, 3)) + std::abs(pose_difference.at<double>(1, 3)) + std::abs(pose_difference.at<double>(2, 3));
std::cout << "translation difference : " << translation_difference << endl;

            // Find new feature in previous image 
            std::cout << "  find feature    " << endl;
            

            // Determinate doing triangulation or not 
            // if (translation_difference < 0.06) caculate_triangulation = false;
            if (translation_difference < 0.06 and 100 * previous_image.pts.size() / before_track_feature_num > 95 ) caculate_triangulation = false;
            else caculate_triangulation = true;
            if (caculate_triangulation)
            {

                

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
                
            }
            


        

            
            // Storage camera pose
            std::cout << "  storage cam pose   " << endl;
            cv::Mat cam_pose_ = cam_pose.clone();
            current_image.cam_pose_.push_back(cam_pose_);
            
            

            
            
            
            



    
        }


        // Storage current image 
        Frame copy_current_image = current_image;
        frame_storage.push_back(copy_current_image);


        // Reprojection 3D to 2D current image
        std::cout << "reprojection 3D to 2D at SolvePnP stage " << endl;
        if(times > initialize_frame_num) 
        {
            cv::Mat project_mat = Mat(map_point).clone();
            project_mat.convertTo(project_mat, CV_32F);
            cv::projectPoints(project_mat, World_R, World_t, K, cv::noArray(), projectionpoints); 
        }
        




        
        int copy = initial_bundle_frame_num;

        



        
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////    
///////////////                  KEYFRAME STORAGE              ///////////////////////////////////////////////////
        
        // Keyframe_storage.
        if (times == initialize_frame_num)
        {
            Map map;
            
            // storage id - keyframe
            map.keyframe_.insert(std::pair<int, Frame>(keyframe_num, current_image));

            // storage id - landmark
            for(int i = 0; i < map_point.size(); i++) 
            {
                std::map<int, cv::Point3d> xyz_;
                xyz_.insert(std::pair<int, cv::Point3d>(i, map_point[i]));
                map.world_xyz.push_back(xyz_);
            }
            initial_bundle_frame_num = times;
            map_storage.push_back(map);
            
            
            cv::imshow(" keyframe image ", map_storage[keyframe_num].keyframe_[keyframe_num].frame);

std::cout << "@@@@@@@@@@ First keyframe selection @@@@@@@" << endl << " keyframe num is  : " << keyframe_num << endl;                    
// for(int j = 0 ; j < map_storage[keyframe_num].world_xyz.size(); j++) std::cout << " keyframe(map_storage) landmark num : " << map_storage[keyframe_num].world_xyz[j] << endl;
// for(int j = 0 ; j < map_storage[keyframe_num].world_xyz.size(); j++) print_map(map_storage[keyframe_num].world_xyz[j]);
        }
        if (new_keyframe_selection)
        {
            keyframe_num += 1;
            Map map;
            
            // storage id - keyframe
            map.keyframe_.insert(std::pair<int, Frame>(keyframe_num, current_image));

            // storage id - landmark
            for(int i = 0; i < map_point.size(); i++) 
            {
                std::map<int, cv::Point3d> xyz_;
                xyz_.insert(std::pair<int, cv::Point3d>(i, map_point[i]));
                map.world_xyz.push_back(xyz_);
            }          
            initial_bundle_frame_num = times;
            map_storage.push_back(map);
            
            cv::imshow(" keyframe image ", map_storage[keyframe_num].keyframe_[keyframe_num].frame);
            
std::cout << "@@@@@@@@@ New keyframe was made @@@@@@@@@" << endl << " Keyframe num is  : " << keyframe_num << endl; 
        }
        

 






////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////               SHOW IMAGE                ////////////////////////////////////////////////////////////      
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





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////            VISUALIZE         ////////////////////////////////////////////////////////////////
        
        // Show gt trajectory (red line)
        // GLfloat inputbuffer[12];
        // for(int j=0;j<12;j++) GT_data >> inputbuffer[j];
        // GLfloat x_gt(inputbuffer[3]), y_gt(inputbuffer[7]), z_gt(inputbuffer[11]);
        // show_trajectory(x_gt, y_gt, z_gt, 1.0, 0.0, 0.0, 3.0);

        // cv::Mat_<double> gt_pose(3, 4);
        // gt_pose << inputbuffer[0], inputbuffer[1], inputbuffer[2], inputbuffer[3], inputbuffer[4], inputbuffer[5], inputbuffer[6], inputbuffer[7], inputbuffer[8], inputbuffer[9], inputbuffer[10], inputbuffer[11];
        // std::cout << " GT pose  : "  << endl << gt_pose << endl;        

        // Show camera estimate pose trajectory ( green line )
        // GLfloat _x_cam_pose(current_image.cam_pose_.at<double>(0, 3)), _y_cam_pose(current_image.cam_pose_.at<double>(1, 3)), _z_cam_pose(current_image.cam_pose_.at<double>(2, 3));
        // show_trajectory(_x_cam_pose, _y_cam_pose, _z_cam_pose, 0.0, 1.0, 0.0, 3.0);

        // show camera estimate pose trajectory aftet BA
        if (new_keyframe_selection)
        { 
            for(int j = copy; j < times - 1; j++) 
            {
                GLfloat x_cam_pose(frame_storage[j].cam_pose_.at<double>(0, 3)), y_cam_pose(frame_storage[j].cam_pose_.at<double>(1, 3)), z_cam_pose(frame_storage[j].cam_pose_.at<double>(2, 3));
                show_trajectory(x_cam_pose, y_cam_pose, z_cam_pose, 0.0, 1.0, 0.0, 3.0);
            }
        }
        
        // Show 3d keyframe map point ( blue dot )
        if (times == initialize_frame_num)
        {
            for( int i = 0 ; i < map_storage[keyframe_num].world_xyz.size(); i++)
            {
                // GLfloat X_map(map_storage[keyframe_num].world_xyz[i][i].x), Y_map(map_storage[keyframe_num].world_xyz[i][i].y), Z_map(map_storage[keyframe_num].world_xyz[i][i].z);
                // show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 1.0, 0.5);
            }        
        }
        if (new_keyframe_selection)
        {   
            if (keyframe_num % 5 == 0)
            {
            for( int i = 0 ; i < map_storage[keyframe_num].world_xyz.size(); i++)
            {
                // GLfloat X_map(map_storage[keyframe_num].world_xyz[i][i].x), Y_map(map_storage[keyframe_num].world_xyz[i][i].y), Z_map(map_storage[keyframe_num].world_xyz[i][i].z);
                // show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 1.0, 0.1);
            }         
            new_keyframe_selection = false;
            }
        }

        glFlush();

        

        previous_image = current_image;
        ++times;
        
        // double fps = video.set(CAP_PROP_FPS, 30);
        double delay = cvRound(1000/30);
        cv::waitKey(delay);
        // if(times > 540) 
        // cv::waitKey();
    }

    return 0;
}
            
        

       

                
    


            


                        
        
            
            

                
            
                