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
// // #include "ceres"
#include "ceres/ceres.h"
#include "glog/logging.h"


using namespace std;
using namespace cv;
using ceres::CauchyLoss;

int main(int argc, char **argv)
{       

////////////////////////////////////////////////////////////////////////////////////    
///////////// Read Data ( Camera Calibration and Image Data ) //////////////////////
////////////////////////////////////////////////////////////////////////////////////    
    
    // Read Calibration Data
    double intrinsic[12];  
    Read_Kitti_Calibration_Data(argv, intrinsic);
    
    // f : focal length , c : principal point (c.x, c.y) , K : Camera Matrix
    double f = intrinsic[0];
    cv::Point2d c(intrinsic[2], intrinsic[6]);
    const cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, c.x, 0, f, c.y, 0, 0, 1);
    std::vector<double> dist_coeff = { -0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 0.0001149909868437517, -0.01803978785585194 };
            
    std::cout << " Camera Matrix  : " << endl << K << endl;
    
    
    // Read Image Data
    cv::VideoCapture video;
    if (!video.open(Read_Kitti_image_Data(argv))) return -1;
    
    // Read GT Data
    ifstream GT_DATA = Read_Kiiti_GT_Data(argv);

    
    glutInit(&argc, argv);
    initialize_window();
 

    int initialize_frame_num = 8;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////          
    int show_map_point_parms = 0;
    int show_trajectory_parms = 0;
    int show_fix_map_point_parms = 0;

    int fix_intial_keyfram_num = 0;
    int optimize_intial_keyframe_num = 1;
    int track_entire_num;    
    int track_inlier_ratio;
    int feature_max_ID;
    bool new_keyframe_selection = false;
    bool caculate_triangulation = true;


    int times = 1;
    int keyframe_num = 0;
    
    cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat P1;
    cv::Mat cam_pose_initialize = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat cam_pose = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat Iden = Mat::eye(4, 4, CV_64F);
    std::vector<Point3d> map_point;
    
    // cv::Mat R, t, Rt;
    
    cv::Mat World_R, World_t, World_Rt;
    std::vector<cv::Point2f> projectionpoints;
    
    cv::Mat X, map_change;
    std::vector<cv::Point2f> previous_track_point_for_triangulate, current_track_point_for_triangulate, keyframe_track_point;
    std::vector<int> previous_track_point_for_triangulate_ID;
    std::vector<cv::Point3d> SolvePnP_tracking_map;
    std::vector<int> SolvePnP_tracking_map_ID;

    // Storage Storage_frame;
    std::vector<cv::Mat> inlier_storage;    
    std::vector<Frame> frame_storage;
    Map map_storage;


    // const int image_x_size = previous_image.frame.cols;
    // const int image_y_size = previous_image.frame.rows;
    

    
    
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    
    Frame previous_image(0);
    
    video >> previous_image.frame;
    if(previous_image.frame.empty()) cv::waitKey();
    if (previous_image.frame.channels() > 1) cv::cvtColor(previous_image.frame, previous_image.frame, cv::COLOR_RGB2GRAY);
    previous_image.cam_pose = homogenous_campose_to_vec6d(cam_pose_initialize);
    
    Frame copy_previous_image = previous_image;
    frame_storage.push_back(copy_previous_image); 
        
    
    while(true)
    {
        // count image num
        std::cout << times << " Frame !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        
        // push new image
        Frame current_image(times);
        cv::Mat image;
        video >> image;
        if (image.empty()) break;
        if (image.channels() > 1) cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
        current_image.frame = image.clone();


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////        Caculate R, t using ESSENTIAL MATRIX or HOMOGRAPHY      //////////////////////////////////////////// 
        if ( times < initialize_frame_num + 1)
        {
            
            // Find feature in previous_image 
            
            std::cout << " feature track " << endl;
            if(times == 1)
            {
                cv::goodFeaturesToTrack(previous_image.frame, previous_image.pts, 1500, 0.01, 10);
std::cout << " new feature num  : " << previous_image.pts.size() << endl;   
                for(int i = 0; i < previous_image.pts.size(); i++)
                {
                    keyframe_track_point.push_back(previous_image.pts[i]);
                    // previous_track_point_for_triangulate.push_back(previous_image.pts[i]);
                }
            }
            // Matching prev_image and current_image using optical flow
            std::cout << " optical flow " << endl;
            // track_opticalflow_and_remove_err_for_triangulate(previous_image.frame, current_image.frame, previous_track_point_for_triangulate, current_track_point_for_triangulate, keyframe_track_point);
            track_opticalflow_and_remove_err_for_triangulate(previous_image.frame, current_image.frame, previous_image.pts, current_image.pts, keyframe_track_point);
std::cout << "after track_feature num  : " << current_image.pts.size() << endl;            
            
            // Caculate relative R, t using Essential Matrix
            std::cout << "essential matrix" << endl;
            cv::Mat E, inlier_mask, R, t, Rt;
            E = cv::findEssentialMat(previous_image.pts, current_image.pts, f, c, cv::RANSAC, 0.99, 1, inlier_mask);
            int inlier_num = cv::recoverPose(E, previous_image.pts, current_image.pts, R, t, f, c, inlier_mask);
            
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
            cv::Mat clone_cam_pose_initialize;
            clone_cam_pose_initialize = cam_pose_initialize.clone();
            current_image.cam_pose = homogenous_campose_to_vec6d(clone_cam_pose_initialize);
std::cout << " current cam pose " << endl << clone_cam_pose_initialize << endl;



            


            if(times < initialize_frame_num)
            {
                // Storage current image 
                Frame copy_current_image = current_image;
                frame_storage.push_back(copy_current_image);
            }

            if(times == initialize_frame_num)
            {
                
                // Caculate projection matrix
                P1 = K * cam_storage_to_projection_matrix(current_image.cam_pose);
                
                // Triangulation between keyframe
                std::cout << " triangulation " << endl;
                cv::triangulatePoints(P0, P1, keyframe_track_point, current_image.pts, X);
                
                // map_point
                std::cout << " storage map point " << endl;
                world_xyz_point_to_homogenous(X);
                X.convertTo(X, CV_64F);
                map_point.clear();
                for (int i = 0; i < keyframe_track_point.size(); i++ ) map_point.push_back(Point3d(X.at<double>(0, i), X.at<double>(1, i), X.at<double>(2, i)));

                
                // remove map_point outlier
std::cout << " map_point size before remove outlier : " <<  map_point.size() << endl;
                remove_map_point_and_2dpoint_outlier(map_point,  current_image.pts, clone_cam_pose_initialize);
std::cout << " map_point size after remove outlier : " <<  map_point.size() << endl;               


                cv::Mat rot, tran, inliers;
                cv::solvePnPRansac(map_point, current_image.pts, K, cv::noArray(), rot, tran, false, 100, 3.0F, 0.99, inliers );
                inlier_storage.push_back(inliers);
std::cout << keyframe_num <<"  keyframe inlier storage rate : " << 100 * inliers.rows / map_point.size() << endl;

                remove_SolvePnP_oulier(map_point, current_image.pts, inliers);
std::cout << " map_point size after remove solvepnp outlier : " <<  map_point.size() << endl;
                
                
                


                // Storage pts, pts_id
                for(int i = 0; i < current_image.pts.size(); i++) current_image.pts_id.push_back(i);
                feature_max_ID = current_image.pts.size() -1;

                
                // Storage current image 
                Frame copy_current_image = current_image;
                frame_storage.push_back(copy_current_image);

                // storage id - keyframe
                map_storage.keyframe.insert(std::pair<int, Frame>(keyframe_num, frame_storage[times]));
cv::imshow(" keyframe image ", map_storage.keyframe[keyframe_num].frame);
std::cout << "@@@@@@@@@@ First keyframe selection @@@@@@@" << endl << " keyframe num is  : " << keyframe_num << endl;                    


                // storage id - landmark
                for(int i = 0; i < map_point.size(); i++) 
                {
                    std::map<int, cv::Point3d> xyz_;
                    xyz_.insert(std::pair<int, cv::Point3d>(i, map_point[i]));
                    map_storage.world_xyz.push_back(xyz_);

                    // SolvePnP_tracking_map.push_back(map_point[i]);
                    // SolvePnP_tracking_map_ID.push_back(i);
                }       

std::cout << " Landmark's num : " << map_storage.world_xyz.size() << endl;
std::cout << " same as landmark num :::: keyframe 2d point size : " << map_storage.keyframe[keyframe_num].pts.size() <<endl;

                // new feature to track
                std::cout << " new feature " << endl;
                current_track_point_for_triangulate.clear();
                cv::goodFeaturesToTrack(current_image.frame, current_track_point_for_triangulate, 1500, 0.01, 10);
std::cout << " new feature num  : " << current_track_point_for_triangulate.size() << endl; 
                track_entire_num = current_track_point_for_triangulate.size();
                
                keyframe_track_point.clear();
                for(int i = 0; i < current_track_point_for_triangulate.size(); i++) keyframe_track_point.push_back(current_track_point_for_triangulate[i]);

                previous_track_point_for_triangulate.clear();
                for(int i = 0; i < current_track_point_for_triangulate.size(); i++) previous_track_point_for_triangulate.push_back(current_track_point_for_triangulate[i]);

                std::vector<cv::Point2f> clone_pts(current_image.pts);
                std::vector<int> clone_pts_id(current_image.pts_id);
                for(int j =0; j < current_track_point_for_triangulate.size(); j++)
                {
                    bool found_pts2d = false;
                    for(int i = 0; i < clone_pts.size(); i++)
                    {
                        double difference_point2d_x = cv::abs(clone_pts[i].x - current_track_point_for_triangulate[j].x);
                        double difference_point2d_y = cv::abs(clone_pts[i].y - current_track_point_for_triangulate[j].y);

                        if(difference_point2d_x < 3.0 and difference_point2d_y < 3.0)
                        {
                            found_pts2d = true;
                            previous_track_point_for_triangulate_ID.push_back(clone_pts_id[i]);
                            clone_pts.erase(clone_pts.begin() + i);
                            clone_pts_id.erase(clone_pts_id.begin() + i);
                            break;
                        }
                    }
                    
                    if(found_pts2d) found_pts2d= false;
                    else
                    {
                        previous_track_point_for_triangulate_ID.push_back(feature_max_ID++);
                    }

                }




                // BA initialize
                // Define CostFunction
                ceres::Problem initilize_ba;

                for ( int i = 0; i < map_storage.keyframe[0].pts.size(); i++)
                {
                    ceres::CostFunction* cost_func = ReprojectionError::create(map_storage.keyframe[0].pts[i], f, cv::Point2d(c.x, c.y));
                    double* camera = (double*)(&map_storage.keyframe[0].cam_pose);
                    double* X_ = (double*)(&(map_storage.world_xyz[i][i]));
                    initilize_ba.AddResidualBlock(cost_func, NULL, camera, X_); 
                }            
                            
                // ceres option       
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::ITERATIVE_SCHUR;
                options.num_threads = 8;
                options.minimizer_progress_to_stdout = true;
                ceres::Solver::Summary summary;

                std::cout << " camera cam pose before intialize ba " << endl <<vec6d_to_homogenous_campose(map_storage.keyframe[0].cam_pose) << endl;

                ceres::Solve(options, &initilize_ba, &summary);

                std::cout << " camera cam pose after intialize ba " << endl <<vec6d_to_homogenous_campose(map_storage.keyframe[0].cam_pose) << endl;
            
                // Reprojection 3D to 2D current image
                std::cout << "reprojection 3D to 2D initialize stage " << endl;
                cv::Mat R_, t_;
                cv::sfm::KRtFromProjection(P1, K, R_, t_);
                // map_point.clear();
                // for (int c = 0; c < map_storage.world_xyz.size(); c++ ) map_point.push_back(Point3d(map_storage.world_xyz[c][c].x, map_storage.world_xyz[c][c].y, map_storage.world_xyz[c][c].z));
                cv::Mat project_mat = Mat(map_point).clone();
                project_mat.convertTo(project_mat, CV_32F);
                cv::projectPoints(project_mat, R_, t_, K, cv::noArray(), projectionpoints);                
            
            }
        
        
        }
            




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////            
// //////////////////////////      SOLVEPNP + TRIANGULATION             ////////////////////////////////////////////////////// 
        if (times > initialize_frame_num)
        {
            // Matching prev_image and current_image using optical flow
            std::cout << "optical flow" << endl;
            
            float before_track_feature_num = previous_image.pts.size();
std::cout << "before_track_feature_num  : " << before_track_feature_num << endl;
// for(int i = 0; i < previous_image.pts_id.size(); i++) std::cout << previous_image.pts_id[i] << "    ";
            
            track_opticalflow_and_remove_err_for_SolvePnP(previous_image.frame, current_image.frame, previous_image.pts, current_image.pts, previous_image.pts_id, map_point);

std::cout << "after track_feature num  : " << current_image.pts.size() << endl;  
            
                
            // Storage corresponding Id
            for(int i = 0; i < current_image.pts.size(); i++) current_image.pts_id.push_back(previous_image.pts_id[i]);
std::cout << endl;
// for(int i = 0; i < current_image.pts_id.size(); i++) std::cout << current_image.pts_id[i] << "    ";           

////////////////////////////////////// track for triangulate///////////////////////////////////////////////////////////////////////////////////////////
            
                

            std::cout << " feature track " << endl;
std::cout << previous_track_point_for_triangulate.size() << "   " << current_track_point_for_triangulate.size() << "    " << previous_track_point_for_triangulate_ID.size() << "   " << keyframe_track_point.size() << endl;
            track_opticalflow_and_remove_err_for_triangulate_(previous_image.frame, current_image.frame, previous_track_point_for_triangulate, current_track_point_for_triangulate, previous_track_point_for_triangulate_ID, keyframe_track_point);
std::cout << previous_track_point_for_triangulate.size() << "   " << current_track_point_for_triangulate.size() << "    " << previous_track_point_for_triangulate_ID.size() << "   " << keyframe_track_point.size() << endl;            

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            // Calculate World R, t using SolvePnP
            std::cout << "SolvePnP" << endl;
std::cout << "SolvePnP map_point num : " << "   " << map_point.size() << endl;
std::cout << "SolvePnP current_image_pts num : " << "   " << current_image.pts.size() << endl;
            cv::Mat inliers; 
            cv::solvePnPRansac(map_point, current_image.pts, K, cv::noArray(), World_R, World_t, false, 100, 3.0F, 0.99, inliers );
            int SolvePnP_inlier_ratio = 100 * inliers.rows / map_point.size();
std::cout << " Inlier ratio : " << SolvePnP_inlier_ratio << " %" << endl;


            // Calculate camera_pose
            std::cout << "caculate pose" << endl;
            cv::Rodrigues(World_R, World_R);
            World_Rt = R_t_to_homogenous(World_R, World_t);
            // cam_pose = -World_R.inv() * World_t;
            cam_pose = World_Rt.inv();

            // Storage camera pose
            std::cout << "  storage cam pose   " << endl;
            cv::Mat cam_pose_ = cam_pose.clone();
            current_image.cam_pose = homogenous_campose_to_vec6d(cam_pose_);

            std::cout << "previous cam pose : " << endl << vec6d_to_homogenous_campose(frame_storage[times - 1].cam_pose) << endl;
            std::cout << "( before motion only BA ) current cam pose : " << endl << vec6d_to_homogenous_campose(current_image.cam_pose) << endl;

            //
            std::vector<cv::Point3d> map_point_for_motion_BA;
            std::vector<cv::Point2f> current_image_pts_for_motion_BA;
  
            // for(int i = 0; i < map_point.size(); i++) map_point_for_motion_BA.push_back(map_point[i]);
            // for(int i = 0; i < current_image.pts.size(); i++) current_image_pts_for_motion_BA.push_back(current_image.pts[i]);
            for(int i = 0; i < inliers.rows; i++)
            {
                map_point_for_motion_BA.push_back(map_point[inliers.at<int>(i, 0)]);
                current_image_pts_for_motion_BA.push_back(current_image.pts[inliers.at<int>(i, 0)]);
                
            }
std::cout << " map point size for motion only BA : " << map_point_for_motion_BA.size() << endl;

            // Motion only BA
            ceres::Problem motion_only_ba;
            
            for ( int i = 0; i < current_image_pts_for_motion_BA.size(); i++)
            {
                ceres::CostFunction* motion_only_cost_func = motion_only_ReprojectionError::create(current_image_pts_for_motion_BA[i], map_point_for_motion_BA[i], f, cv::Point2d(c.x, c.y));
                double* camera_ = (double*)(&current_image.cam_pose);
        
                motion_only_ba.AddResidualBlock(motion_only_cost_func, NULL, camera_); 
            }            
            
            // ceres option       
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::ITERATIVE_SCHUR;
            options.num_threads = 12;
            options.minimizer_progress_to_stdout = false;
            ceres::Solver::Summary summary;
                            
            // solve
            ceres::Solve(options, &motion_only_ba, &summary);

            std::cout <<" current camera pose after motion only BA  : " << endl;
            std::cout << vec6d_to_homogenous_campose(current_image.cam_pose) << endl;

            // Determine Keyframe or not
            // track_inlier_ratio 
            track_inlier_ratio = 100 * current_track_point_for_triangulate.size() / track_entire_num;
std::cout << "  track inlier ratio : " << track_inlier_ratio << endl;

            // Rotation value 
std::cout << "previous vec6d value : " << previous_image.cam_pose << endl;
std::cout << "current vec6d value  : " << current_image.cam_pose << endl;
            double rotation_difference = cv::abs(previous_image.cam_pose[1] - current_image.cam_pose[1]);
std::cout << " Rotation difference : " << rotation_difference << endl;           


            // Determinate select keyframe or not
            if (track_inlier_ratio < 70 or SolvePnP_inlier_ratio < 60) new_keyframe_selection = true;
            if (new_keyframe_selection == false)
            {
                

                // Storage current image 
                Frame copy_current_image = current_image;
                frame_storage.push_back(copy_current_image);
            }            
            
            
            
            cv::Mat pose_difference = cv::abs(vec6d_to_homogenous_campose(current_image.cam_pose)) - cv::abs(vec6d_to_homogenous_campose(previous_image.cam_pose));
std::cout << " pose difference   : " << endl << pose_difference << endl;
            double translation_difference = std::abs(pose_difference.at<double>(0, 3)) + std::abs(pose_difference.at<double>(1, 3)) + std::abs(pose_difference.at<double>(2, 3));
std::cout << "translation difference : " << translation_difference << endl;            
            
            if (new_keyframe_selection)
            {
                
                
/////////////////////////////////////////////////  Inlier Storage //////////////////////////////////////////////////////////////////////            
               
//                 inlier_storage.push_back(inliers);
// std::cout << keyframe_num + 1 << " keyframe inlier storage rate : " << 100 * inliers.rows / map_point.size() << endl;                

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//                 // Determinate doing triangulation or not 
                // if (translation_difference < 0.06 and (100 * previous_image.pts.size() / before_track_feature_num) > 96 ) caculate_triangulation = false;
                // else caculate_triangulation = true;
                if (caculate_triangulation)
                {
                

//                  // Caculate projection matrix for triangulation
                    std::cout << "  projection matrix    " << endl;
                    P0 = K * cam_storage_to_projection_matrix(map_storage.keyframe[keyframe_num].cam_pose);
                    P1 = K * cam_storage_to_projection_matrix(current_image.cam_pose);
                    
            
                    // Triangulation previous_image and current_image
                    std::cout << "  triangulation    " << endl;
                    cv::triangulatePoints(P0, P1, keyframe_track_point, current_track_point_for_triangulate, X);

                    // Map_point
                    std::cout << "  storage map    " << endl;

                    world_xyz_point_to_homogenous(X);
                    X.convertTo(X, CV_64F);
                    map_point.clear();
                    for (int c = 0; c < current_track_point_for_triangulate.size() ; c++ ) map_point.push_back(Point3d(X.at<double>(0, c), X.at<double>(1, c), X.at<double>(2, c)));
std::cout << map_point.size() << "      " << current_track_point_for_triangulate.size() << endl;
                
std::cout << " map point size this keyframe before remove outlier  : " << map_point.size() << endl;
                    cam_pose_ = vec6d_to_homogenous_campose(current_image.cam_pose);
                    remove_map_point_and_2dpoint_outlier_(map_point, current_track_point_for_triangulate, previous_track_point_for_triangulate_ID, cam_pose_);
std::cout << " map point size this keyframe after remove outlier  : " << map_point.size() << endl;



                }            


            
/////////////////////////////////////////////////  Inlier Storage //////////////////////////////////////////////////////////////////////            
               
                cv::Mat rot, tran, inliers_;
std::cout << map_point.size() << "      " << current_track_point_for_triangulate.size() << endl;
                cv::solvePnPRansac(map_point, current_track_point_for_triangulate, K, cv::noArray(), rot, tran, false, 100, 3.0F, 0.99, inliers_ );
                inlier_storage.push_back(inliers_);
std::cout << keyframe_num + 1 << " keyframe inlier storage rate : " << 100 * inliers_.rows / map_point.size() << endl;                

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


                remove_SolvePnP_oulier_(map_point, current_track_point_for_triangulate, previous_track_point_for_triangulate_ID, inliers_);
std::cout << " map_point size after remove solvepnp outlier : " <<  map_point.size() << endl;
                
                // storage landmark id
                for(int i = 0; i < current_track_point_for_triangulate.size(); i++)
                {
                    std::map<int, cv::Point3d> xyz_;
                    xyz_.insert(std::pair<int, cv::Point3d>(previous_track_point_for_triangulate_ID[i], map_point[i]));
                    map_storage.world_xyz.push_back(xyz_);
                }

                
  
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////                               
////////////////////////////////////////////// correspond Landmark and new map point /////////////////////////////////////////////////

/////////////////////////////////////////    11111111111111111111111111111111111   /////////////////////////////////////////////////////
// std::cout << "map storage map size : " << map_storage.world_xyz.size() << endl;               
// std::cout << "map_point size  : "  << map_point.size() << endl;
//                 int map_ID = map_storage.world_xyz.size();
//                 std::vector<int> correspond_id_storage;
//                 bool found_pts = false;
        


// // for(int i = 0; i < current_image.pts_id.size(); i++) std::cout << current_image.pts[i] << "   ";
// // std::cout << endl;
//                 for(int j = 0; j < current_track_point_for_triangulate.size(); j++)
//                 {
// // std::cout << current_image.pts.size() << endl;                    
//                     for(int i = 0; i < current_image.pts.size(); i++)
//                     {
//                         double image_point_defference_x = cv::abs(current_image.pts[i].x - current_track_point_for_triangulate[j].x);
//                         double image_point_defference_y = cv::abs(current_image.pts[i].y - current_track_point_for_triangulate[j].y);
//                         // same image point
//                         if (image_point_defference_x < 1.0 and image_point_defference_y < 1.0)
//                         {
//                             correspond_id_storage.push_back(current_image.pts_id[i]);
//                             current_image.pts.erase(current_image.pts.begin() + i);
//                             current_image.pts_id.erase(current_image.pts_id.begin() + i);
//                             found_pts= true;
//                             break;
                        
//                         }
                       
//                     }
//                     if(found_pts){ found_pts = false;}
//                     else
//                     {
//                             // different image point
//                             correspond_id_storage.push_back(map_ID);
//                             std::map<int, cv::Point3d> xyz_;
//                             xyz_.insert(std::pair<int, cv::Point3d>(map_ID, map_point[j]));
//                             map_storage.world_xyz.push_back(xyz_);
//                             map_ID++;
                            
//                     }

                    
//                 }
// // for(int i = 0; i < correspond_id_storage.size(); i++) std::cout << correspond_id_storage[i] << "   ";
// // std::cout << endl;
// std::cout << current_track_point_for_triangulate.size() << "    " << correspond_id_storage.size() << endl;
                
// std::cout << " map_ID " <<  map_ID << endl;
                
                
                // Storage pts
                current_image.pts.clear();
                current_image.pts_id.clear();
                for(int i = 0; i < current_track_point_for_triangulate.size(); i++) 
                {
                    current_image.pts.push_back(current_track_point_for_triangulate[i]);                
                    current_image.pts_id.push_back(previous_track_point_for_triangulate_ID[i]);
                }

std::cout << " size print " << current_track_point_for_triangulate.size() << "    " << previous_track_point_for_triangulate_ID.size() << endl;

/////////////////////////////////////////   22222222222222222222222222222222   ///////////////////////////////////////////////////////////

//                 // Storage pts
//                 current_image.pts.clear();
//                 for(int i = 0; i < current_track_point_for_triangulate.size(); i++) current_image.pts.push_back(current_track_point_for_triangulate[i]); 

                
//                 // Determinate same map_point and Storage ID - landmark
//                 std::cout << " map_storage " << endl;
// std::cout << "map storage map size : " << map_storage.world_xyz.size() << endl;               
// std::cout << "track map point size : " << SolvePnP_tracking_map.size() << endl;
// std::cout << "new map_point size  : "  << map_point.size() << endl;
                
//                 int map_ID = map_storage.world_xyz.size();
//                 current_image.pts_id.clear();
//                 for(int j = 0; j < map_point.size(); j++)
//                 {

//                     for(int i = 0; i < SolvePnP_tracking_map.size(); i++)
//                     {
//                         double map_point_difference_x = cv::abs(SolvePnP_tracking_map[i].x - map_point[j].x);
//                         double map_point_difference_y = cv::abs(SolvePnP_tracking_map[i].y - map_point[j].y);
//                         double map_point_difference_z = cv::abs(SolvePnP_tracking_map[i].z - map_point[j].z);
//                         double map_point_difference = map_point_difference_x + map_point_difference_y + map_point_difference_z;


//                         // same point

//                         if (map_point_difference_x <2.0 and map_point_difference_y < 2.0 and map_point_difference_z < 2.0 and map_point_difference < 5.0)
//                         {
// // std::cout << "find same point         " << j << " id  : " << i << "    " << itr->first <<endl;
//                             current_image.pts_id.push_back(SolvePnP_tracking_map_ID[i]);
//                             SolvePnP_tracking_map.erase(SolvePnP_tracking_map.begin() + i);
//                             SolvePnP_tracking_map_ID.erase(SolvePnP_tracking_map_ID.begin() + i);
//                             break;
                            
//                         }  
//                     }
                            
                                            
//                         // different point
//                         if(current_image.pts_id.size() > j) continue;
//                         std::map<int, cv::Point3d> xyz_;
//                         xyz_.insert(std::pair<int, cv::Point3d>(map_ID, map_point[j]));
//                         map_storage.world_xyz.push_back(xyz_);
// // std::cout << "different point map id   "<< j << "   " << map_ID << endl;
//                         current_image.pts_id.push_back(map_ID);
                        
//                         map_ID++;     
//                 }     
// std::cout << " map_ID " <<  map_ID << endl;

                
//                 SolvePnP_tracking_map.clear();
//                 SolvePnP_tracking_map_ID.clear();
//                 for(int i = 0; i < map_point.size(); i++) 
//                 {
//                     SolvePnP_tracking_map.push_back(map_point[i]);
//                     SolvePnP_tracking_map_ID.push_back(current_image.pts_id[i]);
//                 }       



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////                
                
                


                // Storage current image 
                Frame copy_current_image = current_image;
                frame_storage.push_back(copy_current_image);
std::cout << " current frame_storage size : " << frame_storage.size() << endl;             
                
                keyframe_num += 1;
                // storage id - keyframe
                map_storage.keyframe.insert(std::pair<int, Frame>(keyframe_num, frame_storage[times]));
            
cv::imshow(" keyframe image ", map_storage.keyframe[keyframe_num].frame);
std::cout << "@@@@@@@@@ New keyframe was made @@@@@@@@@" << endl << " Keyframe num is  : " << keyframe_num << " Frame num is : " << times << endl;   

                for(int i = 0; i < keyframe_num + 1; i ++) 
                {
                    std::cout << "keyframe num  :  " << i << "  pts size :  " << map_storage.keyframe[i].pts.size() << "    " << "pts id size  :   " << map_storage.keyframe[i].pts_id.size() << " cam pose  :  " << cv::Mat(map_storage.keyframe[i].cam_pose) << endl;
                }       
                // ceres::Problem initilize_ba;

                // for ( int i = 0; i < map_storage.keyframe[keyframe_num].pts.size(); i++)
                // {
                //     ceres::CostFunction* cost_func = ReprojectionError::create(map_storage.keyframe[keyframe_num].pts[i], f, cv::Point2d(c.x, c.y));
                //     double* camera = (double*)(&map_storage.keyframe[keyframe_num].cam_pose);
                //     int id_ = map_storage.keyframe[keyframe_num].pts_id[i];
                //     double* X_ = (double*)(&(map_storage.world_xyz[id_][id_]));
                //     initilize_ba.AddResidualBlock(cost_func, NULL, camera, X_); 
                // }            
                            
                // // ceres option       
                // ceres::Solver::Options options;
                // options.linear_solver_type = ceres::ITERATIVE_SCHUR;
                // options.num_threads = 8;
                // options.minimizer_progress_to_stdout = true;
                // ceres::Solver::Summary summary;

                // std::cout << " camera cam pose before intialize ba " << endl <<vec6d_to_homogenous_campose(map_storage.keyframe[keyframe_num].cam_pose) << endl;

                // ceres::Solve(options, &initilize_ba, &summary);

                // std::cout << " camera cam pose after intialize ba " << endl <<vec6d_to_homogenous_campose(map_storage.keyframe[keyframe_num].cam_pose) << endl;
                // Bundle Adjustment between keyframes
                // Define CostFunction
                if(keyframe_num > 6)
                {
                    ceres::Problem keyframe_ba;

                    for(int j = optimize_intial_keyframe_num; j < keyframe_num; j++)
                    {
                        for ( int i = 0; i < map_storage.keyframe[j].pts.size(); i++)
                        {
                                ceres::CostFunction* keyframe_cost_func = ReprojectionError::create(map_storage.keyframe[j].pts[i], f, cv::Point2d(c.x, c.y));
                                // std::cout << j <<  endl;
                                // std::cout << map_storage.keyframe[j].pts.size() <<  "   " << i << endl;

                                double* camera = (double*)(&map_storage.keyframe[j].cam_pose);
                                int id_ = map_storage.keyframe[j].pts_id[i];
                                double* X_ = (double*)(&(map_storage.world_xyz[id_][id_]));
                                keyframe_ba.AddResidualBlock(keyframe_cost_func, NULL, camera, X_); 
                                // std::cout << "@@123" << endl;

                        }            
                    }

std::cout << "1234" << endl;
                    for(int j = fix_intial_keyfram_num; j < optimize_intial_keyframe_num; j++)
                    {
                        for ( int i = 0; i < map_storage.keyframe[j].pts.size(); i++)
                        {
                                ceres::CostFunction* fix_keyframe_cost_func = map_point_only_ReprojectionError::create(map_storage.keyframe[j].pts[i], map_storage.keyframe[j].cam_pose, f, cv::Point2d(c.x, c.y));
                                
                                
                                int id_ = map_storage.keyframe[j].pts_id[i];
                                double* X_ = (double*)(&(map_storage.world_xyz[id_][id_]));
                                keyframe_ba.AddResidualBlock(fix_keyframe_cost_func, NULL, X_); 
                                
                        }            
                    }                    
std::cout << "1235" << endl;
                    // ceres option       
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
                    options.num_threads = 12;
                    options.minimizer_progress_to_stdout = false;
                    ceres::Solver::Summary summary;

                    // Camera pose and map_point before BA
                    std::cout <<" camera pose before keyframe BA " << endl;
                    for(int j = optimize_intial_keyframe_num; j < keyframe_num ; j++) std::cout << " keyframe num : " << j << endl << vec6d_to_homogenous_campose(map_storage.keyframe[j].cam_pose) << endl;
                    // std::cout << " first map point value before BA " << endl;
                    // std::cout << map_storage.world_xyz[0][0] << endl;
                    
                    // solve
                    ceres::Solve(options, &keyframe_ba, &summary);                
                    // std::cout << summary.FullReport() << endl;
                    
                    // Camera pose and map_point after BA
                    std::cout << endl << " camera pose after keyframe BA " << endl;
                    for(int j = optimize_intial_keyframe_num; j < keyframe_num ; j++) std::cout << " keyframe num : " << j << endl << vec6d_to_homogenous_campose(map_storage.keyframe[j].cam_pose) << endl;                
                    // std::cout << " first map point value after BA " << endl;
                    // std::cout << map_storage.world_xyz[0][0] << endl; 
                
                }
                
                // new feature to track
                std::cout << " new feature " << endl;
                current_track_point_for_triangulate.clear();
                cv::goodFeaturesToTrack(current_image.frame, current_track_point_for_triangulate, 1500, 0.01, 10);
std::cout << " new feature num  : " << current_track_point_for_triangulate.size() << endl; 
                track_entire_num = current_track_point_for_triangulate.size();
                    
                keyframe_track_point.clear();
                for(int i = 0; i < current_track_point_for_triangulate.size(); i++) keyframe_track_point.push_back(current_track_point_for_triangulate[i]);            
                
                previous_track_point_for_triangulate_ID.clear();
                std::vector<cv::Point2f> clone_pts(current_image.pts);
                std::vector<int> clone_pts_id(current_image.pts_id);
                for(int j =0; j < current_track_point_for_triangulate.size(); j++)
                {
                    bool found_pts2d = false;
                    for(int i = 0; i < clone_pts.size(); i++)
                    {
                        double difference_point2d_x = cv::abs(clone_pts[i].x - current_track_point_for_triangulate[j].x);
                        double difference_point2d_y = cv::abs(clone_pts[i].y - current_track_point_for_triangulate[j].y);

                        if(difference_point2d_x < 3.0 and difference_point2d_y < 3.0)
                        {
                            found_pts2d = true;
                            previous_track_point_for_triangulate_ID.push_back(clone_pts_id[i]);
                            clone_pts.erase(clone_pts.begin() + i);
                            clone_pts_id.erase(clone_pts_id.begin() + i);
                            break;
                        }
                    }
                    
                    if(found_pts2d) found_pts2d= false;
                    else
                    {
                        previous_track_point_for_triangulate_ID.push_back(feature_max_ID++);
                    }

                }
            }

            previous_track_point_for_triangulate.clear();
            for(int i = 0; i < current_track_point_for_triangulate.size(); i++) previous_track_point_for_triangulate.push_back(current_track_point_for_triangulate[i]);

            // Reprojection 3D to 2D current image
            std::cout << "reprojection 3D to 2D at SolvePnP stage " << endl;
            if(times > initialize_frame_num) 
            {
                cv::Mat project_mat = Mat(map_point).clone();
                project_mat.convertTo(project_mat, CV_32F);
                cv::projectPoints(project_mat, World_R, World_t, K, cv::noArray(), projectionpoints); 
            }
            

            
            


            



    
        }


        




        


        



        


 






////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////               SHOW IMAGE                ////////////////////////////////////////////////////////////      
        std::cout << "show image" << endl;
        // Show image and Matching points and reprojection points
        cv::Mat show_image = current_image.frame;
        if (show_image.channels() < 3) cv::cvtColor(show_image, show_image, cv::COLOR_GRAY2RGB);
        for(int a = 0; a < current_image.pts.size(); a++) circle(show_image, current_image.pts[a], 3, Scalar(0,255,0), 1);
        if(times >= initialize_frame_num) for(int a = 0; a < map_point.size(); a++) circle(show_image, projectionpoints[a], 3, Scalar(0,0,255), 1);
        
        // Put text to image about camera_pose
        cv::Mat campose_vec6d_to_mat_for_text = vec6d_to_homogenous_campose(current_image.cam_pose);
        cv::String info = cv::format(" XYZ: [%.3lf, %.3lf, %.3lf]", campose_vec6d_to_mat_for_text.at<double>(0, 3), campose_vec6d_to_mat_for_text.at<double>(1, 3), campose_vec6d_to_mat_for_text.at<double>(2, 3));
        cv::putText(show_image, info, cv::Point(5, 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Vec3b(255, 0, 0));
        cv::imshow("image", show_image);





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////            VISUALIZE         ////////////////////////////////////////////////////////////////
        
        // Show gt trajectory (red line)
        GLfloat inputbuffer[12];
        for(int j=0;j<12;j++) GT_DATA >> inputbuffer[j];
        GLfloat x_gt(inputbuffer[3]), y_gt(inputbuffer[7]), z_gt(inputbuffer[11]);
        show_trajectory(x_gt, y_gt, z_gt, 1.0, 0.0, 0.0, 3.0);
        show_trajectory_right_mini(x_gt, y_gt, z_gt, 1.0, 0.0, 0.0, 3.0);
        show_trajectory_left_mini(x_gt, y_gt, z_gt, 1.0, 0.0, 0.0, 3.0);


        // show gt 
        cv::Mat_<double> gt_pose(3, 4);
        gt_pose << inputbuffer[0], inputbuffer[1], inputbuffer[2], inputbuffer[3], inputbuffer[4], inputbuffer[5], inputbuffer[6], inputbuffer[7], inputbuffer[8], inputbuffer[9], inputbuffer[10], inputbuffer[11];
        std::cout << " GT pose  : "  << endl << gt_pose << endl;        

        // Show camera estimate pose trajectory_motion_only_ba ( green line )
        cv::Mat campose_vec6d_to_mat_for_visualize = vec6d_to_homogenous_campose(current_image.cam_pose);
        GLdouble _x_cam_pose(campose_vec6d_to_mat_for_visualize.at<double>(0, 3)), _y_cam_pose(campose_vec6d_to_mat_for_visualize.at<double>(1, 3)), _z_cam_pose(campose_vec6d_to_mat_for_visualize.at<double>(2, 3));         
        show_trajectory(_x_cam_pose, _y_cam_pose, _z_cam_pose, 0.0, 1.0, 0.0, 3.0);
        show_trajectory_right_mini(_x_cam_pose, _y_cam_pose, _z_cam_pose, 0.0, 1.0, 0.0, 3.0);

        // show camera estimate pose trajectory keyframe after ba ( black triangle )
        if (new_keyframe_selection)
        { 
            if(keyframe_num > 6)
            {    
                cv::Mat campose_vec6d_to_mat_for_visualize = vec6d_to_homogenous_campose(map_storage.keyframe[optimize_intial_keyframe_num - 1].cam_pose);
                GLdouble _x_cam_pose(campose_vec6d_to_mat_for_visualize.at<double>(0, 3)), _y_cam_pose(campose_vec6d_to_mat_for_visualize.at<double>(1, 3)), _z_cam_pose(campose_vec6d_to_mat_for_visualize.at<double>(2, 3));         
                show_trajectory_left_mini(_x_cam_pose, _y_cam_pose, _z_cam_pose, 0.0, 0.0, 0.0, 3.0);
                cv::Mat rb_t = homogenous_campose_for_keyframe_visualize(campose_vec6d_to_mat_for_visualize, 6.0); // size
                show_trajectory_keyframe(rb_t, 0.0, 0.0, 0.0, 1.0);
                if( optimize_intial_keyframe_num - fix_intial_keyfram_num == 8) fix_intial_keyfram_num++;
                optimize_intial_keyframe_num++;
            }
        }
        // Show 3d keyframe map point ( blue dot )
//         if (times == initialize_frame_num)
//         {
//             for( int i = show_map_point_parms ; i < map_storage.world_xyz.size(); i++)
//             {
//                 GLdouble X_map(map_storage.world_xyz[i][i].x), Y_map(map_storage.world_xyz[i][i].y), Z_map(map_storage.world_xyz[i][i].z);
//                 show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 1.0, 0.1);
//                 show_map_point_parms = map_storage.world_xyz.size();
//             }  
//             show_map_point_parms = map_storage.world_xyz.size();
// std::cout << " map_storage size is : " << map_storage.world_xyz.size() << endl;      
//         }
        if (new_keyframe_selection)
        {   
//             for( int i = show_map_point_parms; i < map_storage.world_xyz.size(); i++)
//             {
//                 GLdouble X_map(map_storage.world_xyz[i][i].x), Y_map(map_storage.world_xyz[i][i].y), Z_map(map_storage.world_xyz[i][i].z);
//                 show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 1.0, 0.1);
                
//             }         
            new_keyframe_selection = false;
//             show_map_point_parms = map_storage.world_xyz.size();
// std::cout << " map_storage size is : " << map_storage.world_xyz.size() << endl;
        }
        
        if (keyframe_num % 50 == 0)
        {

        
             
                for( int i = show_fix_map_point_parms; i < map_storage.world_xyz.size(); i++)
                {
                    GLdouble X_map(map_storage.world_xyz[i][i].x), Y_map(map_storage.world_xyz[i][i].y), Z_map(map_storage.world_xyz[i][i].z);
                    show_trajectory(X_map, Y_map, Z_map, 0.0, 1.0, 1.0, 0.1);
                    
                }         
                show_fix_map_point_parms = map_storage.world_xyz.size();
                // new_keyframe_selection = false;
                // show_map_point_parms = map_storage.world_xyz.size();
    std::cout << " map_storage size is : " << map_storage.world_xyz.size() << endl;
        
        }
        glFlush();

        

        previous_image = current_image;

        ++times;
        
        // double fps = video.set(CAP_PROP_FPS, 30);
        double delay = cvRound(1000/30);
        // cv::waitKey(delay);
        // if(times > 540); 
        // cv::waitKey();
        cv::waitKey(1);
    }
    cv::waitKey();
    return 0;
}