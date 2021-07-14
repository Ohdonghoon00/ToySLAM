#include "DBoW2.h"
#include "assembly.h"
#include "math.h"


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
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include "glog/logging.h"

using namespace std;
using namespace cv;
using namespace DBoW2;
// using ceres::CauchyLoss;



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
 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////        Parameter       /////////////////////////////////////////////////////////////////////          
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////    
    
    int max_keypoint = 1500;        // max feature detection number
    int initialize_frame_num = 6;   // number of 5 point algorithm ( Essential Matrix ) 
    float SolvePnP_reprojection_error = 3.0F;

    // Keyframe Selection
    int KS_track_overlap_ratio= 65;
    int KS_inliers_num = 200;
    double KS_yaw_difference = 100; // 0.055
    
    // Local BA 
    int fix_keyframe_num = 5;
    int active_keyframe_num = 5;

    // show image delay
    bool show_image_one_by_one = false;
    bool realtime_delay = false;
    bool fast_slam = true;
    bool want_frame_num = false;
    int go_to_frame_num = 540;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    double delay = cvRound(1000/30);  // real-time delay
    int track_entire_num;    
    int track_overlap_ratio;
    int feature_max_ID;
    bool new_keyframe_selection = false;
    bool caculate_triangulation = true;
    int show_map_point_parms = 0;
    int show_trajectory_parms = 0;
    int show_fix_map_point_parms = 0;
    int fix_keyframe_parms = fix_keyframe_num;
    int times = 1;
    int keyframe_num = 0;
    int loop_detect_frame_id = 0;
    bool loop_detect = false;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////    
    
    cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat P1;
    cv::Mat cam_pose_initialize = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat cam_pose = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat Iden = Mat::eye(4, 4, CV_64F);
    std::vector<Point3d> map_point;
    vector< vector<cv::Mat > > features;

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


    // DBoW2
    // OrbVocabulary voc("small_voc.yml.gz");
    std::cout << "load voc" << endl;
    OrbVocabulary voc("../0630_KITTI00-22_10_4_voc.yml.gz");
    std::cout << "copy voc to db" << endl;
    OrbDatabase db(voc, false, 0); // false = do not use direct index
  

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
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
        
        if ( times < initialize_frame_num + 1)
        {
            
            // Find feature in previous_image 
            
            std::cout << " feature track " << endl;
            if(times == 1)
            {
                
                cv::goodFeaturesToTrack(previous_image.frame, previous_image.pts, max_keypoint, 0.01, 10);
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
                cv::solvePnPRansac(map_point, current_image.pts, K, cv::noArray(), rot, tran, false, 100, SolvePnP_reprojection_error, 0.99, inliers );
                inlier_storage.push_back(inliers);
std::cout << keyframe_num <<"  keyframe inlier storage rate : " << 100 * inliers.rows / map_point.size() << endl;

                // remove_SolvePnP_oulier(map_point, current_image.pts, inliers);
std::cout << " map_point size after remove solvepnp outlier : " <<  map_point.size() << endl;
                

                // ORB descriptor for loop detection
                cv::Ptr<cv::ORB> orb = cv::ORB::create();
                cout << "Extracting ORB features..." << times << endl;

                vector<cv::KeyPoint> keypoints;
                cv::Mat descriptors;
                cv::Mat mask;
                features.clear();
                orb->detectAndCompute(current_image.frame, mask, keypoints, descriptors);
                features.push_back(vector<cv::Mat >());
                changeStructure(descriptors, features.back());


                db.add(features[0]);

                
                // pose graph

                // create linear solver
                std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linear_solver = 
                g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
               
                    
                // create block solver                            
                std::unique_ptr<g2o::BlockSolverX> block_solver =
                g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver));

                    
                

                


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
                    // std::map<int, cv::Point3d> xyz_;
                    map_storage.world_xyz.insert(std::pair<int, cv::Point3d>(i, map_point[i]));
                    // map_storage.world_xyz.push_back(xyz_);

                    // SolvePnP_tracking_map.push_back(map_point[i]);
                    // SolvePnP_tracking_map_ID.push_back(i);
                }       

std::cout << " Landmark's num : " << map_storage.world_xyz.size() << endl;
std::cout << " same as landmark num :::: keyframe 2d point size : " << map_storage.keyframe[keyframe_num].pts.size() <<endl;

                // new feature to track
                std::cout << " new feature " << endl;
                current_track_point_for_triangulate.clear();
                cv::goodFeaturesToTrack(current_image.frame, current_track_point_for_triangulate, max_keypoint, 0.01, 10);
std::cout << " new feature num  : " << current_track_point_for_triangulate.size() << endl; 
                track_entire_num = current_track_point_for_triangulate.size();
                
                keyframe_track_point.clear();
                for(int i = 0; i < current_track_point_for_triangulate.size(); i++) keyframe_track_point.push_back(current_track_point_for_triangulate[i]);

                previous_track_point_for_triangulate.clear();
                for(int i = 0; i < current_track_point_for_triangulate.size(); i++) previous_track_point_for_triangulate.push_back(current_track_point_for_triangulate[i]);

                std::vector<cv::Point2f> clone_pts(current_image.pts);
                std::vector<int> clone_pts_id(current_image.pts_id);
                int same_point_num(0), different_point_num(0);
std::cout << " remain feature num : " << current_image.pts.size() << endl;

                for(int j = 0; j < current_track_point_for_triangulate.size(); j++)
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
                            same_point_num++;
                            break;
                        }
                    }
                    
                    if(found_pts2d) found_pts2d= false;
                    else
                    {
                        different_point_num++;
                        feature_max_ID++;
                        previous_track_point_for_triangulate_ID.push_back(feature_max_ID);
                    }

                }
std::cout << "total new feature : " << current_track_point_for_triangulate.size() << endl;
std::cout << " same point :  " << same_point_num << " different point num  : " << different_point_num << endl;



                // BA initialize
                // Define CostFunction
                // ceres::Problem initilize_ba;

                // for ( int i = 0; i < map_storage.keyframe[0].pts.size(); i++)
                // {
                //     ceres::CostFunction* cost_func = ReprojectionError::create(map_storage.keyframe[0].pts[i], f, cv::Point2d(c.x, c.y));
                //     double* camera = (double*)(&map_storage.keyframe[0].cam_pose);
                //     double* X_ = (double*)(&(map_storage.world_xyz[i]));
                //     initilize_ba.AddResidualBlock(cost_func, NULL, camera, X_); 
                // }            
                            
                // // ceres option       
                // ceres::Solver::Options options;
                // options.linear_solver_type = ceres::ITERATIVE_SCHUR;
                // options.num_threads = 8;
                // options.minimizer_progress_to_stdout = true;
                // ceres::Solver::Summary summary;

                // std::cout << " camera cam pose before intialize ba " << endl <<vec6d_to_homogenous_campose(map_storage.keyframe[0].cam_pose) << endl;

                // ceres::Solve(options, &initilize_ba, &summary);

                // std::cout << " camera cam pose after intialize ba " << endl <<vec6d_to_homogenous_campose(map_storage.keyframe[0].cam_pose) << endl;
            
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
////////////////////////////      SOLVEPNP + TRIANGULATION             //////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
        
        if (times > initialize_frame_num)
        {
            // Matching prev_image and current_image using optical flow
            std::cout << "optical flow" << endl;
            
            float before_track_feature_num = previous_image.pts.size();
std::cout << "before_track_feature_num  : " << before_track_feature_num << endl;
            
            track_opticalflow_and_remove_err_for_SolvePnP(previous_image.frame, current_image.frame, previous_image.pts, current_image.pts, previous_image.pts_id, map_point);

std::cout << "after track_feature num  : " << current_image.pts.size() << endl;  
            
                
            // Storage corresponding Id
            current_image.pts_id.clear();
            for(int i = 0; i < current_image.pts.size(); i++) current_image.pts_id.push_back(previous_image.pts_id[i]);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// track for triangulate///////////////////////////////////////////////////////////////////////////////////////////
            
            std::cout << " feature track " << endl;
std::cout << previous_track_point_for_triangulate.size() << "   " << current_track_point_for_triangulate.size() << "    " << previous_track_point_for_triangulate_ID.size() << "   " << keyframe_track_point.size() << endl;
            track_opticalflow_and_remove_err_for_triangulate_(previous_image.frame, current_image.frame, previous_track_point_for_triangulate, current_track_point_for_triangulate, previous_track_point_for_triangulate_ID, keyframe_track_point);
std::cout << previous_track_point_for_triangulate.size() << "   " << current_track_point_for_triangulate.size() << "    " << previous_track_point_for_triangulate_ID.size() << "   " << keyframe_track_point.size() << endl;            
                

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            // Calculate World R, t using SolvePnP
            std::cout << "SolvePnP" << endl;
std::cout << "SolvePnP map_point num : " << "   " << map_point.size() << endl;
std::cout << "SolvePnP current_image_pts num : " << "   " << current_image.pts.size() << endl;
            cv::Mat inliers;
            map_point.clear();
            for(int i = 0; i < current_image.pts_id.size(); i++) map_point.push_back(map_storage.world_xyz[current_image.pts_id[i]]);
            std::cout << map_point.size() << "  " << current_image.pts.size() << endl;
            cv::solvePnPRansac(map_point, current_image.pts, K, cv::noArray(), World_R, World_t, false, 100, SolvePnP_reprojection_error, 0.99, inliers );
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

            
            // Motion only BA
//             std::vector<cv::Point3d> map_point_for_motion_BA;
//             std::vector<cv::Point2f> current_image_pts_for_motion_BA;
  
//             // for(int i = 0; i < map_point.size(); i++) map_point_for_motion_BA.push_back(map_point[i]);
//             // for(int i = 0; i < current_image.pts.size(); i++) current_image_pts_for_motion_BA.push_back(current_image.pts[i]);
//             for(int i = 0; i < inliers.rows; i++)
//             {
//                 map_point_for_motion_BA.push_back(map_point[inliers.at<int>(i, 0)]);
//                 current_image_pts_for_motion_BA.push_back(current_image.pts[inliers.at<int>(i, 0)]);
                
//             }
// std::cout << " map point size for motion only BA : " << map_point_for_motion_BA.size() << endl;

//             ceres::Problem motion_only_ba;
            
//             for ( int i = 0; i < current_image_pts_for_motion_BA.size(); i++)
//             {
//                 ceres::CostFunction* motion_only_cost_func = motion_only_ReprojectionError::create(current_image_pts_for_motion_BA[i], map_point_for_motion_BA[i], f, cv::Point2d(c.x, c.y));
//                 double* camera_ = (double*)(&current_image.cam_pose);
        
//                 motion_only_ba.AddResidualBlock(motion_only_cost_func, NULL, camera_); 
//             }            
            
//             // ceres option       
//             ceres::Solver::Options options;
//             options.linear_solver_type = ceres::ITERATIVE_SCHUR;
//             options.num_threads = 12;
//             options.minimizer_progress_to_stdout = false;
//             ceres::Solver::Summary summary;
                            
//             // solve
//             ceres::Solve(options, &motion_only_ba, &summary);

//             std::cout <<" ( after motion only BA )current camera pose  : " << endl;
//             std::cout << vec6d_to_homogenous_campose(current_image.cam_pose) << endl;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////   Determine Keyframe or not    ////////////////////////////////////////////////////////////////////////////////////
            
            // track_overlap_ratio 
            track_overlap_ratio = 100 * current_track_point_for_triangulate.size() / track_entire_num;
std::cout << "  track inlier ratio : " << track_overlap_ratio << endl;

            // Rotation value 
std::cout << "previous vec6d value : " << previous_image.cam_pose << endl;
std::cout << "current vec6d value  : " << current_image.cam_pose << endl;
            double rotation_difference = cv::abs(previous_image.cam_pose[1] - current_image.cam_pose[1]);
std::cout << " Rotation difference : " << rotation_difference << endl;           


            // Determinate select keyframe or not
            if (track_overlap_ratio < KS_track_overlap_ratio or inliers.rows < KS_inliers_num or rotation_difference > KS_yaw_difference ) new_keyframe_selection = true;
            
            if (new_keyframe_selection == false)
            {
                // Storage current image 
                Frame copy_current_image = current_image;
                frame_storage.push_back(copy_current_image);
            }            
                

//             cv::Mat pose_difference = cv::abs(vec6d_to_homogenous_campose(current_image.cam_pose)) - cv::abs(vec6d_to_homogenous_campose(previous_image.cam_pose));
// std::cout << " pose difference   : " << endl << pose_difference << endl;
//             double translation_difference = std::abs(pose_difference.at<double>(0, 3)) + std::abs(pose_difference.at<double>(1, 3)) + std::abs(pose_difference.at<double>(2, 3));
// std::cout << "translation difference : " << translation_difference << endl;            
            
            
            
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////            
            
            if (new_keyframe_selection)
            {


                // Determinate doing triangulation or not 
                // if (translation_difference < 0.06 and (100 * previous_image.pts.size() / before_track_feature_num) > 96 ) caculate_triangulation = false;
                // else caculate_triangulation = true;
                if (caculate_triangulation)
                {
                
                  // Caculate projection matrix for triangulation
                    std::cout << "  projection matrix    " << endl;
                    std::cout << "P0 campose : " << endl << vec6d_to_homogenous_campose(map_storage.keyframe[keyframe_num].cam_pose) << endl;
                    std::cout << "P1 campose : " << endl << vec6d_to_homogenous_campose(current_image.cam_pose) << endl;
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
                    if(map_point.size() < 10) cv::waitKey();
                }            
                
                // storage landmark id
                std::vector<cv::Point2f> clone_current_track_point_for_triangulate(current_track_point_for_triangulate);
                std::vector<cv::Point3d> clone_map_point(map_point);
                std::vector<int> clone_previous_track_point_for_triangulate_ID(previous_track_point_for_triangulate_ID);
                
                    int indexCorrect = 0;
                for(int i = 0; i < clone_current_track_point_for_triangulate.size(); i++)
                {
                    if(map_storage.world_xyz.find(clone_previous_track_point_for_triangulate_ID[i]) == map_storage.world_xyz.end())
                    {

                            map_storage.world_xyz.insert(std::pair<int, cv::Point3d>(clone_previous_track_point_for_triangulate_ID[i], clone_map_point[i]));


                    }
                    else 
                    {

                    double diff_3d_x = std::abs(clone_map_point[i].x - map_storage.world_xyz[clone_previous_track_point_for_triangulate_ID[i]].x);
                    double diff_3d_y = std::abs(clone_map_point[i].y - map_storage.world_xyz[clone_previous_track_point_for_triangulate_ID[i]].y);
                    double diff_3d_z = std::abs(clone_map_point[i].z - map_storage.world_xyz[clone_previous_track_point_for_triangulate_ID[i]].z);
                    
                        if(diff_3d_x < 3.0 and diff_3d_y < 3.0 and diff_3d_z < 3.0)
                        {
                            map_point[i - indexCorrect] = map_storage.world_xyz[clone_previous_track_point_for_triangulate_ID[i]];
                        }
                        else
                        {
                            current_track_point_for_triangulate.erase(current_track_point_for_triangulate.begin() + i - indexCorrect );
                            map_point.erase(map_point.begin() + i - indexCorrect );
                            previous_track_point_for_triangulate_ID.erase(previous_track_point_for_triangulate_ID.begin() + i - indexCorrect);
                            indexCorrect++;

                        }
                    }
                }


                // Storage SolvePnP inlier index
                cv::Mat rot, tran, inliers_;
std::cout << map_point.size() << "      " << current_track_point_for_triangulate.size() << endl;
                map_point.clear();
                for(int i = 0; i < previous_track_point_for_triangulate_ID.size(); i++) map_point.push_back(map_storage.world_xyz[previous_track_point_for_triangulate_ID[i]]);
                cv::solvePnPRansac(map_point, current_track_point_for_triangulate, K, cv::noArray(), rot, tran, false, 100, SolvePnP_reprojection_error, 0.99, inliers_ );
                inlier_storage.push_back(inliers_);
std::cout << keyframe_num + 1 << " keyframe inlier storage rate : " << 100 * inliers_.rows / map_point.size() << endl;                

                
                // Storage pts
                current_image.pts.clear();
                current_image.pts_id.clear();
                for(int i = 0; i < current_track_point_for_triangulate.size(); i++) 
                {
                    current_image.pts.push_back(current_track_point_for_triangulate[i]);                
                    current_image.pts_id.push_back(previous_track_point_for_triangulate_ID[i]);
                }

std::cout << " size print " << current_track_point_for_triangulate.size() << "    " << previous_track_point_for_triangulate_ID.size() << endl;

      
                // Storage current image 
                Frame copy_current_image = current_image;
                frame_storage.push_back(copy_current_image);
std::cout << " current frame_storage size : " << frame_storage.size() << endl;             
                
                keyframe_num += 1;
                // storage id - keyframe
                map_storage.keyframe.insert(std::pair<int, Frame>(keyframe_num, frame_storage[times]));
            
cv::imshow(" keyframe image ", map_storage.keyframe[keyframe_num].frame);
std::cout << "@@@@@@@@@ New keyframe was made @@@@@@@@@" << endl << " Keyframe num is  : " << keyframe_num << " Frame num is : " << times << endl;   




                
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////   Local  Bundle Adjustment    ////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//                 ceres::Problem keyframe_ba;
                int p1 = 0;
                if(keyframe_num - active_keyframe_num + 1 < 0) p1 = keyframe_num - active_keyframe_num + 1;
                else p1 = 0;
                
//                 for(int j = keyframe_num - active_keyframe_num + 1 - p1; j < keyframe_num + 1; j++)
//                 {
// std::cout << " active keyframe num  : " << j << endl;                        
//                     for ( int i = 0; i < inlier_storage[j].rows; i++)
//                     {
//                         ceres::CostFunction* keyframe_cost_func = ReprojectionError::create(map_storage.keyframe[j].pts[inlier_storage[j].at<int>(i, 0)], f, cv::Point2d(c.x, c.y));

//                         double* camera = (double*)(&map_storage.keyframe[j].cam_pose);
//                         int id_ = map_storage.keyframe[j].pts_id[inlier_storage[j].at<int>(i, 0)];
//                         double* X_ = (double*)(&(map_storage.world_xyz[id_]));
//                         keyframe_ba.AddResidualBlock(keyframe_cost_func, NULL, camera, X_); 

//                     }            
//                 }

                int p2 = 0;
                if(keyframe_num - active_keyframe_num + 1 - fix_keyframe_num < 0) p2 = keyframe_num - active_keyframe_num + 1 - fix_keyframe_num;
                else p2 = 0;

//                 for(int j = keyframe_num - active_keyframe_num + 1 - fix_keyframe_num - p2; j < keyframe_num - active_keyframe_num + 1; j++)
//                 {
// std::cout << " fixed keyframe num  : " << j << endl;                         
//                     for ( int i = 0; i < inlier_storage[j].rows; i++)
//                     {
//                         ceres::CostFunction* fix_keyframe_cost_func = map_point_only_ReprojectionError::create(map_storage.keyframe[j].pts[inlier_storage[j].at<int>(i, 0)], map_storage.keyframe[j].cam_pose, f, cv::Point2d(c.x, c.y));
//                         int id_ = map_storage.keyframe[j].pts_id[inlier_storage[j].at<int>(i, 0)];
//                         double* X_ = (double*)(&(map_storage.world_xyz[id_]));
//                         keyframe_ba.AddResidualBlock(fix_keyframe_cost_func, NULL, X_); 
                                
//                     }            
//                 }                    
                                
                                
//                 // ceres option       
//                 ceres::Solver::Options options;
//                 options.linear_solver_type = ceres::ITERATIVE_SCHUR;
//                 options.num_threads = 12;
//                 options.minimizer_progress_to_stdout = false;
//                 ceres::Solver::Summary summary;

//                 // Camera pose and map_point before BA
//                 std::cout <<" camera pose before keyframe BA " << endl;
//                 for(int j = keyframe_num - active_keyframe_num + 1 - p1; j < keyframe_num + 1; j++) std::cout << " keyframe num : " << j << endl << vec6d_to_homogenous_campose(map_storage.keyframe[j].cam_pose) << endl;
                    
//                 // solve
//                 ceres::Solve(options, &keyframe_ba, &summary);                
                    
//                 // Camera pose and map_point after BA
//                 std::cout << endl << " camera pose after keyframe BA " << endl;
//                 for(int j = keyframe_num - active_keyframe_num + 1 - p1; j < keyframe_num + 1; j++) std::cout << " keyframe num : " << j << endl << vec6d_to_homogenous_campose(map_storage.keyframe[j].cam_pose) << endl;                

                // visualize map point ( black dot )
                if(keyframe_num % 3 == 0)
                {
                    if ((keyframe_num - active_keyframe_num + 1 - fix_keyframe_num) >= 0)
                    {
                        int k = keyframe_num - active_keyframe_num + 1 - fix_keyframe_num - p2;
                        for ( int i = 0; i < inlier_storage[k].rows; i++)
                        {
                            int id_ = map_storage.keyframe[k].pts_id[inlier_storage[k].at<int>(i, 0)];
                            GLdouble X_map(map_storage.world_xyz[id_].x), Y_map(map_storage.world_xyz[id_].y), Z_map(map_storage.world_xyz[id_].z);
                            show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 0.0, 0.01);              
                        }
                    }
                }


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////                
  
                cv::Ptr<cv::ORB> orb = cv::ORB::create();
                cout << "Extracting ORB features..." << times << endl;

                vector<cv::KeyPoint> keypoints;
                cv::Mat descriptors;
                cv::Mat mask;
                features.clear();
                orb->detectAndCompute(current_image.frame, mask, keypoints, descriptors);
                features.push_back(vector<cv::Mat >());
                changeStructure(descriptors, features.back());

                // Map ajsdfk.tri();

                QueryResults ret;
                ret.clear();
                int max_entry_id;
                if(keyframe_num > 30) 
                {
                    db.query(features[0], ret, 20);
cout << "Searching for Image " << keyframe_num << ". " << ret << endl;
                                       

                    // loop detect condition

                    // delete nearest frame
                    int index_correction = 0;
                    int clone_ret_size = ret.size();
                    for(int i = 0; i < clone_ret_size; i++)
                    {
                    
                        if(ret[i - index_correction].Id > keyframe_num - 10)
                        { 
                            ret.erase(ret.begin() + i - index_correction);
                            index_correction++;
                        }
                    }   
cout << "erase nearest keyframe "  << ". " << ret[0] << endl;
                        
                    if(ret[0] > 0.2) loop_detect = true;
                    if(loop_detect)
                    {
                        loop_detect_frame_id = ret[0].Id;
                        show_loop_detect_line(map_storage, loop_detect_frame_id, keyframe_num, 0.0f, 1.0f, 0.0f, 1.3);
                        // cv::waitKey();
                        loop_detect = false;
                    }    
                }
                db.add(features[0]);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////                
                
                // new feature to track
                std::cout << " new feature " << endl;
                current_track_point_for_triangulate.clear();
                cv::goodFeaturesToTrack(current_image.frame, current_track_point_for_triangulate, max_keypoint, 0.01, 8);
std::cout << " new feature num  : " << current_track_point_for_triangulate.size() << endl; 
                track_entire_num = current_track_point_for_triangulate.size();
                    
                keyframe_track_point.clear();
                for(int i = 0; i < current_track_point_for_triangulate.size(); i++) keyframe_track_point.push_back(current_track_point_for_triangulate[i]);            
                
                previous_track_point_for_triangulate_ID.clear();
                std::vector<cv::Point2f> clone_pts(current_image.pts);
                std::vector<int> clone_pts_id(current_image.pts_id);
                int same_point_num(0), different_point_num(0);
std::cout << " remain feature num : " << current_image.pts.size() << endl;
                
                
                // feature corresponding 
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
                            same_point_num++;
                            break;
                        }
                    }
                    
                    if(found_pts2d) found_pts2d= false;
                    else
                    {
                        different_point_num++;
                        feature_max_ID++;
                        previous_track_point_for_triangulate_ID.push_back(feature_max_ID);
                    }

                }

std::cout << "total new feature : " << current_track_point_for_triangulate.size() << endl;
std::cout << " same point :  " << same_point_num << " different point num  : " << different_point_num << endl;

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
        // show_trajectory_right_mini(x_gt, y_gt, z_gt, 1.0, 0.0, 0.0, 3.0);
        // show_trajectory_left_mini(x_gt, y_gt, z_gt, 1.0, 0.0, 0.0, 3.0);


        // show gt 
        cv::Mat_<double> gt_pose(3, 4);
        gt_pose << inputbuffer[0], inputbuffer[1], inputbuffer[2], inputbuffer[3], inputbuffer[4], inputbuffer[5], inputbuffer[6], inputbuffer[7], inputbuffer[8], inputbuffer[9], inputbuffer[10], inputbuffer[11];
        std::cout << " GT pose  : "  << endl << gt_pose << endl;        

        // Show camera estimate pose trajectory_motion_only_ba ( green line )
        cv::Mat campose_vec6d_to_mat_for_visualize = vec6d_to_homogenous_campose(current_image.cam_pose);
        GLdouble _x_cam_pose(campose_vec6d_to_mat_for_visualize.at<double>(0, 3)), _y_cam_pose(campose_vec6d_to_mat_for_visualize.at<double>(1, 3)), _z_cam_pose(campose_vec6d_to_mat_for_visualize.at<double>(2, 3));         
        // show_trajectory(_x_cam_pose, _y_cam_pose, _z_cam_pose, 0.0, 1.0, 0.0, 3.0);
        // show_trajectory_right_mini(_x_cam_pose, _y_cam_pose, _z_cam_pose, 0.0, 1.0, 0.0, 3.0);



        // show camera estimate pose trajectory keyframe after ba ( blue triangle )
        if (new_keyframe_selection)
        { 
               
                cv::Mat campose_vec6d_to_mat_for_visualize = vec6d_to_homogenous_campose(map_storage.keyframe[keyframe_num - active_keyframe_num].cam_pose);
                GLdouble _x_cam_pose(campose_vec6d_to_mat_for_visualize.at<double>(0, 3)), _y_cam_pose(campose_vec6d_to_mat_for_visualize.at<double>(1, 3)), _z_cam_pose(campose_vec6d_to_mat_for_visualize.at<double>(2, 3));         
                // show_trajectory_left_mini(_x_cam_pose, _y_cam_pose, _z_cam_pose, 0.0, 0.0, 1.0, 3.0);
                cv::Mat rb_t = homogenous_campose_for_keyframe_visualize(campose_vec6d_to_mat_for_visualize, 5.0); // size
                show_trajectory_keyframe(rb_t, 0.0, 0.0, 1.0, 1.0);
                if(fix_keyframe_parms - fix_keyframe_num == fix_keyframe_num) fix_keyframe_parms++;
        }
    

        // Show 3d keyframe map point ( black dot )
        if (times == initialize_frame_num)
        {
            for( int i = 0 ; i < map_storage.keyframe[0].pts_id.size(); i++)
            {
                int show_map_id = map_storage.keyframe[0].pts_id[i];
                GLdouble X_map(map_storage.world_xyz[show_map_id].x), Y_map(map_storage.world_xyz[show_map_id].y), Z_map(map_storage.world_xyz[show_map_id].z);
                // show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 0.0, 0.01);
                show_map_point_parms = map_storage.world_xyz.size();
            }  
            show_map_point_parms = map_storage.world_xyz.size();
std::cout << " map_storage size is : " << map_storage.world_xyz.size() << endl;      
        }

        
        if (new_keyframe_selection)
        {   
            if(keyframe_num % 3 == 0)
            {
            for( int i = 0 ; i < map_storage.keyframe[keyframe_num].pts.size(); i++)
            // {
                // for (std::map<int, cv::Point3d>::iterator itr = map_storage.world_xyz.begin(); itr != map_storage.world_xyz.end(); ++itr) 
                {
                //    itr->second
                                
                
                    int show_map_id = map_storage.keyframe[keyframe_num].pts_id[i];
                    GLdouble X_map(map_storage.world_xyz[show_map_id].x), Y_map(map_storage.world_xyz[show_map_id].y), Z_map(map_storage.world_xyz[show_map_id].z);
                    // show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 0.0, 0.01);
                }
            // }
            }         
            new_keyframe_selection = false;
            show_map_point_parms = map_storage.world_xyz.size();
std::cout << " map_storage size is : " << map_storage.world_xyz.size() << endl;
        }
//         if (new_keyframe_selection)
//         {   
//             // for( int i = 0 ; i < map_storage.world_xyz.size(); i++)
//             // {
//                 for (std::map<int, cv::Point3d>::iterator itr = map_storage.world_xyz.begin(); itr != map_storage.world_xyz.end(); ++itr) 
//                 {
//                 //    itr->second
                                
                
//                     // int show_map_id = map_storage.keyframe[keyframe_num].pts_id[i];
//                     GLdouble X_map(itr->second.x), Y_map(itr->second.y), Z_map(itr->second.z);
//                     show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 1.0, 0.1);
//                 }
//             // }         
//             new_keyframe_selection = false;
//             show_map_point_parms = map_storage.world_xyz.size();
// std::cout << " map_storage size is : " << map_storage.world_xyz.size() << endl;
//         }

        glFlush();

        

        previous_image = current_image;

        ++times;
        
        // double fps = video.set(CAP_PROP_FPS, 30);
        
        if(realtime_delay) cv::waitKey(delay);
        if(fast_slam) cv::waitKey(1);
        if(show_image_one_by_one) cv::waitKey();
        if(want_frame_num) if(times > go_to_frame_num) cv::waitKey(); 
    }
std::cout << " Finish SLAM " << endl;    
    cv::waitKey();
    return 0;
}
