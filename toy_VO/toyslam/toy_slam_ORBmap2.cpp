#include "assembly.h"



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

#include "ceres/ceres.h"
#include "gurobi_c++.h"


using namespace std;
using namespace cv;
using namespace DBoW2;
using namespace g2o;
using ceres::CauchyLoss;
using ceres::HuberLoss;



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
    int initialize_frame_num = 5;   // number of 5 point algorithm ( Essential Matrix ) 
    float SolvePnP_reprojection_error = 3.0F;
    int SolevePnPFlags = 0;

    // Keyframe Selection
    // int KS_track_overlap_ratio= 65;
    // int KS_inliers_num = 300;
    // double KS_yaw_difference = 100; // 0.055
    
    // Local BA 
    int fix_keyframe_num = 3;
    int active_keyframe_num = 3;

    // show image delay
    bool show_image_one_by_one = false;
    bool realtime_delay = false;
    bool fast_slam = true;
    bool want_frame_num = false;
    int go_to_frame_num = 540;
    int made_vertex_id = 0;
    int loop_KF_threshold = 0;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    double delay = cvRound(1000/30);  // real-time delay
    int track_entire_num;    
    int track_overlap_ratio;
    int feature_max_ID = 0;
    int TrackFeatureMaxID = 0;
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
    bool GoodLoopEdge = true;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////    
    
    cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat P1;
    cv::Mat init_campose = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat CurrCamPose = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat Iden = Mat::eye(4, 4, CV_64F);
    vector< vector<cv::Mat > > features;

    // parameter for track
    std::vector<cv::Point2f> PrevTrackPts, CurrTrackPts, PrevTriangulateTrackPts, CurrTriangulateTrackPts, PrevPnPTrackPts, CurrPnPTrackPts;
    std::vector<Point3d> MPforTrack;
    std::vector<int> TrackForTriangulatePtsID;

    
    cv::Mat World_R, World_t, World_Rt;
    std::vector<cv::Point2f> projectionpoints;
    
    cv::Mat X, map_change;
    std::vector<cv::Point2f> previous_track_point_for_triangulate, current_track_point_for_triangulate, keyframe_track_point;
    std::vector<cv::Point3d> SolvePnP_tracking_map;
    std::vector<int> SolvePnP_tracking_map_ID;
    std::vector<double> vec_scale;
    
    // Storage Storage_frame;
    std::vector<cv::Mat> inlier_storage;    
    std::vector<Frame> frame_storage;
    std::vector<cv::Mat> GT_Storage;
    std::vector<cv::Mat> draw_campose;
    Map map_storage;
    std::vector<cv::Point3d> inlier_map_storage;
    // posegraph
    std::vector<g2o::SE3Quat> vec_pose;
    std::vector<int> optimized_map_id;

    // DBoW2
    // OrbVocabulary voc("small_voc.yml.gz");
    std::cout << "load voc" << endl;
    OrbVocabulary voc("../0630_KITTI00-22_10_4_voc.yml.gz");
    std::cout << "copy voc to db" << endl;
    OrbDatabase db(voc, false, 0); // false = do not use direct index
  

    // const int image_x_size = previous_image.frame.cols;
    // const int image_y_size = previous_image.frame.rows;
    
    // parameter for Map
    std::vector<cv::KeyPoint> PrevKeypoints, CurrKeypoints;
    cv::Mat PrevDescriptors, CurrDescriptors, PrevMask, CurrMask, PrevInlier_mask, CurrInlier_mask;
    
    std::vector<cv::Point2f> PrevPts, CurrPts;
    std::vector<Point3d> CurrMP;
    std::vector<cv::DMatch> PrevGoodMatches, CurrGoodMatches;


    
    
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    
    Frame previous_image(0);
    
    video >> previous_image.frame;
    if(previous_image.frame.empty()) cv::waitKey();
    if (previous_image.frame.channels() > 1) cv::cvtColor(previous_image.frame, previous_image.frame, cv::COLOR_RGB2GRAY);
    previous_image.cam_pose = homogenous_campose_to_vec6d(init_campose);


    Frame copy_previous_image = previous_image;
    frame_storage.push_back(copy_previous_image);        
    
    // pose graph
    // SE3
    // create linear solver
    std::unique_ptr<g2o::BlockSolver_7_3::LinearSolverType> linear_solver = 
    g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_7_3::PoseMatrixType>>();
               
                    
    // create block solver                            
    std::unique_ptr<g2o::BlockSolver_7_3> block_solver =
    g2o::make_unique<g2o::BlockSolver_7_3>(std::move(linear_solver));

    g2o::OptimizationAlgorithm* algorithm
    = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));


    // g2o::SparseOptimizer* optimizer = new SparseOptimizer;
    // optimizer->setAlgorithm(algorithm);
    // optimizer->setVerbose(true); 
    
    // sim3 create linear solver and block solver

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<7, 7>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer_sim3;
    optimizer_sim3.setAlgorithm(solver);
    optimizer_sim3.setVerbose(true);   
    
    while(true)
    {
        // count image num
        std::cout << times << " Frame !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        
        // push new image
        Frame current_image(times);
        video >> current_image.frame;
        if (current_image.frame.empty()) break;
        if (current_image.frame.channels() > 1) cv::cvtColor(current_image.frame, current_image.frame, cv::COLOR_RGB2GRAY);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////        Caculate R, t using ESSENTIAL MATRIX or HOMOGRAPHY      ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
        
        if ( times < initialize_frame_num + 1)
        {
            
            // init image            
            if(times == 1)
            {   
                // Extract Feature ( ORB feature for map / goodfeature for track )
                cv::Ptr<cv::ORB> orb = cv::ORB::create(4000);
                orb->detectAndCompute(previous_image.frame, PrevMask, PrevKeypoints, PrevDescriptors);
                cv::goodFeaturesToTrack(previous_image.frame, PrevTrackPts, max_keypoint, 0.01, 10);
                PrevTriangulateTrackPts = PrevTrackPts;
std::cout << " Init Extract track point num  : " << PrevTrackPts.size() << endl;   
                

                // for(int i = 0; i < previous_image.pts.size(); i++)
                // {
                //     keyframe_track_point.push_back(previous_image.pts[i]);
                //     // previous_track_point_for_triangulate.push_back(previous_image.pts[i]);
                // }
            }
            
            
            // Matching Feature using 2d view for CamPose   
            TrackOpticalFlowAndRemoveErrForTriangulate(previous_image.frame, current_image.frame, PrevTrackPts, CurrTrackPts, PrevTriangulateTrackPts);
std::cout << "After Matching track point num  : " << CurrTrackPts.size() << endl;            
            
            
            // Caculate relative R, t using Essential Matrix
            std::cout << "essential matrix" << endl;
            cv::Mat E, inlier_mask, R, t, Rt;
            E = cv::findEssentialMat(PrevTrackPts, CurrTrackPts, f, c, cv::RANSAC, 0.99, 1, inlier_mask);
            int inlier_num = cv::recoverPose(E, PrevTrackPts, CurrTrackPts, R, t, f, c, inlier_mask);
            


            // Update cam_pose using R, t
            Rt = R_t_to_homogenous(R, t);
            CurrCamPose = CurrCamPose * Rt.inv();
            
            // Storage cam_pose
            cv::Mat clone_CurrCamPose;
            clone_CurrCamPose = CurrCamPose.clone();
            current_image.cam_pose = homogenous_campose_to_vec6d(clone_CurrCamPose);
std::cout << " current cam pose " << std::endl << clone_CurrCamPose << std::endl;

            if(times < initialize_frame_num)
            {
                // Storage current image 
                Frame copy_current_image = current_image;
                frame_storage.push_back(copy_current_image);
            }


            



            if(times == initialize_frame_num)
            {
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////       Triangulate for Track  /////////////////////////////////////////////////////////////
                
                std::cout << "Track point triangulation " << endl;
                
                // Make 3d Point for track ( PnP ) 
                P1 = K * cam_storage_to_projection_matrix(current_image.cam_pose);
                cv::triangulatePoints(P0, P1, PrevTriangulateTrackPts, CurrTrackPts, X);
                
                // MP
                std::cout << " storage map point " << endl;
                world_xyz_point_to_homogenous(X);
                X.convertTo(X, CV_64F);
                MPforTrack.clear();
                for (int i = 0; i < CurrTrackPts.size(); i++ ) MPforTrack.push_back(Point3d(X.at<double>(0, i), X.at<double>(1, i), X.at<double>(2, i)));
                
                // Remove trackMP Outlier ( remove back MP ? and PnP outlier )
std::cout << " TrackMP size before remove outlier : " <<  MPforTrack.size() << endl;
                RemoveTrackMPOutlier(map_storage, MPforTrack,  CurrTrackPts, keyframe_num, vec6d_to_homogenous_campose(current_image.cam_pose));
std::cout << " TrackMP size after remove back outlier : " <<  MPforTrack.size() << endl;               

                cv::Mat rot, tran, rottran, inliers;
                cv::solvePnPRansac(MPforTrack, CurrTrackPts, K, cv::noArray(), rot, tran, false, 100, SolvePnP_reprojection_error, 0.99, inliers, SolevePnPFlags );
                int BeforeRemovePnPOutlierNum = CurrTrackPts.size();
                RemoveTrackMPPnPOutlier(MPforTrack, CurrTrackPts, inliers);
std::cout << keyframe_num << "  keyframe Track PnP Inlier rate : " << 100 * inliers.rows / BeforeRemovePnPOutlierNum << endl;
                cv::Rodrigues(rot, rot);
                rottran = R_t_to_homogenous(rot, tran);
std::cout << " SolvePnP Track outlier Test pose : " << std::endl << rottran.inv() << std::endl;
                
                PrevPnPTrackPts = CurrTrackPts;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                
                // Storage Track Pts,ID
                current_image.TrackPts.clear();
                current_image.TrackId.clear();
                for(int i = 0; i < CurrTrackPts.size(); i++){
                    current_image.TrackPts.push_back(CurrTrackPts[i]);
                    current_image.TrackId.push_back(TrackFeatureMaxID);
                    map_storage.TrackMP.insert(std::pair<int, cv::Point3d>(TrackFeatureMaxID, MPforTrack[i]));
                    TrackFeatureMaxID++;
                }
                
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////       Triangulate for Map  /////////////////////////////////////////////////////////////
                
                // Extract ORB Feature for Map 
                cv::Ptr<cv::ORB> orb = cv::ORB::create(4000);
                std::cout << "Extracting ORB features in " << keyframe_num << " keyframe " << std::endl;


                features.clear();
                orb->detectAndCompute(current_image.frame, CurrMask, CurrKeypoints, CurrDescriptors);
                features.push_back(vector<cv::Mat >());
                changeStructure(CurrDescriptors, features.back());
                
                // map_storage.LoopDescriptor.insert(std::pair<int, cv::Mat>(keyframe_num, descriptors));
                // map_storage.LoopKeyPoint.insert(std::pair<int, std::vector<cv::KeyPoint>>(keyframe_num, keypoints));


                db.add(features[0]);
                
                // Matching ORB Feature 
                cv::Ptr<DescriptorMatcher> matcher = cv::BFMatcher::create(NORM_HAMMING);
                std::vector<cv::DMatch> matches;
                matcher -> match(PrevDescriptors, CurrDescriptors, matches);
                std::sort(matches.begin(), matches.end());

                CurrGoodMatches.clear();
                CurrGoodMatches = matches;

std::cout << "During 4000 ,,, Best Distance : " << CurrGoodMatches[0].distance << "     " <<
                        " Median Distance : " << CurrGoodMatches[2000].distance << "    " <<
                        " Worst Distance : " << CurrGoodMatches[3990].distance << std::endl;

                for(size_t i = 0; i < matches.size(); i++)
                {
                    PrevPts.push_back(PrevKeypoints[matches[i].queryIdx].pt);
                    CurrPts.push_back(CurrKeypoints[matches[i].trainIdx].pt);
                } 
                
                // Remove Essential Outlier
                RemoveEssentialOutlier(map_storage, PrevPts, CurrPts, keyframe_num, K);

                // Good Match
                GoodMatch(map_storage, keyframe_num, 300);
std::cout << " During remove Essential  ,,, Best Distance : " << CurrGoodMatches[map_storage.MapMatchIdx[keyframe_num][0]].distance << std::endl;
                PrevPts.clear();
                CurrPts.clear();
                for(size_t i = 0; i < map_storage.MapMatchIdx[keyframe_num].size(); i++)
                {
                    PrevPts.push_back(PrevKeypoints[matches[map_storage.MapMatchIdx[keyframe_num][i]].queryIdx].pt);
                    CurrPts.push_back(CurrKeypoints[matches[map_storage.MapMatchIdx[keyframe_num][i]].trainIdx].pt);
                }                

                // // drawing the results
                // std::vector<cv::DMatch> DrawMatch;
                // Mat img_matches;
                // for(int i = 0; i < map_storage.MapMatchIdx[keyframe_num].size(); i++)
                // {
                //     DrawMatch.push_back(matches[map_storage.MapMatchIdx[keyframe_num][i]]);
                // }
                
                // cv::drawMatches(frame_storage[0].frame, PrevKeypoints, current_image.frame, CurrKeypoints,
                //     DrawMatch, img_matches, Scalar::all(-1), Scalar::all(-1),
                //     std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                // cv::imshow("matches", img_matches);
                // cv::waitKey();

                
                



                cv::triangulatePoints(P0, P1, PrevPts, CurrPts, X);
                world_xyz_point_to_homogenous(X);
                X.convertTo(X, CV_64F);
                CurrMP.clear();
                for (int i = 0; i < CurrPts.size(); i++ ) CurrMP.push_back(Point3d(X.at<double>(0, i), X.at<double>(1, i), X.at<double>(2, i)));


                


                // Remove MP Outlier ( remove back MP ? and PnP outlier )
std::cout << " MP size before remove outlier : " <<  CurrMP.size() << endl;
                RemoveMPOutlier(map_storage, CurrMP,  CurrPts, keyframe_num, vec6d_to_homogenous_campose(current_image.cam_pose));
std::cout << " MP size after remove back outlier : " <<  CurrMP.size() << endl;               

                
                cv::solvePnPRansac(CurrMP, CurrPts, K, cv::noArray(), rot, tran, false, 100, SolvePnP_reprojection_error, 0.99, inliers, SolevePnPFlags );
                BeforeRemovePnPOutlierNum = CurrPts.size();
                RemoveMPPnPOutlier(map_storage, CurrMP, CurrPts, inliers, keyframe_num);
std::cout << keyframe_num << "th  keyframe Track PnP Inlier rate : " << 100 * inliers.rows / BeforeRemovePnPOutlierNum << endl;

                cv::Rodrigues(rot, rot);
                rottran = R_t_to_homogenous(rot, tran);
std::cout << " SolvePnP MP outlier Test pose : " << std::endl << rottran.inv() << std::endl;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


                // Storage pts, pts_id, lankmark
                for(int i = 0; i < CurrPts.size(); i++) 
                {
                    map_storage.world_xyz.insert(std::pair<int, cv::Point3d>(feature_max_ID, CurrMP[i]));
                    current_image.pts_id.push_back(feature_max_ID);
                    feature_max_ID++;
                }    
                std::cout << std::endl;
                for(int i = 0; i < CurrPts.size(); i++) current_image.pts.push_back(CurrPts[i]);

                
                // Storage current image 
                Frame copy_current_image = current_image;
                frame_storage.push_back(copy_current_image);

                // storage id - keyframe
                map_storage.keyframe[keyframe_num] = frame_storage[times];
cv::imshow(" keyframe image ", map_storage.keyframe[keyframe_num].frame);
std::cout << "@@@@@@@@@@ First keyframe selection @@@@@@@" << std::endl
<< " keyframe num is  : " << keyframe_num << std::endl;                    


     
                // MapToKF_ids
                for(int i = 0; i < map_storage.keyframe[keyframe_num].pts_id.size(); i++)
                {
                    int id = map_storage.keyframe[keyframe_num].pts_id[i];
                    map_storage.MapToKF_ids[id].push_back(keyframe_num);
                }

std::cout << " Total Landmark's num : " << map_storage.world_xyz.size() << std::endl;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





                

                


                
                // intitialize pose graph vertex
                g2o::VertexSim3Expmap* v_sim3 = new g2o::VertexSim3Expmap();
                v_sim3->setId(0);
                v_sim3->setMarginalized(false);
                cv::Mat cam_pose4d = vec6d_to_homogenous_campose(map_storage.keyframe[0].cam_pose);
                cam_pose4d = cam_pose4d.inv();
                Eigen::Matrix3d sim3_r = RotationMatToEigen3d(homogenous_campose_to_R(cam_pose4d));
                Eigen::Vector3d sim3_t(homogenous_campose_to_t(cam_pose4d).at<double>(0, 0),
                                        homogenous_campose_to_t(cam_pose4d).at<double>(1, 0),
                                        homogenous_campose_to_t(cam_pose4d).at<double>(2, 0));
                g2o::Sim3 sim3(sim3_r, sim3_t, 1.0);
                v_sim3->setEstimate(sim3);

                
                v_sim3->setFixed(true);
                optimizer_sim3.addVertex(v_sim3);
                        

                // new feature to track
                std::cout << " New Track Feature " << endl;
                cv::goodFeaturesToTrack(current_image.frame, CurrTrackPts, max_keypoint, 0.01, 10);

                PrevTriangulateTrackPts.clear();
                PrevTriangulateTrackPts.assign( CurrTrackPts.begin(), CurrTrackPts.end() );
std::cout << " New Track feature num  : " << PrevTriangulateTrackPts.size() << endl; 
                track_entire_num = PrevTriangulateTrackPts.size();
                
                // Correspondance New Feature
                TrackForTriangulatePtsID.clear();
                std::vector<cv::Point2f> clone_pts(current_image.TrackPts);
                std::vector<int> clone_pts_id(current_image.TrackId);
                int same_point_num(0), different_point_num(0);

                for(int j = 0; j < CurrTrackPts.size(); j++)
                {
                    bool DifferentPts = true;
                    for(int i = 0; i < clone_pts.size(); i++)
                    {
                        double difference_point2d_x = cv::abs(clone_pts[i].x - CurrTrackPts[j].x);
                        double difference_point2d_y = cv::abs(clone_pts[i].y - CurrTrackPts[j].y);

                        if(difference_point2d_x < 3.0 and difference_point2d_y < 3.0)
                        {
                            TrackForTriangulatePtsID.push_back(clone_pts_id[i]);
                            clone_pts.erase(clone_pts.begin() + i);
                            clone_pts_id.erase(clone_pts_id.begin() + i);
                            same_point_num++;
                            DifferentPts = false;
                            break;
                        }
                    }
                    
                    if(DifferentPts)
                    {
                        TrackForTriangulatePtsID.push_back(TrackFeatureMaxID);
                        different_point_num++;
                        TrackFeatureMaxID++;
                    }

                }
std::cout << " Track same point :  " << same_point_num << " Track different point num  : " << different_point_num << endl;
                

                // BA initialize
                // Define CostFunction
                ceres::Problem initilize_ba;

                for ( int i = 0; i < map_storage.keyframe[0].TrackPts.size(); i++)
                {
                    ceres::CostFunction* cost_func = ReprojectionError::create(map_storage.keyframe[0].TrackPts[i], f, cv::Point2d(c.x, c.y));
                    double* camera = (double*)(&map_storage.keyframe[0].cam_pose);
                    double* X_ = (double*)(&(map_storage.TrackMP[i]));
                    initilize_ba.AddResidualBlock(cost_func, new CauchyLoss(0.2), camera, X_); 
                }            
                            
                // ceres option       
                ceres::Solver::Options options;
                // options.max_num_iterations = 50;
                options.linear_solver_type = ceres::DENSE_SCHUR;
                options.num_threads = 8;
                options.minimizer_progress_to_stdout = true;
                ceres::Solver::Summary summary;

                std::cout << " camera cam pose before intialize ba " << std::endl << vec6d_to_homogenous_campose(map_storage.keyframe[0].cam_pose) << endl;

                ceres::Solve(options, &initilize_ba, &summary);

                std::cout << " camera cam pose after intialize ba " << std::endl << vec6d_to_homogenous_campose(map_storage.keyframe[0].cam_pose) << endl;


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////






                // // Reprojection 3D to 2D current image
                // std::cout << "reprojection 3D to 2D initialize stage " << std::endl;
                // cv::Mat R_, t_;
                // cv::sfm::KRtFromProjection(P1, K, R_, t_);
                // currMP.clear();
                // for (int c = 0; c < map_storage.world_xyz.size(); c++ ) currMP.push_back(Point3d(map_storage.world_xyz[c].x, map_storage.world_xyz[c].y, map_storage.world_xyz[c].z));
                // cv::Mat project_mat = Mat(currMP).clone();
                // project_mat.convertTo(project_mat, CV_32F);
                // cv::projectPoints(project_mat, R_, t_, K, cv::noArray(), projectionpoints);                

                PrevGoodMatches.clear();
                PrevKeypoints.clear();
                PrevDescriptors.release();
                PrevMask.release();

                PrevGoodMatches.assign( CurrGoodMatches.begin(), CurrGoodMatches.end() );
                PrevKeypoints.assign( CurrKeypoints.begin(), CurrKeypoints.end() );
                PrevDescriptors = CurrDescriptors.clone();
                PrevMask = CurrMask.clone();
            }
        
        
        }
            




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////            
////////////////////////////      SOLVEPNP + TRIANGULATION             //////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
        
        if (times > initialize_frame_num)
        {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////            
//////////////////////////////////////////////////// Track for SolvePnP //////////////////////////////////////////////            
            
            // Matching prev_image and current_image using optical flow
std::cout << " before PnP track point num  : " << PrevPnPTrackPts.size() << std::endl;  
            // track_opticalflow_and_remove_err_for_SolvePnP(previous_image.frame, current_image.frame, PrevPnPTrackPts, CurrPnPTrackPts, MPforTrack);
            // PrevPnPTrackPts = CurrPnPTrackPts;
            track_opticalflow_and_remove_err_for_SolvePnP_(previous_image.frame, current_image.frame, previous_image.TrackPts, current_image.TrackPts, previous_image.TrackId, MPforTrack);
            current_image.TrackId.clear();
            for(int i = 0; i < current_image.TrackPts.size(); i++) current_image.TrackId.push_back(previous_image.TrackId[i]);

            
            // Calculate World R, t using SolvePnP
            std::cout << "SolvePnP" << endl;
std::cout << "SolvePnP map_point num : " << "   " << MPforTrack.size() << endl;
std::cout << "SolvePnP current_image_pts num : " << "   " << current_image.TrackPts.size() << endl;
            cv::Mat inliers;
            MPforTrack.clear();
            for(int i = 0; i < previous_image.TrackId.size(); i ++) MPforTrack.push_back(map_storage.TrackMP[current_image.TrackId[i]]);
            cv::solvePnPRansac(MPforTrack, current_image.TrackPts, K, cv::noArray(), World_R, World_t, false, 100, SolvePnP_reprojection_error, 0.99, inliers, SolevePnPFlags );
            int SolvePnP_inlier_ratio = 100 * inliers.rows / MPforTrack.size();
std::cout << " SolvePnP Track inlier num : " << inliers.rows << std::endl;            
std::cout << " Track PnP Inlier ratio : " << SolvePnP_inlier_ratio << " %" << endl;


            // Calculate camera_pose
            std::cout << "Caculate PnP pose" << endl;
            cv::Rodrigues(World_R, World_R);
            World_Rt = R_t_to_homogenous(World_R, World_t);
            CurrCamPose = World_Rt.inv();

            // Storage camera pose
            cv::Mat clone_CurrCamPose;
            clone_CurrCamPose = CurrCamPose.clone();
            current_image.cam_pose = homogenous_campose_to_vec6d(clone_CurrCamPose);
std::cout << " current cam pose " << std::endl << clone_CurrCamPose << std::endl;

            
            // Motion only BA
            ceres::Problem motion_only_ba;
            
            for ( int i = 0; i < current_image.TrackPts.size(); i++)
            {
                ceres::CostFunction* motion_only_cost_func = motion_only_ReprojectionError::create(current_image.TrackPts[i], MPforTrack[i], f, cv::Point2d(c.x, c.y));
                double* camera_ = (double*)(&current_image.cam_pose);
        
                motion_only_ba.AddResidualBlock(motion_only_cost_func, new CauchyLoss(0.2), camera_); 
            }            
            
            // ceres option       
            ceres::Solver::Options options;
            // options.max_num_iterations = 50;
            // options.gradient_tolerance = 1e-9;
            // options.function_tolerance = 1e-9;
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.num_threads = 8;
            options.minimizer_progress_to_stdout = false;
            ceres::Solver::Summary summary;

            std::cout <<" ( before motion only BA )current camera pose  : " << std::endl;
            std::cout << vec6d_to_homogenous_campose(current_image.cam_pose) << endl;

            // solve
            ceres::Solve(options, &motion_only_ba, &summary);

            std::cout <<" ( after motion only BA )current camera pose  : " << std::endl;
            std::cout << vec6d_to_homogenous_campose(current_image.cam_pose) << endl;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// track for triangulate///////////////////////////////////////////////////////////////////////////////////////////
            
            std::cout << " Track for Triangulate " << endl;
            TrackOpticalFlowAndRemoveErrForTriangulate(previous_image.frame, current_image.frame, PrevTrackPts, CurrTrackPts, PrevTriangulateTrackPts, TrackForTriangulatePtsID);
std::cout  << " Curr Matching Track point num :  " << CurrTrackPts.size() << std::endl;       
                


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////   Determine Keyframe or not    ////////////////////////////////////////////////////////////////////////////////////
            
            // track_overlap_ratio 
            track_overlap_ratio = 100 * CurrTrackPts.size() / track_entire_num;
std::cout << "  Track overlap ratio : " << track_overlap_ratio << endl;

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
                

          
            
            
            
// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////            
            
            if (new_keyframe_selection)
            {
                keyframe_num++;

                glClear(GL_COLOR_BUFFER_BIT);

                
                // Caculate projection matrix for triangulation
                std::cout << "  projection matrix    " << endl;
                std::cout << "P0 campose : " << std::endl << vec6d_to_homogenous_campose(map_storage.keyframe[keyframe_num - 1].cam_pose) << endl;
                std::cout << "P1 campose : " << std::endl << vec6d_to_homogenous_campose(current_image.cam_pose) << endl;
                P0 = K * cam_storage_to_projection_matrix(map_storage.keyframe[keyframe_num - 1].cam_pose);
                P1 = K * cam_storage_to_projection_matrix(current_image.cam_pose);
                    
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////       Triangulate for Track  /////////////////////////////////////////////////////////////
                std::cout << "Track point triangulation " << endl;
                
                // Make 3d Point for track ( PnP ) 
                cv::triangulatePoints(P0, P1, PrevTriangulateTrackPts, CurrTrackPts, X);
                
                // MP
                std::cout << " storage map point " << endl;
                world_xyz_point_to_homogenous(X);
                X.convertTo(X, CV_64F);
                MPforTrack.clear();
                for (int i = 0; i < CurrTrackPts.size(); i++ ) MPforTrack.push_back(Point3d(X.at<double>(0, i), X.at<double>(1, i), X.at<double>(2, i)));
                
                // Remove trackMP Outlier ( remove back MP ? and PnP outlier )
std::cout << " TrackMP size before remove outlier : " <<  MPforTrack.size() << endl;
                RemoveTrackMPOutlier(map_storage, MPforTrack,  CurrTrackPts, keyframe_num, vec6d_to_homogenous_campose(current_image.cam_pose), TrackForTriangulatePtsID);
std::cout << " TrackMP size after remove back outlier : " <<  MPforTrack.size() << endl;               

                cv::Mat rot, tran, rottran, inliers;
                cv::solvePnPRansac(MPforTrack, CurrTrackPts, K, cv::noArray(), rot, tran, false, 100, SolvePnP_reprojection_error, 0.99, inliers, SolevePnPFlags );
                int BeforeRemovePnPOutlierNum = CurrTrackPts.size();
                RemoveTrackMPPnPOutlier(MPforTrack, CurrTrackPts, inliers, TrackForTriangulatePtsID);
std::cout << " TrackMP size after remove SolvePnP outlier : " <<  MPforTrack.size() << endl;               

std::cout << keyframe_num << "  keyframe Track PnP Inlier rate : " << 100 * inliers.rows / BeforeRemovePnPOutlierNum << endl;
                cv::Rodrigues(rot, rot);
                rottran = R_t_to_homogenous(rot, tran);
std::cout << " SolvePnP Track outlier Test pose : " << std::endl << rottran.inv() << std::endl;
                PrevPnPTrackPts = CurrTrackPts;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                
                // Storage Track Pts,ID
                current_image.TrackPts.clear();
                current_image.TrackId.clear();
                for(int i = 0; i < CurrTrackPts.size(); i++){
                    current_image.TrackPts.push_back(CurrTrackPts[i]);
                    current_image.TrackId.push_back(TrackForTriangulatePtsID[i]);
                    
                    if(map_storage.TrackMP.find(TrackForTriangulatePtsID[i]) == map_storage.TrackMP.end())
                    {

                            map_storage.TrackMP.insert(std::pair<int, cv::Point3d>(TrackForTriangulatePtsID[i], MPforTrack[i]));


                    }
                }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////       Triangulate for Map  /////////////////////////////////////////////////////////////
                
                
                // Extract ORB Feature for Map 
                cv::Ptr<cv::ORB> orb = cv::ORB::create(4000);
                std::cout << "Extracting ORB features in " << keyframe_num << " keyframe " << std::endl;


                features.clear();
                orb->detectAndCompute(current_image.frame, CurrMask, CurrKeypoints, CurrDescriptors);
                features.push_back(vector<cv::Mat >());
                changeStructure(CurrDescriptors, features.back());
                
                // map_storage.LoopDescriptor.insert(std::pair<int, cv::Mat>(keyframe_num, descriptors));
                // map_storage.LoopKeyPoint.insert(std::pair<int, std::vector<cv::KeyPoint>>(keyframe_num, keypoints));


                db.add(features[0]);
                
                // Matching ORB Feature 
                cv::Ptr<DescriptorMatcher> matcher = cv::BFMatcher::create(NORM_HAMMING);
                std::vector<cv::DMatch> matches;
                matcher -> match(PrevDescriptors, CurrDescriptors, matches);
                std::sort(matches.begin(), matches.end());
                CurrGoodMatches.clear();
                CurrGoodMatches = matches;
std::cout << "During 4000 ,,, Best Distance : " << CurrGoodMatches[0].distance << "     " <<
                        " Median Distance : " << CurrGoodMatches[2000].distance << "    " <<
                        " Worst Distance : " << CurrGoodMatches[3990].distance << std::endl;
                PrevPts.clear();
                CurrPts.clear();
                for(size_t i = 0; i < matches.size(); i++)
                {
                    PrevPts.push_back(PrevKeypoints[matches[i].queryIdx].pt);
                    CurrPts.push_back(CurrKeypoints[matches[i].trainIdx].pt);
                } 
                
                // Remove Essential Outlier
                RemoveEssentialOutlier(map_storage, PrevPts, CurrPts, keyframe_num, K);
                
                // Good Match
                GoodMatch(map_storage, keyframe_num, 300);
std::cout << " During remove Essential  ,,, Best Distance : " << CurrGoodMatches[map_storage.MapMatchIdx[keyframe_num][0]].distance << std::endl;

                PrevPts.clear();
                CurrPts.clear();
                for(size_t i = 0; i < map_storage.MapMatchIdx[keyframe_num].size(); i++)
                {
                    PrevPts.push_back(PrevKeypoints[matches[map_storage.MapMatchIdx[keyframe_num][i]].queryIdx].pt);
                    CurrPts.push_back(CurrKeypoints[matches[map_storage.MapMatchIdx[keyframe_num][i]].trainIdx].pt);
                }
                // // drawing the results
                // std::vector<cv::DMatch> DrawMatch;
                // Mat img_matches;
                // for(int i = 0; i < map_storage.MapMatchIdx[keyframe_num].size(); i++)
                // {
                //     DrawMatch.push_back(matches[map_storage.MapMatchIdx[keyframe_num][i]]);
                // }
                // std::cout << DrawMatch.size() << std::endl;
                
                // cv::drawMatches(map_storage.keyframe[keyframe_num - 1].frame, PrevKeypoints, current_image.frame, CurrKeypoints,
                //     DrawMatch, img_matches, Scalar::all(-1), Scalar::all(-1),
                //     std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                // cv::imshow("matches", img_matches);
                // cv::waitKey();

                cv::triangulatePoints(P0, P1, PrevPts, CurrPts, X);
                world_xyz_point_to_homogenous(X);
                X.convertTo(X, CV_64F);
                CurrMP.clear();
                for (int i = 0; i < CurrPts.size(); i++ ) CurrMP.push_back(Point3d(X.at<double>(0, i), X.at<double>(1, i), X.at<double>(2, i)));


                


                // Remove MP Outlier ( remove back MP ? and PnP outlier )
std::cout << " MP size before remove outlier : " <<  CurrMP.size() << endl;
                RemoveMPOutlier(map_storage, CurrMP,  CurrPts, keyframe_num, vec6d_to_homogenous_campose(current_image.cam_pose));
std::cout << " MP size after remove back outlier : " <<  CurrMP.size() << endl;               

                
                cv::solvePnPRansac(CurrMP, CurrPts, K, cv::noArray(), rot, tran, false, 100, SolvePnP_reprojection_error, 0.99, inliers, SolevePnPFlags );
                BeforeRemovePnPOutlierNum = CurrPts.size();
                RemoveMPPnPOutlier(map_storage, CurrMP, CurrPts, inliers, keyframe_num);
std::cout << " MP size after remove SolvePnP outlier : " <<  CurrMP.size() << endl;               

std::cout << keyframe_num << "th  keyframe MP PnP Inlier rate : " << 100 * inliers.rows / BeforeRemovePnPOutlierNum << endl;                

                cv::Rodrigues(rot, rot);
                rottran = R_t_to_homogenous(rot, tran);
std::cout << " SolvePnP Track outlier Test pose : " << std::endl << rottran.inv() << std::endl;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////                
                
                // Storage pts, pts_id, lankmark
                int NewCnt(0), OldCnt(0);
                for(int i = 0; i < map_storage.MapMatchIdx[keyframe_num].size(); i++){
                    
                    bool NewID = true;
                    int CurrQueryIdx = CurrGoodMatches[map_storage.MapMatchIdx[keyframe_num][i]].queryIdx;
                    
                    for(int j = 0; j < map_storage.MapMatchIdx[keyframe_num - 1].size(); j++){

                        // Same Point -> Same ID
                        int PrevTrainIdx = PrevGoodMatches[map_storage.MapMatchIdx[keyframe_num - 1][j]].trainIdx;
                        if(CurrQueryIdx == PrevTrainIdx){
                            int id = map_storage.keyframe[keyframe_num - 1].pts_id[j];
                            current_image.pts_id.push_back(id);
                            NewID = false;
                            OldCnt++;
                            break;
                        }
                        
                    }
                    // Different Point -> New ID
                    if(NewID){

                        map_storage.world_xyz.insert(std::pair<int, cv::Point3d>(feature_max_ID, CurrMP[i]));
                        current_image.pts_id.push_back(feature_max_ID);
                        feature_max_ID++;
                        NewCnt++;
                    }
                }
                std::cout << std::endl;
                std::cout << " new point num : " << NewCnt << "         " << " present point num : " << OldCnt << std::endl;

                std::cout << std::endl;
                for(int i = 0; i < CurrPts.size(); i++) current_image.pts.push_back(CurrPts[i]);

                
                // Storage current image 
                Frame copy_current_image = current_image;
                frame_storage.push_back(copy_current_image);

                // storage id - keyframe
                map_storage.keyframe[keyframe_num] = frame_storage[times];
cv::imshow(" keyframe image ", (map_storage.keyframe[keyframe_num]).frame);
std::cout << "@@@@@@@@@@ First keyframe selection @@@@@@@" << std::endl 
<< " keyframe num is  : " << keyframe_num << endl;                    


     
                // MapToKF_ids
                for(int i = 0; i < map_storage.keyframe[keyframe_num].pts_id.size(); i++)
                {
                    int id = map_storage.keyframe[keyframe_num].pts_id[i];
                    map_storage.MapToKF_ids[id].push_back(keyframe_num);
                }

std::cout << " Total Landmark's num : " << map_storage.world_xyz.size() << std::endl;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


                // // storage landmark id
                // std::vector<cv::Point2f> clone_current_track_point_for_triangulate(current_track_point_for_triangulate);
                // std::vector<cv::Point3d> clone_map_point(map_point);
                // std::vector<int> clone_previous_track_point_for_triangulate_ID(previous_track_point_for_triangulate_ID);
                
                //     int indexCorrect = 0;
                // for(int i = 0; i < clone_current_track_point_for_triangulate.size(); i++)
                // {
                //     if(map_storage.world_xyz.find(clone_previous_track_point_for_triangulate_ID[i]) == map_storage.world_xyz.end())
                //     {

                //             map_storage.world_xyz.insert(std::pair<int, cv::Point3d>(clone_previous_track_point_for_triangulate_ID[i], clone_map_point[i]));


                //     }
                //     else 
                //     {

                //     double diff_3d_x = std::abs(clone_map_point[i].x - map_storage.world_xyz[clone_previous_track_point_for_triangulate_ID[i]].x);
                //     double diff_3d_y = std::abs(clone_map_point[i].y - map_storage.world_xyz[clone_previous_track_point_for_triangulate_ID[i]].y);
                //     double diff_3d_z = std::abs(clone_map_point[i].z - map_storage.world_xyz[clone_previous_track_point_for_triangulate_ID[i]].z);
                    
                //         if(diff_3d_x < 3.0 and diff_3d_y < 3.0 and diff_3d_z < 3.0)
                //         {
                //             map_point[i - indexCorrect] = map_storage.world_xyz[clone_previous_track_point_for_triangulate_ID[i]];
                //         }
                //         else
                //         {
                //             current_track_point_for_triangulate.erase(current_track_point_for_triangulate.begin() + i - indexCorrect );
                //             map_point.erase(map_point.begin() + i - indexCorrect );
                //             previous_track_point_for_triangulate_ID.erase(previous_track_point_for_triangulate_ID.begin() + i - indexCorrect);
                //             indexCorrect++;

                //         }
                //     }
                // }


//                 // Storage SolvePnP inlier index
//                 cv::Mat rot, tran, inliers_;
// std::cout << map_point.size() << "      " << current_track_point_for_triangulate.size() << endl;
//                 map_point.clear();
//                 for(int i = 0; i < previous_track_point_for_triangulate_ID.size(); i++) map_point.push_back(map_storage.world_xyz[previous_track_point_for_triangulate_ID[i]]);
//                 cv::solvePnPRansac(map_point, current_track_point_for_triangulate, K, cv::noArray(), rot, tran, false, 100, SolvePnP_reprojection_error, 0.99, inliers_ );
//                 inlier_storage.push_back(inliers_);
// std::cout << keyframe_num + 1 << " keyframe inlier storage rate : " << 100 * inliers_.rows / map_point.size() << endl;                
                



//                 // Storage pts
//                 current_image.pts.clear();
//                 current_image.pts_id.clear();
//                 for(int i = 0; i < current_track_point_for_triangulate.size(); i++) 
//                 {
//                     current_image.pts.push_back(current_track_point_for_triangulate[i]);                
//                     current_image.pts_id.push_back(previous_track_point_for_triangulate_ID[i]);
//                 }

// std::cout << " size print " << current_track_point_for_triangulate.size() << "    " << previous_track_point_for_triangulate_ID.size() << endl;

      
//                 // Storage current image 
//                 Frame copy_current_image = current_image;
//                 frame_storage.push_back(copy_current_image);
// std::cout << " current frame_storage size : " << frame_storage.size() << endl;             
                
//                 keyframe_num += 1;
//                 // storage id - keyframe
//                 map_storage.keyframe.insert(std::pair<int, Frame>(keyframe_num, frame_storage[times]));
            
// cv::imshow(" keyframe image ", map_storage.keyframe[keyframe_num].frame);
// std::cout << "@@@@@@@@@ New keyframe was made @@@@@@@@@" << endl << " Keyframe num is  : " << keyframe_num << " Frame num is : " << times << endl;   

//                 // MapToKF_ids
//                 for(int i = 0; i < map_storage.keyframe[keyframe_num].pts_id.size(); i++)
//                 {
//                     int id_ = map_storage.keyframe[keyframe_num].pts_id[i];
                
//                     map_storage.MapToKF_ids[id_].push_back(keyframe_num);
//                 }            


                
// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// /////////////////////////////////////////   Local  Bundle Adjustment    ////////////////////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


                ceres::Problem keyframe_ba;
                int p1 = 0;
                if(keyframe_num - active_keyframe_num + 1 < 0) p1 = keyframe_num - active_keyframe_num + 1;
                else p1 = 0;
                
                for(int j = keyframe_num - active_keyframe_num + 1 - p1; j < keyframe_num + 1; j++)
                {
std::cout << " active keyframe num  : " << j << endl;                        
                    for ( int i = 0; i < map_storage.keyframe[j].pts.size(); i++)
                    {
                        ceres::CostFunction* keyframe_cost_func = ReprojectionError::create(map_storage.keyframe[j].pts[i], f, cv::Point2d(c.x, c.y));

                        double* camera = (double*)(&map_storage.keyframe[j].cam_pose);
                        int id_ = map_storage.keyframe[j].pts_id[i];
                        double* X_ = (double*)(&(map_storage.world_xyz[id_]));
                        keyframe_ba.AddResidualBlock(keyframe_cost_func, NULL, camera, X_); 

                    }            
                }

                int p2 = 0;
                if(keyframe_num - active_keyframe_num + 1 - fix_keyframe_num < 0) p2 = keyframe_num - active_keyframe_num + 1 - fix_keyframe_num;
                else p2 = 0;

                for(int j = keyframe_num - active_keyframe_num + 1 - fix_keyframe_num - p2; j < keyframe_num - active_keyframe_num + 1; j++)
                {
std::cout << " fixed keyframe num  : " << j << endl;                         
                    for ( int i = 0; i < map_storage.keyframe[j].pts.size(); i++)
                    {
                        ceres::CostFunction* fix_keyframe_cost_func = map_point_only_ReprojectionError::create(map_storage.keyframe[j].pts[i], map_storage.keyframe[j].cam_pose, f, cv::Point2d(c.x, c.y));
                        int id_ = map_storage.keyframe[j].pts_id[i];
                        double* X_ = (double*)(&(map_storage.world_xyz[id_]));
                        keyframe_ba.AddResidualBlock(fix_keyframe_cost_func, NULL, X_); 
                                
                    }            
                }                    
                                
                                
                // ceres option       
                ceres::Solver::Options options;
                // options.max_num_iterations = 50;
                // options.gradient_tolerance = 1e-9;
                // options.function_tolerance = 1e-9;
                options.linear_solver_type = ceres::ITERATIVE_SCHUR;
                options.num_threads = 8;
                options.minimizer_progress_to_stdout = false;
                ceres::Solver::Summary summary;

                // Camera pose and map_point before BA
                std::cout <<" camera pose before keyframe BA " << endl;
                for(int j = keyframe_num - active_keyframe_num + 1 - p1; j < keyframe_num + 1; j++) std::cout << " keyframe num : " << j << endl << vec6d_to_homogenous_campose(map_storage.keyframe[j].cam_pose) << endl;
                    
                // solve
                ceres::Solve(options, &keyframe_ba, &summary);                
                    
                // Camera pose and map_point after BA
                std::cout << endl << " camera pose after keyframe BA " << endl;
                for(int j = keyframe_num - active_keyframe_num + 1 - p1; j < keyframe_num + 1; j++) std::cout << " keyframe num : " << j << endl << vec6d_to_homogenous_campose(map_storage.keyframe[j].cam_pose) << endl;                



// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
//                 cv::Ptr<cv::ORB> orb = cv::ORB::create(4000);
//                 cout << "Extracting ORB features..." << times << endl;

//                 vector<cv::KeyPoint> keypoints;
//                 cv::Mat descriptors;
//                 cv::Mat mask, inlier_mask;
//                 features.clear();
//                 orb->detectAndCompute(current_image.frame, mask, keypoints, descriptors);
//                 features.push_back(vector<cv::Mat >());
//                 changeStructure(descriptors, features.back());
               
//                 cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
//                 std::vector<cv::DMatch> matches;               
               
//                 map_storage.LoopDescriptor.insert(std::pair<int, cv::Mat>(keyframe_num, descriptors));
//                 map_storage.LoopKeyPoint.insert(std::pair<int, std::vector<cv::KeyPoint>>(keyframe_num, keypoints));
                
//                 // ORB LOOP MAP
//                 // matcher->match(MapST.LoopDescriptor[keyframe_num - 1], MapST.LoopDescriptor[keyframe_num], matches);
//                 // for(int i = 0; i < matches.size(); i++)
//                 // {
//                 //     map_storage.LoopPoint2f[keyframe_num - 1].push_back(MapST.LoopKeyPoint[keyframe_num - 1][matches[i].queryIdx].pt);
//                 //     map_storage.LoopPoint2f[keyframe_num].push_back(MapST.LoopKeyPoint[keyframe_num][matches[i].trainIdx].pt);
//                 // }
//                 // cv::Mat E = cv::findEssentialMat(map_storage.LoopPoint2f[keyframe_num - 1], map_storage.LoopPoint2f[keyframe_num], f, c, cv::RANSAC, 0.99, 1, inlier_mask);
                
//                 // for(int i = 0; i < inlier_mask.rows; i++)
//                 // {
                
//                 //     if(inlier_mask.at<bool>(i, 0) == 1)
//                 //     {
//                 //         LoopPoint_.push_back(LoopPoint[i]);
//                 //         BeforeLoopPoint_.push_back(BeforeLoopPoint[i]);
//                 //     }
//                 // }
                
//                 // P0 = K_ * cam_storage_to_projection_matrix(map_storage.keyframe[loop_edge_id].cam_pose);
//                 // P1 = K_ * cam_storage_to_projection_matrix(map_storage.keyframe[loop_edge_id - 1].cam_pose);     
//                 // cv::triangulatePoints(P0, P1, LoopPoint_, BeforeLoopPoint_, X);

//                 // world_xyz_point_to_homogenous(X);
//                 // X.convertTo(X, CV_64F);
//                 // map_point_.clear();
//                 // for (int i = 0; i < LoopPoint_.size(); i++ ) map_point_.push_back(Point3d(X.at<double>(0, i), X.at<double>(1, i), X.at<double>(2, i)));

                
                 

//                 QueryResults ret;
//                 ret.clear();
//                 int max_entry_id;
//                 if(keyframe_num > 30) 
//                 {
//                     db.query(features[0], ret, 20);
// cout << "Searching for Image " << keyframe_num << ". " << ret << endl;
                                       

//                     // loop detect condition

//                     // delete nearest frame
//                     int index_correction = 0;
//                     int clone_ret_size = ret.size();
//                     for(int i = 0; i < clone_ret_size; i++)
//                     {
                    
//                         if(ret[i - index_correction].Id > keyframe_num - 10)
//                         { 
//                             ret.erase(ret.begin() + i - index_correction);
//                             index_correction++;
//                         }
//                     }   
// std::cout << "High score keyframe  num : "  << ret[0].Id << "       Score : " << ret[0].Score << std::endl;

                    
//                     if(loop_KF_threshold > 15)
//                     { 
//                         if(ret[0].Score > 0.36) 
//                         {
//                             loop_detect = true;
//                             loop_KF_threshold = 0;
//                             // cv::waitKey();
//                         }
//                     }    
//                         loop_KF_threshold ++;
                        
//                     if(loop_detect)
//                     {   



//                         loop_detect_frame_id = ret[0].Id;
//                         show_loop_detect_line(map_storage, loop_detect_frame_id, keyframe_num, 0.0f, 1.0f, 0.0f, 1.3);



//                         // add vertex to posegraph sim3
//                         for(int i = made_vertex_id; i < keyframe_num; i++)
//                         {
//                             // g2o::VertexSE3* v_se3 = static_cast<g2o::VertexSE3*>(optimizer->vertex(i));
//                             g2o::VertexSim3Expmap* v_sim3 = new g2o::VertexSim3Expmap();
//                             v_sim3->setId(i + 1);
//                             v_sim3->setMarginalized(false);
//                             cv::Mat cam_pose4d = vec6d_to_homogenous_campose(map_storage.keyframe[i + 1].cam_pose);
//                             cam_pose4d = cam_pose4d.inv();
//                             Eigen::Matrix3d sim3_r = RotationMatToEigen3d(homogenous_campose_to_R(cam_pose4d));
//                             Eigen::Vector3d sim3_t(homogenous_campose_to_t(cam_pose4d).at<double>(0, 0),
//                                                     homogenous_campose_to_t(cam_pose4d).at<double>(1, 0),
//                                                     homogenous_campose_to_t(cam_pose4d).at<double>(2, 0));
//                             g2o::Sim3 sim3(sim3_r, sim3_t, 1.0);
//                             v_sim3->setEstimate(sim3);

//                             v_sim3->setFixed(false);
                            
//                             optimizer_sim3.addVertex(v_sim3);
//                             // std::cout << " Add " << i + 1 << " vertex " << std::endl;
//                         }

                            
//                         // add edge to posegraph sim3
//                         int edge_index = 0;
//                         for(int j = 0; j < 2; j++)
//                         {
//                             for (int i = made_vertex_id; i < keyframe_num; i++)
//                             {    
//                                 // g2o::SE3Quat relpose;
//                                 int idx0 = i - j;
//                                 int idx1 = i + 1;
//                                 if(idx0 < 0) continue;
//                                 cv::Mat relpose = vec6d_to_homogenous_campose(map_storage.keyframe[idx0].cam_pose).inv() * vec6d_to_homogenous_campose(map_storage.keyframe[idx1].cam_pose);
//                                 relpose = relpose.inv();
//                                 Eigen::Matrix3d sim3_r;
//                                 sim3_r <<   relpose.at<double>(0, 0), relpose.at<double>(0, 1), relpose.at<double>(0, 2),
//                                             relpose.at<double>(1, 0), relpose.at<double>(1, 1), relpose.at<double>(1, 2),
//                                             relpose.at<double>(2, 0), relpose.at<double>(2, 1), relpose.at<double>(2, 2);
//                                 Eigen::Vector3d sim3_t(relpose.at<double>(0, 3), relpose.at<double>(1, 3), relpose.at<double>(2, 3));
                                                    
                                
//                                 g2o::EdgeSim3* e_sim3 = new g2o::EdgeSim3();
//                                 g2o::Sim3 sim3(sim3_r, sim3_t, 1.0);
//                                 e_sim3->setMeasurement(sim3);
                               
//                                 // addEdgePosePose(optimizer, i , i + 1, relpose);
                                
//                                 e_sim3->setId(edge_index++);
//                                 e_sim3->setVertex(0, optimizer_sim3.vertices()[idx0]);
//                                 e_sim3->setVertex(1, optimizer_sim3.vertices()[idx1]);
//                                 e_sim3->information() = Eigen::Matrix<double, 7, 7>::Identity();
//                                 // std::cout << "id0 : " << idx0 << "      " << "id1 : " << idx1 << std::endl;

//                                 optimizer_sim3.addEdge(e_sim3);
//                                 // std::cout << idx0 << "  edge  " << idx1 << std::endl;
//                             }
//                         }
//                         // std::cout << " Add " glPointSize_num;
//                         made_vertex_id = keyframe_num;
//                         // std::cout << " edge index : " << edge_index << std::endl;

//                         cv::Mat relpose_, t;
//                         double scale = VerifyLoop(loop_detect_frame_id, keyframe_num, map_storage, K, relpose_);
                        
//                         // cv::waitKey();
//                         // double scale = FindLoopEdgeScale(loop_detect_frame_id, keyframe_num, map_storage, K, inlier_storage, relpose_);                            
//                         if(scale == 1.0 or scale < 0.3 or scale >2.0)
//                         {
//                             loop_KF_threshold = 16;
//                             GoodLoopEdge = false;
//                         }
//                         if(GoodLoopEdge)
//                         {
//                             Eigen::Matrix3d loop_sim3_r;
//                             loop_sim3_r <<   relpose_.at<double>(0, 0), relpose_.at<double>(0, 1), relpose_.at<double>(0, 2),
//                                                 relpose_.at<double>(1, 0), relpose_.at<double>(1, 1), relpose_.at<double>(1, 2),
//                                                 relpose_.at<double>(2, 0), relpose_.at<double>(2, 1), relpose_.at<double>(2, 2);
//                             Eigen::Vector3d loop_sim3_t(relpose_.at<double>(0, 3), relpose_.at<double>(1, 3), relpose_.at<double>(2, 3));
//                                 // add loop adge
//                                 int idx0 = keyframe_num;
//                                 int idx1 = loop_detect_frame_id;
//                                 // cv::Mat relpose = vec6d_to_homogenous_campose(map_storage.keyframe[idx0].cam_pose).inverse() * vec6d_to_homogenous_campose(map_storage.keyframe[idx1].cam_pose);
                                    
//                                 // Eigen::Matrix3d sim3_r <<   relpose.at<double>(0, 0), relpose.at<double>(0, 1), relpose.at<double>(0, 2),
//                                 //                                 relpose.at<double>(1, 0), relpose.at<double>(1, 1), relpose.at<double>(1, 2),
//                                 //                                 relpose.at<double>(2, 0), relpose.at<double>(2, 1), relpose.at<double>(2, 2);
//                                 //     Eigen::Vector3d sim3_t(relpose.at<double>(0, 3), relpose.at<double>(1, 3), relpose.at<double>(2, 3));
                                                        
                                    
//                                     g2o::EdgeSim3* e_sim3 = new g2o::EdgeSim3();
//                                     // Eigen::Matrix3d loop_sim3_r = Eigen::Matrix<double, 3, 3>::Identity();
//                                     // Eigen::Vector3d loop_sim3_t(0, 0, 0);
//                                     g2o::Sim3 sim3(loop_sim3_r, loop_sim3_t, scale);
//                                     e_sim3->setMeasurement(sim3);
                                
//                                     // addEdgePosePose(optimizer, i , i + 1, relpose);
                                    
//                                     e_sim3->setId(edge_index++);
//                                     e_sim3->setVertex(0, optimizer_sim3.vertices()[idx0]);
//                                     e_sim3->setVertex(1, optimizer_sim3.vertices()[idx1]);
//                                     e_sim3->information() = Eigen::Matrix<double, 7, 7>::Identity();
//                                     // std::cout << "!!! add loop edge !!!  id0 : " << idx0 << "      " << "id1 : " << idx1 << std::endl;

//                                     optimizer_sim3.addEdge(e_sim3);
                                
//                             // }
                            
//                             std::cout << " edge num : " << optimizer_sim3.edges().size() << std::endl;
//                             // cv::waitKey();
//                         }
                        
//                         GoodLoopEdge = true;

//                         std::cout << " Start optimize PGO " << std::endl;
//                         bool initialize_bool = optimizer_sim3.initializeOptimization();
//                         std::cout << initialize_bool << std::endl;
//                         optimizer_sim3.optimize(100);
//                         std::cout << " End optimize PGO " << std::endl;

//                         // Recover Landmark
//                         for(auto tmp : map_storage.world_xyz)
//                         {
//                             int iter = map_storage.MapToKF_ids[tmp.first].size() - 1; 
//                             int KFid = map_storage.MapToKF_ids[tmp.first][0];
//                             Eigen::Vector4d old_map;
//                             old_map << tmp.second.x, tmp.second.y, tmp.second.z, 1;
//                             cv::Mat MMatrix_ = vec6d_to_homogenous_campose(map_storage.keyframe[KFid].cam_pose);
//                             MMatrix_ = MMatrix_.inv();
//                             Eigen::Matrix4d old_PMatrix = Mat44dToEigen44d(MMatrix_);

//                             g2o::VertexSim3Expmap* vtx = static_cast<g2o::VertexSim3Expmap*>(optimizer_sim3.vertex(KFid));
//                             g2o::Sim3 sim3 = vtx->estimate();
//                             Eigen::Matrix3d r_ = sim3.rotation().toRotationMatrix();
//                             Eigen::Vector3d t_ = sim3.translation();
//                             double s = sim3.scale();
//                             t_ *= (1./s);
                            
//                             Eigen::Matrix4d new_MMatrix = RtToEigen44Md(r_, t_);

//                             Eigen::Vector4d new_map = new_MMatrix.inverse() * old_PMatrix * old_map;


//                             map_storage.world_xyz[tmp.first].x = new_map[0];
//                             map_storage.world_xyz[tmp.first].y = new_map[1];
//                             map_storage.world_xyz[tmp.first].z = new_map[2];
//                         }
                        
                        
//                         for(int i = 0; i < keyframe_num + 1; i++)
//                         {
//                             g2o::VertexSim3Expmap* vtx = static_cast<g2o::VertexSim3Expmap*>(optimizer_sim3.vertex(i));
//                             g2o::Sim3 sim3 = vtx->estimate();

//                             double s = sim3.scale();
//                             vec_scale.push_back(s);
//                             // std::cout << i << " Scale : " << s << std::endl;
//                             // g2o::VertexSE3* vtx = static_cast<g2o::VertexSE3*>(optimizer->vertex(i));
//                             // g2o::Isometry3 se3 = vtx->estimate();
//                             // // std::cout << "se3" << se3 << std::endl;
//                             Eigen::Matrix3d r_ = sim3.rotation().toRotationMatrix();
//                             Eigen::Vector3d t_ = sim3.translation();
//                             t_ *= (1./s);
//                             // r_ = r_.transposec;
//                             // Eigen::Vector3d t_traj = -r_ * t_;
//                             cv::Mat rot = Eigen3dToRotationMat(r_);
//                             cv::Rodrigues(rot, rot);
//                             map_storage.keyframe[i].cam_pose[0] = rot.at<double>(0, 0);
//                             map_storage.keyframe[i].cam_pose[1] = rot.at<double>(1, 0);
//                             map_storage.keyframe[i].cam_pose[2] = rot.at<double>(2, 0);
//                             map_storage.keyframe[i].cam_pose[3] = t_[0];
//                             map_storage.keyframe[i].cam_pose[4] = t_[1];
//                             map_storage.keyframe[i].cam_pose[5] = t_[2];
//                             // std::cout << "rot" << std::endl << rot << std::endl;
//                             // std::cout << "t_" << std::endl << t_ << std::endl;
//                         }
//                             std::cout << " Scale : " << scale << std::endl;
//                             std::cout << "relpose" << std::endl << relpose_.inv() << std::endl;
//                             vec_pose.clear();
//                             // cv::waitKey();
                        
                  

                        
//                         ceres::Problem global_BA; 
//                         for(int j = 0; j < keyframe_num + 1; j++)
//                         {
//                             for ( int i = 0; i < inlier_storage[j].rows; i++)
//                             {
//                                 ceres::CostFunction* map_only_cost_func = map_point_only_ReprojectionError::create(map_storage.keyframe[j].pts[inlier_storage[j].at<int>(i, 0)], map_storage.keyframe[j].cam_pose, f, cv::Point2d(c.x, c.y));
//                                 int id_ = map_storage.keyframe[j].pts_id[inlier_storage[j].at<int>(i, 0)];
//                                 double* X_ = (double*)(&(map_storage.world_xyz[id_]));
//                                 global_BA.AddResidualBlock(map_only_cost_func, NULL, X_); 
                                        
//                             } 
//                         }
//                         ceres::Solver::Options options;
//                         options.linear_solver_type = ceres::ITERATIVE_SCHUR;
//                         options.num_threads = 8;
//                         options.minimizer_progress_to_stdout = false;
//                         ceres::Solver::Summary summary;
//                         std::cout << " Start optimize map point " << std::endl;
//                         ceres::Solve(options, &global_BA, &summary);                
//                         std::cout << " End optimize map point " << std::endl;
//                     }    
//                 }
//                 vec_scale.clear();
//                 db.add(features[0]);
// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////                
                
                // new feature to track
                std::cout << " New Track Feature " << endl;
                cv::goodFeaturesToTrack(current_image.frame, CurrTrackPts, max_keypoint, 0.01, 10);
                PrevTriangulateTrackPts.clear();
                PrevTriangulateTrackPts = CurrTrackPts;
std::cout << " New Track feature num  : " << PrevTriangulateTrackPts.size() << endl; 
                track_entire_num = PrevTriangulateTrackPts.size();
                    
                // Correspondance New Feature
                TrackForTriangulatePtsID.clear();
                std::vector<cv::Point2f> clone_pts(current_image.TrackPts);
                std::vector<int> clone_pts_id(current_image.TrackId);
                int same_point_num(0), different_point_num(0);

                for(int j = 0; j < CurrTrackPts.size(); j++)
                {
                    bool DifferentPts = true;
                    for(int i = 0; i < clone_pts.size(); i++)
                    {
                        double difference_point2d_x = cv::abs(clone_pts[i].x - CurrTrackPts[j].x);
                        double difference_point2d_y = cv::abs(clone_pts[i].y - CurrTrackPts[j].y);

                        if(difference_point2d_x < 3.0 and difference_point2d_y < 3.0)
                        {
                            TrackForTriangulatePtsID.push_back(clone_pts_id[i]);
                            clone_pts.erase(clone_pts.begin() + i);
                            clone_pts_id.erase(clone_pts_id.begin() + i);
                            same_point_num++;
                            DifferentPts = false;
                            break;
                        }
                    }
                    
                    if(DifferentPts)
                    {
                        TrackForTriangulatePtsID.push_back(TrackFeatureMaxID);
                        different_point_num++;
                        TrackFeatureMaxID++;
                    }

                }
std::cout << " Track same point :  " << same_point_num << " Track different point num  : " << different_point_num << endl;

                PrevGoodMatches.clear();
                PrevKeypoints.clear();
                PrevDescriptors.release();
                PrevMask.release();


                PrevGoodMatches.assign( CurrGoodMatches.begin(), CurrGoodMatches.end() );
                PrevKeypoints.assign( CurrKeypoints.begin(), CurrKeypoints.end() );
                PrevDescriptors = CurrDescriptors.clone();
                PrevMask = CurrMask.clone();
            }

//             previous_track_point_for_triangulate.clear();
//             for(int i = 0; i < current_track_point_for_triangulate.size(); i++) previous_track_point_for_triangulate.push_back(current_track_point_for_triangulate[i]);

//             // Reprojection 3D to 2D current image
//             std::cout << "reprojection 3D to 2D at SolvePnP stage " << endl;
//             if(times > initialize_frame_num) 
//             {
//                 cv::Mat project_mat = Mat(map_point).clone();
//                 project_mat.convertTo(project_mat, CV_32F);
//                 cv::projectPoints(project_mat, World_R, World_t, K, cv::noArray(), projectionpoints); 
//             }
            

        }



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////               SHOW IMAGE                ////////////////////////////////////////////////////////////      
        // std::cout << "show image" << endl;
        // // Show image and Matching points and reprojection points
        // cv::Mat show_image = current_image.frame;
        // if (show_image.channels() < 3) cv::cvtColor(show_image, show_image, cv::COLOR_GRAY2RGB);
        // for(int a = 0; a < current_image.pts.size(); a++) circle(show_image, current_image.pts[a], 3, Scalar(0,255,0), 1);
        // if(times >= initialize_frame_num) for(int a = 0; a < map_point.size(); a++) circle(show_image, projectionpoints[a], 3, Scalar(0,0,255), 1);
        
        // // Put text to image about camera_pose
        // cv::Mat campose_vec6d_to_mat_for_text = vec6d_to_homogenous_campose(current_image.cam_pose);
        // cv::String info = cv::format(" XYZ: [%.3lf, %.3lf, %.3lf]", campose_vec6d_to_mat_for_text.at<double>(0, 3), campose_vec6d_to_mat_for_text.at<double>(1, 3), campose_vec6d_to_mat_for_text.at<double>(2, 3));
        // cv::putText(show_image, info, cv::Point(5, 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Vec3b(255, 0, 0));
        // cv::imshow("image", show_image);

            
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////            VISUALIZE         ////////////////////////////////////////////////////////////////
        

        
        // Show gt trajectory (red line)
        GLfloat GT_Pose[12];
        for(int j=0;j<12;j++) GT_DATA >> GT_Pose[j];

        // show_trajectory_right_mini(x_gt, y_gt, z_gt, 1.0, 0.0, 0.0, 3.0);
        // show_trajectory_left_mini(x_gt, y_gt, z_gt, 1.0, 0.0, 0.0, 3.0);


        // show gt 
        cv::Mat_<double> gt_pose(3, 4);
        gt_pose << GT_Pose[0], GT_Pose[1], GT_Pose[2], GT_Pose[3], GT_Pose[4], GT_Pose[5], GT_Pose[6], GT_Pose[7], GT_Pose[8], GT_Pose[9], GT_Pose[10], GT_Pose[11];
        std::cout << " current GT pose  : "  << endl << gt_pose << endl;        
        GT_Storage.push_back(gt_pose);
        for(int i = 0 ; i < GT_Storage.size(); i++)
        {
            GLfloat x_gt(GT_Storage[i].at<double>(0, 3)), y_gt(GT_Storage[i].at<double>(1, 3)), z_gt(GT_Storage[i].at<double>(2, 3));
            show_trajectory(x_gt, y_gt, z_gt, 1.0, 0.0, 0.0, 3.0);
        }
        // Show camera estimate pose trajectory_motion_only_ba ( green line )
        cv::Mat campose_vec6d_to_mat_for_visualize = vec6d_to_homogenous_campose(current_image.cam_pose);
        GLdouble _x_cam_pose(campose_vec6d_to_mat_for_visualize.at<double>(0, 3)), _y_cam_pose(campose_vec6d_to_mat_for_visualize.at<double>(1, 3)), _z_cam_pose(campose_vec6d_to_mat_for_visualize.at<double>(2, 3));         
        // show_trajectory(_x_cam_pose, _y_cam_pose, _z_cam_pose, 0.0, 1.0, 0.0, 3.0);
        // show_trajectory_right_mini(_x_cam_pose, _y_cam_pose, _z_cam_pose, 0.0, 1.0, 0.0, 3.0);



        // show camera estimate pose trajectory keyframe after ba ( blue triangle )
        if (new_keyframe_selection)
        { 
               


//                 for(int i = 0; i < keyframe_num; i++)
//                 {
//                     cv::Mat campose_vec6d_to_mat_for_visualize = vec6d_to_homogenous_campose(map_storage.keyframe[i].cam_pose);
//                     cv::Mat rb_t_ = homogenous_campose_for_keyframe_visualize(campose_vec6d_to_mat_for_visualize, 8.0); // size
//                     show_trajectory_keyframe(rb_t_, 0.0, 0.0, 1.0, 1.0);
//                 }
//                 cv::Mat campose_vec6d_to_mat_for_visualize = vec6d_to_homogenous_campose(map_storage.keyframe[keyframe_num].cam_pose);
//                 cv::Mat rb_t_ = homogenous_campose_for_keyframe_visualize(campose_vec6d_to_mat_for_visualize, 8.0);
//                 show_trajectory_keyframe(rb_t_, 0.0, 1.0, 0.0, 1.0);
//                 if(fix_keyframe_parms - fix_keyframe_num == fix_keyframe_num) fix_keyframe_parms++;
            for(int i = 0; i < keyframe_num; i++)
            {
                cv::Mat campose_vec6d_to_mat_for_visualize = vec6d_to_homogenous_campose(map_storage.keyframe[i].cam_pose);
                cv::Mat rb_t_ = homogenous_campose_for_keyframe_visualize(campose_vec6d_to_mat_for_visualize, 8.0); // size
                show_trajectory_keyframe(rb_t_, 0.0, 0.0, 1.0, 1.0);
            }
            cv::Mat campose_vec6d_to_mat_for_visualize = vec6d_to_homogenous_campose(map_storage.keyframe[keyframe_num].cam_pose);
            cv::Mat rb_t_ = homogenous_campose_for_keyframe_visualize(campose_vec6d_to_mat_for_visualize, 8.0);
            show_trajectory_keyframe(rb_t_, 0.0, 1.0, 0.0, 1.0);
        }
    

//         // Show 3d keyframe map point ( black dot )
        if (times == initialize_frame_num)
        {
//             for( int i = 0 ; i < map_storage.keyframe[0].pts_id.size(); i++)
//             {
//                 int show_map_id = map_storage.keyframe[0].pts_id[i];
//                 GLdouble X_map(map_storage.world_xyz[show_map_id].x), Y_map(map_storage.world_xyz[show_map_id].y), Z_map(map_storage.world_xyz[show_map_id].z);
//                 // show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 0.0, 0.01);
//                 show_map_point_parms = map_storage.world_xyz.size();
//             }  
//             show_map_point_parms = map_storage.world_xyz.size();
// std::cout << " map_storage size is : " << map_storage.world_xyz.size() << endl;    
            for(int i = 0 ; i < map_storage.world_xyz.size(); i++)
            {
                GLdouble X_map(map_storage.world_xyz[i].x), Y_map(map_storage.world_xyz[i].y), Z_map(map_storage.world_xyz[i].z);
                show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 0.0, 0.01);            
            }   
        }
            

        // storage inlier map point 
        if (new_keyframe_selection)
        { 
//             inlier_map_storage.clear();
//             map_storage.InlierID.clear();
//             for(int j = 0; j < keyframe_num + 1; j ++)
//             {
//                 for(int i = 0; i < inlier_storage[j].rows; i++)
//                 {
//                     int id_ = map_storage.keyframe[j].pts_id[inlier_storage[j].at<int>(i, 0)];
//                     inlier_map_storage.push_back(map_storage.world_xyz[id_]);
//                     map_storage.InlierID.push_back(id_);
//                 }
//             }
// std::cout << " Inlier Map_Storage size is  : " << inlier_map_storage.size() << std::endl;
//             for(int i = 0 ; i < inlier_map_storage.size(); i++)
//             {
//                 GLdouble X_map(inlier_map_storage[i].x), Y_map(inlier_map_storage[i].y), Z_map(inlier_map_storage[i].z);
//                 show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 0.0, 0.01);            
//             }
            for(int i = 0 ; i < map_storage.world_xyz.size(); i++)
            {
                GLdouble X_map(map_storage.world_xyz[i].x), Y_map(map_storage.world_xyz[i].y), Z_map(map_storage.world_xyz[i].z);
                show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 0.0, 0.01);            
            }  
        }


        glFlush();

//         // glClearColor(1.0,1.0,1.0, 1.0);
        if (new_keyframe_selection) {
            new_keyframe_selection = false;
            // cv::waitKey();
        }
        
//         if(loop_detect)
//         {
//             // cv::waitKey();
 
//             loop_detect = false;
//         }

        cv::imshow("currentimage", current_image.frame);
        
        PrevTrackPts.clear();
        PrevTrackPts = CurrTrackPts;
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
    sort(map_storage.InlierID.begin(), map_storage.InlierID.end());
    map_storage.InlierID.erase(unique(map_storage.InlierID.begin(),map_storage.InlierID.end()),map_storage.InlierID.end());
    
    for(int i = 0; i < map_storage.InlierID.size(); i++)
        map_storage.InlierMap.push_back(map_storage.world_xyz[map_storage.InlierID[i]]);

    // for(int i = 0; i < map_storage.InlierID.size(); i++) std::cout << map_storage.InlierID[i] << "      ";
    std::cout << "  Keyframe num : " << map_storage.keyframe.size() << std::endl;
    std::cout << " Inlier Map Storage num : " << map_storage.InlierID.size() << "   " << map_storage.InlierMap.size() << std::endl;
    std::cout << " world_xyz num ( all 3d point ) : " << map_storage.world_xyz.size() << std::endl;
    
    std::cout << inlier_storage.size() << std::endl;
    


    // Map Compression
std::cout << "Map Compression ... " << std::endl;
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);
    int PointCloudNum = map_storage.InlierMap.size();
    
std::cout << " Create Variables ... " << std:: endl;
    // Create Variables
    std::vector<GRBVar> x = CreateVariablesBinaryVector(PointCloudNum, model);

std::cout << " Set Objective ... " << std:: endl;
    // Set Objective
    Eigen::Matrix<double, Eigen::Dynamic, 1> q = CalculateObservationCountWeight(map_storage);
    SetObjectiveILP(x, q, model);

std::cout << " Add Constraint ... " << std:: endl;    
    // Add Constraint
    Eigen::MatrixXd A =CalculateVisibilityMatrix(map_storage, inlier_storage);
    AddConstraint(map_storage, model, A, x);

std::cout << " Optimize model ... " << std:: endl;
    // Optimize model
    model.optimize();

    std::cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
    // cout << " x : "; 
    // for(int i = 0; i < x.size(); i++) 
    // {
    //     std::cout   << x[i].get(GRB_DoubleAttr_X) << "  ";
    // }
    // std::cout << std::endl;


    // Compression Map Visialize
    for(int i = 0; i < x.size(); i++)
    {
        if(x[i].get(GRB_DoubleAttr_X) == 1)
        {
            map_storage.CompressionMap.push_back(map_storage.InlierMap[i]);
        }
        
    }
    std::cout << " Remain Map point num : " << map_storage.CompressionMap.size() << std::endl;
    cv::waitKey();

    glClear(GL_COLOR_BUFFER_BIT);

    for(int i = 0 ; i < map_storage.CompressionMap.size(); i++)
    {
        GLdouble X_map(map_storage.CompressionMap[i].x), Y_map(map_storage.CompressionMap[i].y), Z_map(map_storage.CompressionMap[i].z);
        show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 0.0, 0.01);            
    }
    
    for(int i = 0; i < keyframe_num; i++)
    {
        cv::Mat campose_vec6d_to_mat_for_visualize = vec6d_to_homogenous_campose(map_storage.keyframe[i].cam_pose);
        cv::Mat rb_t_ = homogenous_campose_for_keyframe_visualize(campose_vec6d_to_mat_for_visualize, 8.0); // size
        show_trajectory_keyframe(rb_t_, 0.0, 0.0, 1.0, 1.0);
    }
    cv::Mat campose_vec6d_to_mat_for_visualize = vec6d_to_homogenous_campose(map_storage.keyframe[keyframe_num].cam_pose);
    cv::Mat rb_t_ = homogenous_campose_for_keyframe_visualize(campose_vec6d_to_mat_for_visualize, 8.0);
    show_trajectory_keyframe(rb_t_, 0.0, 1.0, 0.0, 1.0);
    glFlush();


    cv::waitKey();
    return 0;
}
