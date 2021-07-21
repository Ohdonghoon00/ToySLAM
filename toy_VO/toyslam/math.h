#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/sfm/projection.hpp"
#include "Eigen/Dense"
#include "types/Map.h"

void print_map(std::map<int, cv::Point3d>& m);


cv::Mat R_t_to_homogenous(cv::Mat r, cv::Mat t);


cv::Mat homogenous_campose_to_R(cv::Mat h);


cv::Mat homogenous_campose_to_t(cv::Mat h);


cv::Mat homogenous_campose_for_keyframe_visualize(cv::Mat h, double size);



void world_xyz_point_to_homogenous(cv::Mat &X_);


cv::Vec6d homogenous_campose_to_vec6d(cv::Mat cam);


cv::Mat vec6d_to_homogenous_campose(cv::Vec6d Rt_cam);


cv::Mat cam_storage_to_projection_matrix(cv::Vec6d cam_storage);


void track_opticalflow_and_remove_err(cv::Mat &previous_, cv::Mat &current_, std::vector<cv::Point2f> &previous_pts_, std::vector<cv::Point2f> &current_pts_);


void track_opticalflow_and_remove_err_for_triangulate(cv::Mat &previous_, cv::Mat &current_, std::vector<cv::Point2f> &previous_pts_, std::vector<cv::Point2f> &current_pts_, std::vector<cv::Point2f> &keyframe_track_point_);


void track_opticalflow_and_remove_err_for_triangulate_(cv::Mat &previous_, cv::Mat &current_, std::vector<cv::Point2f> &previous_pts_, std::vector<cv::Point2f> &current_pts_, std::vector<int> &previous_track_point_for_triangulate_ID_, std::vector<cv::Point2f> &keyframe_track_point_);


void track_opticalflow_and_remove_err_for_SolvePnP(cv::Mat &previous_, cv::Mat &current_, std::vector<cv::Point2f> &previous_pts_, std::vector<cv::Point2f> &current_pts_, std::vector<int> &previous_pts_id_, std::vector<cv::Point3d> &map_point_);


void remove_map_point_and_2dpoint_outlier (std::vector<cv::Point3d> &map_point_, std::vector<cv::Point2f> &current_pts_, cv::Mat &current_cam_pose_);


void remove_map_point_and_2dpoint_outlier_(std::vector<cv::Point3d> &map_point_, std::vector<cv::Point2f> &current_pts_, std::vector<int> &previous_track_point_for_triangulate_ID_, cv::Mat &current_cam_pose_);


void remove_SolvePnP_oulier (std::vector<cv::Point3d> &map_point_, std::vector<cv::Point2f> &current_pts_, cv::Mat inliers_index);


void remove_SolvePnP_oulier_ (std::vector<cv::Point3d> &map_point_, std::vector<cv::Point2f> &current_pts_, std::vector<int> &previous_track_point_for_triangulate_ID_, cv::Mat inliers_index);

void changeStructure(const cv::Mat &plain, std::vector<cv::Mat> &out);

Eigen::Matrix3d RotationMatToEigen3d(cv::Mat r_);

cv::Mat Eigen3dToRotationMat(Eigen::Matrix3d ae);

Eigen::Vector3d PoseToEigen3d(Map &conversion_pose, int num);

Eigen::Quaterniond getQuaternionFromRotationMatrix(const Eigen::Matrix3d& mat);
// Eigen::Quaterniond RotationToQuan(Map &conversion_rot)
// {
//     Eigen::Quaterniond abc;

// }