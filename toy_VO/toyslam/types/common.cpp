

// #include "opencv2/opencv.hpp"
// #include "opencv2/sfm/projection.hpp"

#include "common.h"


void print_map(std::map<int, cv::Point3d>& m) 
{
    for (std::map<int, cv::Point3d>::iterator itr = m.begin(); itr != m.end(); ++itr) 
    {
        std::cout << itr->first << " " << itr->second << "     ";
    }
}

cv::Mat R_t_to_homogenous(cv::Mat r, cv::Mat t)
{
    cv::Mat Iden = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat rt;
    cv::hconcat(r, t, rt);
    cv::vconcat(rt, Iden.row(3), rt);
    
    return rt;
}

cv::Mat homogenous_campose_to_R(cv::Mat h)
{
    cv::Mat a = (cv::Mat_<double>(3, 3) <<  h.at<double>(0, 0), h.at<double>(0, 1), h.at<double>(0, 2),
                                            h.at<double>(1, 0), h.at<double>(1, 1), h.at<double>(1, 2),
                                            h.at<double>(2, 0), h.at<double>(2, 1), h.at<double>(2, 2));
    
    return a;
}

cv::Mat homogenous_campose_to_t(cv::Mat h)
{
    cv::Mat a = (cv::Mat_<double>(3, 1) <<  h.at<double>(0, 3), 
                                            h.at<double>(1, 3),
                                            h.at<double>(2, 3));
    
    return a;
}

cv::Mat homogenous_campose_for_keyframe_visualize(cv::Mat h, double size)
{
    cv::Mat r = homogenous_campose_to_R(h);
    cv::Mat t = homogenous_campose_to_t(h);
    cv::Mat b = (cv::Mat_<double>(3, 3) <<      0,  -size,   size,
                                                0,      0,      0,
                                                0,  -size,  -size);
    
    cv::Mat rb = r * b;
    rb.col(0) = rb.col(0) + t;
    rb.col(1) = rb.col(1) + t;
    rb.col(2) = rb.col(2) + t;
    // cv::Mat rb_t = R_t_to_homogenous(rb, t);
    
    return rb;

}

void world_xyz_point_to_homogenous(cv::Mat &X_)
{
    for (int i = 0 ; i < X_.cols; i++)
    {
        X_.col(i).row(0) = X_.col(i).row(0) / X_.col(i).row(3);
        X_.col(i).row(1) = X_.col(i).row(1) / X_.col(i).row(3);
        X_.col(i).row(2) = X_.col(i).row(2) / X_.col(i).row(3);
        X_.col(i).row(3) = 1;
    }
}

cv::Vec6d homogenous_campose_to_vec6d(cv::Mat cam)
{
    cv::Mat P = cam.inv(), K, R, t;
    P.resize(3);
    cv::sfm::KRtFromProjection(P, K, R, t);
    Rodrigues(R, R);
    cv::Vec6d Rt_cam_(R.at<double>(0, 0), R.at<double>(1, 0), R.at<double>(2, 0), t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0));
    
    return Rt_cam_;

} 

cv::Mat vec6d_to_homogenous_campose(cv::Vec6d Rt_cam)
{
    cv::Vec3d _rvec(Rt_cam[0], Rt_cam[1], Rt_cam[2]); 
    cv::Mat _t = (cv::Mat_<double>(3, 1) << Rt_cam[3], Rt_cam[4], Rt_cam[5]);
    cv::Mat _R;
    cv::Rodrigues(_rvec, _R);
    cv::Mat _Rt = R_t_to_homogenous(_R, _t);

    return _Rt.inv();
}

Eigen::Matrix4d Mat44dToEigen44d(cv::Mat a)
{
    Eigen::Matrix4d bb;
    bb <<   a.at<double>(0, 0), a.at<double>(0, 1), a.at<double>(0, 2), a.at<double>(0, 3),
            a.at<double>(1, 0), a.at<double>(1, 1), a.at<double>(1, 2), a.at<double>(1, 3),
            a.at<double>(2, 0), a.at<double>(2, 1), a.at<double>(2, 2), a.at<double>(2, 3),
            a.at<double>(3, 0), a.at<double>(3, 1), a.at<double>(3, 2), a.at<double>(3, 3);

    return bb;
}

Eigen::Matrix4d RtToEigen44Md(Eigen::Matrix3d a, Eigen::Vector3d b)
{
    Eigen::Matrix4d bb;
    bb <<   a(0, 0), a(0, 1), a(0, 2), b[0],
            a(1, 0), a(1, 1), a(1, 2), b[1],
            a(2, 0), a(2, 1), a(2, 2), b[2],
                  0,       0,       0,    1;
    return bb;
}

cv::Mat cam_storage_to_projection_matrix(cv::Vec6d cam_storage)
{
    cv::Mat P_ = vec6d_to_homogenous_campose(cam_storage);
    P_ = P_.inv();
    P_.resize(3);    
    return P_;

}

void track_opticalflow_and_remove_err(cv::Mat &previous_, cv::Mat &current_, std::vector<cv::Point2f> &previous_pts_, std::vector<cv::Point2f> &current_pts_)
{
    std::vector<uchar> status_;
    cv::Mat err_;
    // std::vector<int> correspond_id_;

    current_pts_.clear();

    cv::calcOpticalFlowPyrLK(previous_, current_, previous_pts_, current_pts_, status_, err_);

    const int image_x_size_ = previous_.cols;
    const int image_y_size_ = previous_.rows;

    // remove err point
    int indexCorrection = 0;

    for( int i=0; i<status_.size(); i++)
    {
        cv::Point2f pt = current_pts_.at(i- indexCorrection);
        if((pt.x < 0)||(pt.y < 0 )||(pt.x > image_x_size_)||(pt.y > image_y_size_)) status_[i] = 0;
        if (status_[i] == 0)	
        {
                    
                    previous_pts_.erase ( previous_pts_.begin() + i - indexCorrection);
                    // previous_track_point_for_triangulate_.erase(previous_track_point_for_triangulate_.begin() + i - indexCorrection);
                    // previous_pts_id_.erase( previous_pts_id_.begin() + i - indexCorrection);
                    current_pts_.erase (current_pts_.begin() + i - indexCorrection);
                    indexCorrection++;
        }

    }     



}

void track_opticalflow_and_remove_err_for_triangulate(cv::Mat &previous_, cv::Mat &current_, std::vector<cv::Point2f> &previous_pts_, std::vector<cv::Point2f> &current_pts_, std::vector<cv::Point2f> &keyframe_track_point_)
{
    std::vector<uchar> status_;
    cv::Mat err_;
    // std::vector<int> correspond_id_;

    current_pts_.clear();

    cv::calcOpticalFlowPyrLK(previous_, current_, previous_pts_, current_pts_, status_, err_);

    const int image_x_size_ = previous_.cols;
    const int image_y_size_ = previous_.rows;

    // remove err point
    int indexCorrection = 0;

    for( int i=0; i<status_.size(); i++)
    {
        cv::Point2f pt = current_pts_.at(i- indexCorrection);
        if((pt.x < 0)||(pt.y < 0 )||(pt.x > image_x_size_)||(pt.y > image_y_size_)) status_[i] = 0;
        if (status_[i] == 0)	
        {
                    
                    previous_pts_.erase ( previous_pts_.begin() + i - indexCorrection);
                    keyframe_track_point_.erase(keyframe_track_point_.begin() + i - indexCorrection);
                    // previous_track_point_for_triangulate_ID_.erase(previous_track_point_for_triangulate_ID_.begin() + i - indexCorrection);
                    current_pts_.erase (current_pts_.begin() + i - indexCorrection);
                    indexCorrection++;
        }

    }     



}

void track_opticalflow_and_remove_err_for_triangulate_(cv::Mat &previous_, cv::Mat &current_, std::vector<cv::Point2f> &previous_pts_, std::vector<cv::Point2f> &current_pts_, std::vector<int> &previous_track_point_for_triangulate_ID_, std::vector<cv::Point2f> &keyframe_track_point_)
{
    std::vector<uchar> status_;
    cv::Mat err_;
    // std::vector<int> correspond_id_;

    current_pts_.clear();

    cv::calcOpticalFlowPyrLK(previous_, current_, previous_pts_, current_pts_, status_, err_);

    const int image_x_size_ = previous_.cols;
    const int image_y_size_ = previous_.rows;

    // remove err point
    int indexCorrection = 0;

    for( int i=0; i<status_.size(); i++)
    {
        cv::Point2f pt = current_pts_.at(i- indexCorrection);
        if((pt.x < 0)||(pt.y < 0 )||(pt.x > image_x_size_)||(pt.y > image_y_size_)) status_[i] = 0;
        if (status_[i] == 0)	
        {
                    
                    previous_pts_.erase ( previous_pts_.begin() + i - indexCorrection);
                    keyframe_track_point_.erase(keyframe_track_point_.begin() + i - indexCorrection);
                    previous_track_point_for_triangulate_ID_.erase(previous_track_point_for_triangulate_ID_.begin() + i - indexCorrection);
                    current_pts_.erase (current_pts_.begin() + i - indexCorrection);
                    indexCorrection++;
        }

    }     



}

void track_opticalflow_and_remove_err_for_SolvePnP_(cv::Mat &previous_, cv::Mat &current_, std::vector<cv::Point2f> &previous_pts_, std::vector<cv::Point2f> &current_pts_, std::vector<int> &previous_pts_id_, std::vector<cv::Point3d> &map_point_)
{
    std::vector<uchar> status_;
    cv::Mat err_;
    // std::vector<int> correspond_id_;

    current_pts_.clear();

    cv::calcOpticalFlowPyrLK(previous_, current_, previous_pts_, current_pts_, status_, err_);

    const int image_x_size_ = previous_.cols;
    const int image_y_size_ = previous_.rows;

    // remove err point
    int indexCorrection = 0;

    for( int i=0; i<status_.size(); i++)
    {
        cv::Point2f pt = current_pts_.at(i- indexCorrection);
        if((pt.x < 0)||(pt.y < 0 )||(pt.x > image_x_size_)||(pt.y > image_y_size_)) status_[i] = 0;        
        if (status_[i] == 0)	
        {

                    previous_pts_.erase ( previous_pts_.begin() + i - indexCorrection);
                    previous_pts_id_.erase( previous_pts_id_.begin() + i - indexCorrection);
                    current_pts_.erase (current_pts_.begin() + i - indexCorrection);
                    map_point_.erase (map_point_.begin() + i - indexCorrection);
                    // previous_track_point_for_triangulate_.erase(previous_track_point_for_triangulate_.begin() + i - indexCorrection);
                    indexCorrection++;
        }
        // else 
        // {
        //     correspond_id_.push_back(i);
        // }

    }     

    

}

void track_opticalflow_and_remove_err_for_SolvePnP(cv::Mat &previous_, cv::Mat &current_, std::vector<cv::Point2f> &previous_pts_, std::vector<cv::Point2f> &current_pts_, std::vector<cv::Point3d> &map_point_)
{
    std::vector<uchar> status_;
    cv::Mat err_;
    // std::vector<int> correspond_id_;

    current_pts_.clear();

    cv::calcOpticalFlowPyrLK(previous_, current_, previous_pts_, current_pts_, status_, err_);

    const int image_x_size_ = previous_.cols;
    const int image_y_size_ = previous_.rows;

    // remove err point
    int indexCorrection = 0;

    for( int i=0; i<status_.size(); i++)
    {
        cv::Point2f pt = current_pts_.at(i- indexCorrection);
        if((pt.x < 0)||(pt.y < 0 )||(pt.x > image_x_size_)||(pt.y > image_y_size_)) status_[i] = 0;        
        if (status_[i] == 0)	
        {

                    previous_pts_.erase ( previous_pts_.begin() + i - indexCorrection);
                    current_pts_.erase (current_pts_.begin() + i - indexCorrection);
                    map_point_.erase (map_point_.begin() + i - indexCorrection);
                    indexCorrection++;
        }
        // else 
        // {
        //     correspond_id_.push_back(i);
        // }

    }     

    

}

void track_opticalflow_and_remove_err_for_SolvePnP_noid(cv::Mat &previous_, cv::Mat &current_, std::vector<cv::Point2f> &previous_pts_, std::vector<cv::Point2f> &current_pts_, std::vector<cv::Point3d> &map_point_)
{
    std::vector<uchar> status_;
    cv::Mat err_;
    // std::vector<int> correspond_id_;

    current_pts_.clear();

    cv::calcOpticalFlowPyrLK(previous_, current_, previous_pts_, current_pts_, status_, err_);

    const int image_x_size_ = previous_.cols;
    const int image_y_size_ = previous_.rows;

    // remove err point
    int indexCorrection = 0;

    for( int i=0; i<status_.size(); i++)
    {
        cv::Point2f pt = current_pts_.at(i- indexCorrection);
        if((pt.x < 0)||(pt.y < 0 )||(pt.x > image_x_size_)||(pt.y > image_y_size_)) status_[i] = 0;        
        if (status_[i] == 0)	
        {

                    previous_pts_.erase ( previous_pts_.begin() + i - indexCorrection);
                    
                    current_pts_.erase (current_pts_.begin() + i - indexCorrection);
                    map_point_.erase (map_point_.begin() + i - indexCorrection);
                    // previous_track_point_for_triangulate_.erase(previous_track_point_for_triangulate_.begin() + i - indexCorrection);
                    indexCorrection++;
        }
        // else 
        // {
        //     correspond_id_.push_back(i);
        // }

    }     

    

}

void remove_map_point_and_2dpoint_outlier (std::vector<cv::Point3d> &map_point_, std::vector<cv::Point2f> &current_pts_, cv::Mat current_cam_pose_)
{


    // Intial camera viewing vector
    cv::Mat initilize_viewing_camera = (cv::Mat_<double>(4, 1) << 0, 0, 1, 0);
    
    // Map point type change ( std::vector<cv::Point3d> -> N*3 Mat channel 1 )
    std::vector<cv::Point3d> clone_map_point(map_point_);
    // for(int i = 0; i < map_point_.size(); i++) clone_map_point.push_back(map_point_[i]);
    cv::Mat clone_map_point_Mat = cv::Mat(clone_map_point).reshape(1);   

    // Current camera viewing vector              
    cv::Mat current_viewing_camera  = current_cam_pose_ * initilize_viewing_camera;
    current_viewing_camera.resize(3);
    cv::Mat current_camera_pose_translation = (cv::Mat_<double>(3, 1) << current_cam_pose_.at<double>(0, 3), current_cam_pose_.at<double>(1, 3), current_cam_pose_.at<double>(2, 3));

    int indexCorrection = 0;
                for(int i = 0; i < clone_map_point.size(); i++)
                {

                    
                    clone_map_point_Mat.at<double>(i, 0) -= current_camera_pose_translation.at<double>(0, 0);
                    clone_map_point_Mat.at<double>(i, 1) -= current_camera_pose_translation.at<double>(1, 0);
                    clone_map_point_Mat.at<double>(i, 2) -= current_camera_pose_translation.at<double>(2, 0);
                    
                    cv::Mat vector_map_point = (cv::Mat_<double>(3, 1) << clone_map_point_Mat.at<double>(i, 0), clone_map_point_Mat.at<double>(i, 1), clone_map_point_Mat.at<double>(i, 2));
// std::cout << vector_map_point.dot(vector_map_point) << std::endl;      
            //  or vector_map_point.dot(vector_map_point) > 30000 
                    if(current_viewing_camera.dot(vector_map_point) < 0 or vector_map_point.dot(vector_map_point) > 30000)
                    {
                        
                        map_point_.erase(map_point_.begin() + i - indexCorrection);
                        current_pts_.erase(current_pts_.begin() + i - indexCorrection);
                        indexCorrection++;
                    
                    }
                
                }               
                    
}


void remove_map_point_and_2dpoint_outlier_(std::vector<cv::Point3d> &map_point_, std::vector<cv::Point2f> &current_pts_, std::vector<int> &previous_track_point_for_triangulate_ID_, cv::Mat current_cam_pose_)
{


    // Intial camera viewing vector
    cv::Mat initilize_viewing_camera = (cv::Mat_<double>(4, 1) << 0, 0, 1, 0);
    
    // Map point type change ( std::vector<cv::Point3d> -> N*3 Mat channel 1 )
    std::vector<cv::Point3d> clone_map_point(map_point_);
    std::vector<cv::Point2f> clone_current_pts_(current_pts_);
    std::vector<int> clone_previous_track_point_for_triangulate_ID_(previous_track_point_for_triangulate_ID_);
    // for(int i = 0; i < map_point_.size(); i++) clone_map_point.push_back(map_point_[i]);
    cv::Mat clone_map_point_Mat = cv::Mat(clone_map_point).reshape(1);   

    // Current camera viewing vector              
    cv::Mat current_viewing_camera  = current_cam_pose_ * initilize_viewing_camera;
    current_viewing_camera.resize(3);
    cv::Mat current_camera_pose_translation = (cv::Mat_<double>(3, 1) << current_cam_pose_.at<double>(0, 3), current_cam_pose_.at<double>(1, 3), current_cam_pose_.at<double>(2, 3));

    int indexCorrection = 0;
                for(int i = 0; i < clone_map_point.size(); i++)
                {

                    
                    clone_map_point_Mat.at<double>(i, 0) -= current_camera_pose_translation.at<double>(0, 0);
                    clone_map_point_Mat.at<double>(i, 1) -= current_camera_pose_translation.at<double>(1, 0);
                    clone_map_point_Mat.at<double>(i, 2) -= current_camera_pose_translation.at<double>(2, 0);
                    
                    cv::Mat vector_map_point = (cv::Mat_<double>(3, 1) << clone_map_point_Mat.at<double>(i, 0), clone_map_point_Mat.at<double>(i, 1), clone_map_point_Mat.at<double>(i, 2));
// std::cout << vector_map_point.dot(vector_map_point) << std::endl;      
            //  or vector_map_point.dot(vector_map_point) > 30000
                    if(current_viewing_camera.dot(vector_map_point) < 0 or vector_map_point.dot(vector_map_point) > 30000 )
                    {
                        
                        map_point_.erase(map_point_.begin() + i - indexCorrection);
                        current_pts_.erase(current_pts_.begin() + i - indexCorrection);
                        previous_track_point_for_triangulate_ID_.erase(previous_track_point_for_triangulate_ID_.begin() + i - indexCorrection);
                        indexCorrection++;
                    
                    }
                
                }               
    if (map_point_.size() < 10)
    {
        map_point_.clear();
        current_pts_.clear();
        previous_track_point_for_triangulate_ID_.clear();

        map_point_.assign(clone_map_point.begin(), clone_map_point.end());
        current_pts_.assign(clone_current_pts_.begin(), clone_current_pts_.end());
        previous_track_point_for_triangulate_ID_.assign(clone_previous_track_point_for_triangulate_ID_.begin(), clone_previous_track_point_for_triangulate_ID_.end());
    }                
}

void remove_SolvePnP_oulier (std::vector<cv::Point3d> &map_point_, std::vector<cv::Point2f> &current_pts_, cv::Mat inliers_index)
{
    std::vector<cv::Point3d> clone_map_point(map_point_);
    std::vector<cv::Point2f> clone_current_pts(current_pts_);
    map_point_.clear();
    current_pts_.clear();
    for (int i = 0; i < inliers_index.rows;  i++)
    {
        map_point_.push_back(clone_map_point[inliers_index.at<int>(i, 0)]);
        current_pts_.push_back(clone_current_pts[inliers_index.at<int>(i, 0)]);
    }

}

void remove_SolvePnP_oulier_ (std::vector<cv::Point3d> &map_point_, std::vector<cv::Point2f> &current_pts_, std::vector<int> &previous_track_point_for_triangulate_ID_, cv::Mat inliers_index)
{
    std::vector<cv::Point3d> clone_map_point(map_point_);
    std::vector<cv::Point2f> clone_current_pts(current_pts_);
    std::vector<int> clone_previous_track_point_for_triangulate_ID(previous_track_point_for_triangulate_ID_);
    
    map_point_.clear();
    current_pts_.clear();
    previous_track_point_for_triangulate_ID_.clear();
    
    for (int i = 0; i < inliers_index.rows;  i++)
    {
        map_point_.push_back(clone_map_point[inliers_index.at<int>(i, 0)]);
        current_pts_.push_back(clone_current_pts[inliers_index.at<int>(i, 0)]);
        previous_track_point_for_triangulate_ID_.push_back(clone_previous_track_point_for_triangulate_ID[inliers_index.at<int>(i, 0)]);
    }

}

void changeStructure(const cv::Mat &plain, std::vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
    // std::cout << out[i].size() << std::endl;
  }
}

Eigen::Matrix3d RotationMatToEigen3d(cv::Mat r_)
{   
    
    Eigen::Matrix3d aa;
    aa <<   r_.at<double>(0, 0), r_.at<double>(0, 1), r_.at<double>(0, 2),
            r_.at<double>(1, 0), r_.at<double>(1, 1), r_.at<double>(1, 2),
            r_.at<double>(2, 0), r_.at<double>(2, 1), r_.at<double>(2, 2);

    return aa;
}

cv::Mat Eigen3dToRotationMat(Eigen::Matrix3d ae)
{   
    cv::Mat aa = (cv::Mat_<double>(3, 3) <<     ae(0, 0), ae(0, 1), ae(0, 2),
                                                ae(1, 0), ae(1, 1), ae(1, 2),
                                                ae(2, 0), ae(2, 1), ae(2, 2));


    return aa;
}

Eigen::Vector3d PoseToEigen3d(Map &conversion_pose, int num)
{
    cv::Mat aqw = vec6d_to_homogenous_campose(conversion_pose.keyframe[num].cam_pose);
    Eigen::Vector3d abc(aqw.at<double>(0, 3), aqw.at<double>(1, 3), aqw.at<double>(2, 3));
    return abc;
}


Eigen::Quaterniond getQuaternionFromRotationMatrix(const Eigen::Matrix3d& mat)
{
    // Eigen::AngleAxisd aa; 
    // aa = mat;
    Eigen::Quaterniond q(mat);// conversion error

    return q;
}

// Eigen::Quaterniond RotationToQuan(Map &conversion_rot)
// {
//     Eigen::Quaterniond abc;

// }

std::vector<cv::Point2f> KeypointToPoint2f(std::vector<cv::KeyPoint> keypoint)
{
    std::vector<cv::Point2f> pts;
    for(int i = 0; i < keypoint.size(); i++)
    {
        pts.push_back(keypoint[i].pt);
    }
}
    
void RemovePnPOutlier(std::vector<cv::Point3d>& MP, std::vector<cv::Point2f>& pts, cv::Mat inliers)
{
    std::vector<cv::Point3d> clone_MP;
    std::vector<cv::Point2f> clone_pts;
    
    for(int i = 0; i < inliers.rows; i++)
    {
        clone_MP.push_back(MP[inliers.at<int>(i, 0)]);
        clone_pts.push_back(pts[inliers.at<int>(i, 0)]);

    }
    
    MP.clear();
    pts.clear();

    MP = clone_MP;
    pts = clone_pts;
}    

