#include "opencv2/opencv.hpp"
#include "opencv2/sfm/projection.hpp"

void print_map(std::map<int, cv::Point3d>& m) {
    for (std::map<int, cv::Point3d>::iterator itr = m.begin(); itr != m.end(); ++itr) {
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

cv::Mat cam_storage_to_projection_matrix(cv::Vec6d cam_storage)
{
    cv::Mat P_ = vec6d_to_homogenous_campose(cam_storage);
    P_ = P_.inv();
    P_.resize(3);    
    return P_;

}

std::vector<int> track_opticalflow_and_remove_err(cv::Mat &previous_, cv::Mat &current_, std::vector<cv::Point2f> &previous_pts_, std::vector<cv::Point2f> &current_pts_)
{
    std::vector<uchar> status_;
    cv::Mat err_;
    std::vector<int> correspond_id_;

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
                    indexCorrection++;
        }
        else 
        {
            correspond_id_.push_back(i);
        }
    }     

    return correspond_id_;

}

std::vector<int> track_opticalflow_and_remove_err_for_SolvePnP(cv::Mat &previous_, cv::Mat &current_, std::vector<cv::Point2f> &previous_pts_, std::vector<cv::Point2f> &current_pts_, std::vector<cv::Point3d> &map_point_)
{
    std::vector<uchar> status_;
    cv::Mat err_;
    std::vector<int> correspond_id_;

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
        else 
        {
            correspond_id_.push_back(i);
        }

    }     

    return correspond_id_;

}

void remove_map_point_and_2dpoint_outlier (std::vector<cv::Point3d> &map_point_, std::vector<cv::Point2f> &current_pts_, cv::Mat &current_cam_pose_)
{


    // Intial camera viewing vector
    cv::Mat initilize_viewing_camera = (cv::Mat_<double>(4, 1) << 0, 0, 1, 0);
    
    // Map point type change ( std::vector<cv::Point3d> -> N*3 Mat channel 1 )
    std::vector<cv::Point3d> clone_map_point;
    for(int i = 0; i < map_point_.size(); i++) clone_map_point.push_back(map_point_[i]);
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
                    if(current_viewing_camera.dot(vector_map_point) < 0)
                    {
                        
                        map_point_.erase(map_point_.begin() + i - indexCorrection);
                        current_pts_.erase(current_pts_.begin() + i - indexCorrection);
                        indexCorrection++;
                    
                    }
                
                }               
                    
}