#include "opencv2/opencv.hpp"
#include "opencv2/sfm/projection.hpp"

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

void track_opticalflow_and_remove_err(cv::Mat &previous_, cv::Mat &current_, std::vector<cv::Point2f> &previous_pts_, std::vector<cv::Point2f> &current_pts_)
{
    std::vector<uchar> status_;
    cv::Mat err_;

    current_pts_.clear();

    cv::calcOpticalFlowPyrLK(previous_, current_, previous_pts_, current_pts_, status_, err_);

    const int image_x_size_ = previous_.cols;
    const int image_y_size_ = previous_.rows;

    // remove err point
    int indexCorrection = 0;

    for( int i=0; i<status_.size(); i++)
    {
        cv::Point2f pt = current_pts_.at(i- indexCorrection);
        if ((status_[i] == 0)||(pt.x<0)||(pt.y<0)||(pt.x>image_x_size_)||(pt.y>image_y_size_))	
        {
                    if((pt.x < 0)||(pt.y < 0 )||(pt.x > image_x_size_)||(pt.y > image_y_size_)) status_[i] = 0;
                    previous_pts_.erase ( previous_pts_.begin() + i - indexCorrection);
                    current_pts_.erase (current_pts_.begin() + i - indexCorrection);
                    indexCorrection++;
        }
    }     



}

void track_opticalflow_and_remove_err_for_SolvePnP(cv::Mat &previous_, cv::Mat &current_, std::vector<cv::Point2f> &previous_pts_, std::vector<cv::Point2f> &current_pts_, std::vector<cv::Point3d> &map_point_)
{
    std::vector<uchar> status_;
    cv::Mat err_;

    current_pts_.clear();

    cv::calcOpticalFlowPyrLK(previous_, current_, previous_pts_, current_pts_, status_, err_);

    const int image_x_size_ = previous_.cols;
    const int image_y_size_ = previous_.rows;

    // remove err point
    int indexCorrection = 0;

    for( int i=0; i<status_.size(); i++)
    {
        cv::Point2f pt = current_pts_.at(i- indexCorrection);
        if ((status_[i] == 0)||(pt.x<0)||(pt.y<0)||(pt.x>image_x_size_)||(pt.y>image_y_size_))	
        {
                    if((pt.x < 0)||(pt.y < 0 )||(pt.x > image_x_size_)||(pt.y > image_y_size_)) status_[i] = 0;
                    previous_pts_.erase ( previous_pts_.begin() + i - indexCorrection);
                    current_pts_.erase (current_pts_.begin() + i - indexCorrection);
                    map_point_.erase (map_point_.begin() + i - indexCorrection);
                    indexCorrection++;
        }
    }     


}