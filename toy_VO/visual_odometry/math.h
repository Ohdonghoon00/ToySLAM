#include "opencv2/opencv.hpp"

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
        if((pt.x < 0)||(pt.y < 0 )||(pt.x > image_x_size_)||(pt.y > image_y_size_)) status_[i] = 0;
        if (status_[i] == 0)	
        {
                    
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
        if((pt.x < 0)||(pt.y < 0 )||(pt.x > image_x_size_)||(pt.y > image_y_size_)) status_[i] = 0;
        if (status_[i] == 0)	
        {
                    
                    previous_pts_.erase ( previous_pts_.begin() + i - indexCorrection);
                    current_pts_.erase (current_pts_.begin() + i - indexCorrection);
                    map_point_.erase (map_point_.begin() + i - indexCorrection);
                    indexCorrection++;
        }
    }     


}