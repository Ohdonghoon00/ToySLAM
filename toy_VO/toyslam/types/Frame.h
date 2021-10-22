#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
// #include <Eigen/Dense>

class Frame
{
    public:
        int Frame_id;
        cv::Mat frame;
        std::vector<int> pts_id;
        std::vector<cv::Point2f> pts;
        
        std::vector<int> TrackId;
        std::vector<cv::Point2f> TrackPts;
        
        cv::Vec6d cam_pose;
        
        Frame( )
        {}
        Frame ( int init_Frame_id)
            : Frame_id(init_Frame_id)
        {}
        Frame(const Frame &tc)
        {
            Frame_id = tc.Frame_id;
            frame = tc.frame;
            pts_id = tc.pts_id;
            pts = tc.pts;
            TrackId = tc.TrackId;
            TrackPts = tc.TrackPts;
            cam_pose = tc.cam_pose;
        }    
            


};