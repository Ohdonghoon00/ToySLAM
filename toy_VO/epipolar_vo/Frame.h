#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class Frame
{
    public:
        cv::Mat frame;
        int Frame_id;
        std::vector<int> frame_pts_id;
        std::vector<cv::Point2f> image_pts;
        std::vector<cv::Point3d> cam_pose = {{0,0,0}};
        
        Frame( )
        {}
        Frame ( int init_Frame_id)
            : Frame_id(init_Frame_id)
        {}    


};