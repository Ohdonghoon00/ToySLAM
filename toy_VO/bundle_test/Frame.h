#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class Frame
{
    public:
        int Frame_id;
        cv::Mat frame;
        std::vector<int> pts_id;
        std::vector<cv::Point2f> pts;
        
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
            cam_pose = tc.cam_pose;
        }    
            


};

// class Keyframe 
// {
//     public :
//         int keyframe_id;
//         Frame keyframe;

//     Keyframe()
//     {}
//     Keyframe(int init_keyframe_id)
//         : keyframe_id(init_keyframe_id)
//     {}
//     Keyframe(const Keyframe &tc)
//     {
//             keyframe_id = tc.keyframe_id;
//             keyframe = tc.keyframe;
//     }    

// };

    
        




