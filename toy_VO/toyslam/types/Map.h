#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "Frame.h"
#include <map>


class Map
{
    public:
        std::map< int, Frame > keyframe;
        std::map< int, cv::Point3d> world_xyz;
        std::map< int, std::vector<int> > MapToKF_ids;
        std::vector<cv::Point3d> InlierMap;
        std::vector<int> InlierID;

        std::vector<cv::Point3d> CompressionMap;
        
        std::map< int, std::vector<cv::KeyPoint> > LoopKeyPoint;
        std::map< int, std::vector<cv::Point2f> > LoopPoint2f;
        std::map< int, cv::Mat> LoopDescriptor;
        std::map< int, cv::Point3d> LoopMap;

        Map()
        {}

        Map(std::map< int, Frame> _keyframe)
            : keyframe(_keyframe)
        {}
        
        Map(std::map< int, Frame> _keyframe, std::map< int, cv::Point3d> _world_xyz)
            : keyframe(_keyframe), world_xyz(_world_xyz)
        {}

        Map(const Map &tc)
        {
            keyframe = tc.keyframe;
            world_xyz = tc.world_xyz;
            MapToKF_ids = tc.MapToKF_ids;
            LoopKeyPoint = tc.LoopKeyPoint;
            LoopPoint2f = tc.LoopPoint2f;
            LoopDescriptor = tc.LoopDescriptor;
            InlierMap = tc.InlierMap;
            InlierID = tc.InlierID;
            CompressionMap = tc.CompressionMap;
            LoopMap = tc.LoopMap;
        }

        // void MakeLoopMap(int BeforeNum, int CurrNum, Map MapInfo)
        // {
        //     cv::Ptr<cv::ORB> orb = cv::ORB::create(4000);
        //     std::vector<cv::KeyPoint> keypoints;
        //     cv::Mat descriptors;
        //     cv::Mat mask;
        // }
};