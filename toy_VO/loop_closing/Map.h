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
        // Keyframe keyframe_;
        // std::vector<cv::Point3d> world_xyz;

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

        }
};

