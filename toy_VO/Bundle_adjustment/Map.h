#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "Frame.h"

class Map
{
    public:
        Keyframe keyframe_;
        std::vector<cv::Point3d> world_xyz;

        Map()
        {}

        Map(Keyframe _keyframe, std::vector<cv::Point3d> _world_xyz)
            : keyframe_(_keyframe), world_xyz(_world_xyz)
        {}

        Map(const Map &tc)
        {
            keyframe_ = tc.keyframe_;
            world_xyz = tc.world_xyz;

        }
};

