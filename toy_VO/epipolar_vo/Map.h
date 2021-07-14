#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "Frame.h"

class Map
{
    public:
        std::vector<cv::Vec6f> LandMark;
        Frame keyframe;
};

