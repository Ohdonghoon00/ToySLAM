#include "opencv2/opencv.hpp"
#include <cmath>
#include "opencv2/sfm/triangulation.hpp"
#include "opencv2/sfm/projection.hpp"
#include "TOY_VO.h"

using namespace std;
using namespace cv;

int main()
{
    Mat R = Mat::eye(3, 3, CV_64F), t = Mat::eye(3, 1, CV_64F), Rt;

    Rt = R_t_to_homogenous(R, t);
    cout << Rt << endl;


    return 0;
}