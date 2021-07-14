#include <string>
#include <fstream>







// Kitti data
std::ifstream Prepare_Kiiti_dataset(const char kitti_data_num)
{
    std::string input_image_path ("../../../../dataset/sequences/");
    std::string cam_calib_path ("../../../../dataset/sequences/");
    std::string GT_data_path ("../../../../dataset/poses/");
    
    std::string final_input_image_path = input_image_path + kitti_data_num + "/image_0/%06d.png";
    std::string final_came_calib_path = cam_calib_path + kitti_data_num + "/calib.txt";
    std::string final_GT_data_path = GT_data_path + kitti_data_num + ".txt";

    std::ifstream calib_data(final_came_calib_path);
    std::ifstream GT_data(final_GT_data_path);

    // return 
}

std::ifstream Read_Kitti_Calibration_Data(const char &kitti_calibration_data_num)
{
    std::string cam_calib_path ("../../../../dataset/sequences/");
    std::string final_came_calib_path = cam_calib_path + kitti_calibration_data_num + "/calib.txt";
    std::ifstream calib_data(final_came_calib_path);

    return calib_data;
}

void Read_Kiiti_GT_Data(const char &kitti_gt_data_num)
{
    std::string GT_data_path ("../../../../dataset/poses/");
    std::string final_GT_data_path = GT_data_path + kitti_gt_data_num + ".txt";
    std::ifstream GT_data(final_GT_data_path);


    // return GT_data;
}
