


#include "Data.h"






////////////////////////////////////////////////////////////////
////////////////////// Kitti data //////////////////////////////
std::string Read_Kitti_image_Data(char **kitti_data_num)
{
    // std::string input_image_path ("/mnt/mydisk/dataset/KITTI_Odom_dataset/data_odometry_gray/dataset/sequences/");
    std::string input_image_path ("/home/donghoon/Data/Kitti/");

    std::string final_input_image_path = input_image_path + kitti_data_num[1] + "/image_0/%06d.png";

    return final_input_image_path;
}
    
double* Read_Kitti_Calibration_Data(char **kitti_calibration_data_num, double intrinsic_[])
{
    std::string tmp;
    std::string buffer2[13];
    

    // std::string cam_calib_path ("/mnt/mydisk/dataset/KITTI_Odom_dataset/data_odometry_calib/dataset/sequences/");
    std::string cam_calib_path ("/home/donghoon/Data/Kitti/");
    std::string final_came_calib_path = cam_calib_path + kitti_calibration_data_num[1] + "/calib.txt";
    std::ifstream calib_data(final_came_calib_path);

    std::getline(calib_data,tmp);
    std::istringstream tmp_arr(tmp);

    for (int i =0; i < tmp.size(); i++) tmp_arr >> buffer2[i];
    for(int i= 0; i < 12; i++) intrinsic_[i] = std::stod(buffer2[i + 1]);


    return intrinsic_;
            
}
        


std::ifstream Read_Kiiti_GT_Data(char **kitti_gt_data_num)
{
    // std::string GT_data_path ("/mnt/mydisk/dataset/KITTI_Odom_dataset/data_odometry_poses/dataset/poses/");
    std::string GT_data_path ("/home/donghoon/Data/Kitti/");
    std::string final_GT_data_path = GT_data_path + kitti_gt_data_num[1] + "/GT.txt";
    std::ifstream GT_data(final_GT_data_path);


    return GT_data;
}


////////////////////////////////////////////////////////////////
////////////////////// Euroc Data //////////////////////////////








////////////////////////////////////////////////////////////////////
////////////////////// NaverLabs Data //////////////////////////////


