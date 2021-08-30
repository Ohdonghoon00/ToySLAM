
#include "pose_graph.h"

#include <Eigen/Dense>

// G2O_USE_TYPE_GROUP(slam2d);
// G2O_USE_TYPE_GROUP(slam3d);



int getNewID()
{
    static int vertex_id = 0;
    std::cout << " vertex id  : " << vertex_id + 1 << std::endl; 
    return vertex_id++;
    
}

void addPoseVertex(g2o::SparseOptimizer* optimizer, g2o::SE3Quat& pose, bool set_fixed, int id)
{
    // std::cout << "add pose: t=" << pose.translation().transpose()
    //           << " r=" << pose.rotation().coeffs().transpose() << std::endl;
    g2o::VertexSE3* v_se3 = new g2o::VertexSE3;
    v_se3->setId(id);
    // if(set_fixed)
      v_se3->setEstimate(pose);
    v_se3->setFixed(set_fixed);
    optimizer->addVertex(v_se3);
}

void addEdgePosePose(g2o::SparseOptimizer* optimizer, int id0, int id1, g2o::SE3Quat& relpose)
{
    // std::cout << "add edge: id0 = " << id0 << ", id1 = " << id1
              // << ", t=" << relpose.translation().transpose()
              // << ", r=" << relpose.rotation().coeffs().transpose() << std::endl;

    g2o::EdgeSE3* edge = new g2o::EdgeSE3;
    edge->setVertex(0, optimizer->vertices().find(id0)->second);
    edge->setVertex(1, optimizer->vertices().find(id1)->second);
    edge->setMeasurement(relpose);
    Eigen::MatrixXd info_matrix = Eigen::MatrixXd::Identity(6,6)* 10;
    edge->setInformation(info_matrix);
    optimizer->addEdge(edge);
}

void ToVertexSim3(const g2o::VertexSE3 &v_se3,
                  g2o::VertexSim3Expmap *const v_sim3)
{
  Eigen::Isometry3d se3 = v_se3.estimate().inverse();
  Eigen::Matrix3d r = se3.rotation();
  Eigen::Vector3d t = se3.translation();

  // std::cout<<"Convert vertices to Sim3 !!! " <<  " r : " << std::endl
  //     <<se3.rotation()<< std::endl << " t : "
  //     <<se3.translation()<<std::endl;
  g2o::Sim3 sim3(r, t, 1.0);

  v_sim3->setEstimate(sim3);
}



// Converte EdgeSE3 to EdgeSim3
void ToEdgeSim3(const g2o::EdgeSE3 &e_se3, g2o::EdgeSim3 *const e_sim3, double scale)
{
  Eigen::Isometry3d se3 = e_se3.measurement().inverse();
  Eigen::Matrix3d r = se3.rotation();
  Eigen::Vector3d t = se3.translation();

  // std::cout<<"Convert edges to Sim3 !!! " << " r : " << std::endl
  //     <<se3.rotation()<< std::endl << " t : " 
  //     <<se3.translation()<<std::endl;
  g2o::Sim3 sim3(r, t, scale);

  e_sim3->setMeasurement(sim3);
}

double FindLoopEdgeScale(int loop_edge_id, int curr_id, Map MapST, cv::Mat K_, std::vector<cv::Mat> map_point_inlier, cv::Mat &relpose)
{
    std::vector<cv::Point2f> track_point_for_add_PG_edge_;
    std::vector<cv::Point2f> loopframe_inlier_2dpoint;
    std::vector<cv::Point3d> loopframe_inlier_3dpoint;
    std::vector<Eigen::Vector3d> loopframe_inlier_3dpoint_;
    cv::Mat inliers;
    cv::Mat R_, t_;
    std::vector<double> distance_Loopframe, distance_currframe, distance_ratio;
    Eigen::Vector3d curr_pose, loop_pose;
    double s_ = 1.0;
    double f = K_.at<double>(0, 0);
    cv::Point2d c(K_.at<double>(0, 2), K_.at<double>(1, 2));
    // const cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, c.x, 0, f, c.y, 0, 0, 1);
    
    loop_pose <<  vec6d_to_homogenous_campose(MapST.keyframe[loop_edge_id].cam_pose).at<double>(0, 3),
                  vec6d_to_homogenous_campose(MapST.keyframe[loop_edge_id].cam_pose).at<double>(1, 3),
                  vec6d_to_homogenous_campose(MapST.keyframe[loop_edge_id].cam_pose).at<double>(2, 3);


      
    for(int i = 0; i < map_point_inlier[loop_edge_id].rows; i++)
    {
      loopframe_inlier_2dpoint.push_back(MapST.keyframe[loop_edge_id].pts[map_point_inlier[loop_edge_id].at<int>(i, 0)]);
      int id_ = MapST.keyframe[loop_edge_id].pts_id[map_point_inlier[loop_edge_id].at<int>(i, 0)];
      loopframe_inlier_3dpoint.push_back(MapST.world_xyz[id_]);
    } 
    
    
    track_opticalflow_and_remove_err_for_SolvePnP_noid( MapST.keyframe[loop_edge_id].frame, 
                                      MapST.keyframe[curr_id].frame, 
                                      loopframe_inlier_2dpoint,
                                      track_point_for_add_PG_edge_,
                                      loopframe_inlier_3dpoint);
    // cv::Mat inlier_mask;
    // cv::Mat E = cv::findEssentialMat(loopframe_inlier_2dpoint, track_point_for_add_PG_edge_, f, c, cv::RANSAC, 0.99, 1, inlier_mask);
    // int inlier_num = cv::recoverPose(E, loopframe_inlier_2dpoint, track_point_for_add_PG_edge_, R, t, f, c, inlier_mask);

    cv::solvePnPRansac(loopframe_inlier_3dpoint, track_point_for_add_PG_edge_, K_, cv::noArray(), R_, t_, false, 100, 3.0F, 0.99, inliers );
    cv::Rodrigues(R_, R_);
    cv::Mat World_Rt = R_t_to_homogenous(R_, t_);
    cv::Mat cam_pose = World_Rt.inv();
    relpose = vec6d_to_homogenous_campose(MapST.keyframe[loop_edge_id].cam_pose).inv() * cam_pose;
    relpose = relpose.inv();
    curr_pose <<  cam_pose.at<double>(0, 3),
                  cam_pose.at<double>(1, 3),
                  cam_pose.at<double>(2, 3);
    
    for(int i = 0; i < inliers.rows; i++)
    {
        Eigen::Vector3d abc;
        abc <<  (loopframe_inlier_3dpoint[inliers.at<int>(i, 0)]).x, 
                (loopframe_inlier_3dpoint[inliers.at<int>(i, 0)]).y, 
                (loopframe_inlier_3dpoint[inliers.at<int>(i, 0)]).z;
        loopframe_inlier_3dpoint_.push_back(abc);        
    }
     
    for(int i = 0; i < loopframe_inlier_3dpoint_.size(); i++)
    {
      double distance_loop, distance_curr;
      distance_loop = sqrt((loopframe_inlier_3dpoint_[i] - loop_pose).dot(loopframe_inlier_3dpoint_[i] - loop_pose));
      // distance_Loopframe.push_back(distance);

      distance_curr = sqrt((loopframe_inlier_3dpoint_[i] - curr_pose).dot(loopframe_inlier_3dpoint_[i] - curr_pose));
      // distance_currframe.push_back(distance);

      distance_ratio.push_back(distance_curr/distance_loop);
    }
    
    sort(distance_ratio.begin(), distance_ratio.end());
    
// for(int i = 0; i < distance_ratio.size(); i++) std::cout << distance_ratio[i] << std::endl;
int inlier_ratio = 100 * inliers.rows / loopframe_inlier_3dpoint.size();
  if(inlier_ratio < 50) return s_;
  else
  {
    std::cout << "inlier ratio  : " << inlier_ratio << std::endl;
    return distance_ratio[distance_ratio.size()/2];
  }
}


double VerifyLoop(int loop_edge_id, int curr_id, Map MapST, cv::Mat K_, cv::Mat& relpose)
{
    Eigen::Vector3d curr_pose, loop_pose;
    std::vector<Eigen::Vector3d> loopframe_inlier_3dpoint_;
    cv::Mat P0, P1, X, inlier_mask_LB, inlier_mask_LC, R, t, inliers, inliers_;
    double f = K_.at<double>(0, 0);
    cv::Point2d c(K_.at<double>(0, 2), K_.at<double>(1, 2));
    int inlier_num = 0;
    double s_ = 1.0; 
    std::vector<double> distance_ratio;
    std::vector<cv::Point2f> LoopPoint, LoopPoint_;
    std::vector<cv::Point2f> CurrPoint, CurrPoint_;
    std::vector<cv::Point2f> BeforeLoopPoint, BeforeLoopPoint_;
    std::vector<int> inlier_id;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches_BeforeLoop, matches_LoopCurr;
    std::vector<cv::Point3d> map_point_, map_point;
    
    std::cout << MapST.LoopDescriptor[loop_edge_id].size() << std::endl;
    
    // Loop - Curr matching
    matcher->match(MapST.LoopDescriptor[loop_edge_id], MapST.LoopDescriptor[curr_id], matches_LoopCurr);
    
    // Loop - Before Loop matching
    matcher->match(MapST.LoopDescriptor[loop_edge_id], MapST.LoopDescriptor[loop_edge_id - 1], matches_BeforeLoop);
    for(int i = 0; i < matches_BeforeLoop.size(); i++)
    { 
      LoopPoint.push_back(MapST.LoopKeyPoint[loop_edge_id][matches_BeforeLoop[i].queryIdx].pt);
      BeforeLoopPoint.push_back(MapST.LoopKeyPoint[loop_edge_id - 1][matches_BeforeLoop[i].trainIdx].pt);
      CurrPoint.push_back(MapST.LoopKeyPoint[curr_id][matches_LoopCurr[i].trainIdx].pt);
    }
    cv::Mat E = cv::findEssentialMat(LoopPoint, BeforeLoopPoint, f, c, cv::RANSAC, 0.99, 1, inlier_mask_LB);
    cv::Mat E_ = cv::findEssentialMat(LoopPoint, CurrPoint, f, c, cv::RANSAC, 0.99, 1, inlier_mask_LC);    
    
    for(int i = 0; i < inlier_mask_LB.rows; i++)
    {
      
      if(inlier_mask_LB.at<bool>(i, 0) == 1 && inlier_mask_LC.at<bool>(i, 0) == 1)
      {
        LoopPoint_.push_back(LoopPoint[i]);
        BeforeLoopPoint_.push_back(BeforeLoopPoint[i]);
        CurrPoint_.push_back(CurrPoint[i]);
      }
    }

    LoopPoint.clear();
    CurrPoint.clear();
    
    P0 = K_ * cam_storage_to_projection_matrix(MapST.keyframe[loop_edge_id].cam_pose);
    P1 = K_ * cam_storage_to_projection_matrix(MapST.keyframe[loop_edge_id - 1].cam_pose);     
    cv::triangulatePoints(P0, P1, LoopPoint_, BeforeLoopPoint_, X);

    world_xyz_point_to_homogenous(X);
    X.convertTo(X, CV_64F);
    map_point_.clear();
    for (int i = 0; i < LoopPoint_.size(); i++ ) map_point_.push_back(cv::Point3d(X.at<double>(0, i), X.at<double>(1, i), X.at<double>(2, i)));

    cv::solvePnPRansac(map_point_, LoopPoint_, K_, cv::noArray(), R, t, false, 100, 3.0F, 0.99, inliers );
    for (int i = 0; i < inliers.rows; i++)
    {
      LoopPoint.push_back(LoopPoint_[inliers.at<int>(i, 0)]);
      CurrPoint.push_back(CurrPoint_[inliers.at<int>(i, 0)]);
      map_point.push_back(map_point_[inliers.at<int>(i, 0)]);
    }
    int SolvePnP_inlier_ratio = 100 * inliers.rows / map_point_.size();

  std::cout << " inlier_num : " << LoopPoint.size() << "     " << CurrPoint.size() << "     " << map_point.size() <<std::endl;

    cv::solvePnPRansac(map_point, CurrPoint, K_, cv::noArray(), R, t, false, 100, 3.0F, 0.99, inliers_ );
    cv::Rodrigues(R, R);
    cv::Mat World_Rt = R_t_to_homogenous(R, t);
    cv::Mat cam_pose = World_Rt.inv();
    relpose = vec6d_to_homogenous_campose(MapST.keyframe[loop_edge_id].cam_pose).inv() * cam_pose;
    relpose = relpose.inv();   

    // Scale 
    curr_pose <<  cam_pose.at<double>(0, 3),
                  cam_pose.at<double>(1, 3),
                  cam_pose.at<double>(2, 3);    
    
    loop_pose <<  vec6d_to_homogenous_campose(MapST.keyframe[loop_edge_id].cam_pose).at<double>(0, 3),
                  vec6d_to_homogenous_campose(MapST.keyframe[loop_edge_id].cam_pose).at<double>(1, 3),
                  vec6d_to_homogenous_campose(MapST.keyframe[loop_edge_id].cam_pose).at<double>(2, 3);

    for(int i = 0; i < inliers_.rows; i++)
    {
        Eigen::Vector3d abc;
        abc <<  (map_point[inliers_.at<int>(i, 0)]).x, 
                (map_point[inliers_.at<int>(i, 0)]).y, 
                (map_point[inliers_.at<int>(i, 0)]).z;
        loopframe_inlier_3dpoint_.push_back(abc);        
    }

    for(int i = 0; i < loopframe_inlier_3dpoint_.size(); i++)
    {
      double distance_loop, distance_curr;
      distance_loop = sqrt((loopframe_inlier_3dpoint_[i] - loop_pose).dot(loopframe_inlier_3dpoint_[i] - loop_pose));
      // distance_Loopframe.push_back(distance);

      distance_curr = sqrt((loopframe_inlier_3dpoint_[i] - curr_pose).dot(loopframe_inlier_3dpoint_[i] - curr_pose));
      // distance_currframe.push_back(distance);

      distance_ratio.push_back(distance_curr/distance_loop);
    }
    
    sort(distance_ratio.begin(), distance_ratio.end());
    
    int inlier_ratio = 100 * inliers_.rows / map_point.size();
    std::cout << "inlier ratio  : " << inlier_ratio << std::endl;
    
    if(inlier_ratio < 40) return s_;
    else
    {
      return distance_ratio[distance_ratio.size()/2];
    }

  
}