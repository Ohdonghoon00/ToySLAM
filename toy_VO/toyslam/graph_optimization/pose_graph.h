#pragma once

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
// #include "g2o/solvers/csparse/linear_solver_csparse.h"
// #include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/factory.h"
#include "g2o/core/robust_kernel_impl.h"

#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"

#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/edge_se3.h"
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d/se3quat.h>

#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/sim3/types_seven_dof_expmap.h"

#include "../types/Map.h"
#include "../math.h"
#include "opencv2/opencv.hpp"
#include <iostream>

// G2O_USE_TYPE_GROUP(slam2d);
// G2O_USE_TYPE_GROUP(slam3d);



int getNewID();


void addPoseVertex(g2o::SparseOptimizer* optimizer, g2o::SE3Quat& pose, bool set_fixed, int id);


void addEdgePosePose(g2o::SparseOptimizer* optimizer, int id0, int id1, g2o::SE3Quat& relpose);

void ToVertexSim3(const g2o::VertexSE3 &v_se3,
                  g2o::VertexSim3Expmap *const v_sim3);

void ToEdgeSim3(const g2o::EdgeSE3 &e_se3, g2o::EdgeSim3 *const e_sim3, double scale);

double FindLoopEdgeScale(int loop_edge_id, int curr_id, Map MapST, cv::Mat K_, std::vector<cv::Mat> map_point_inlier, cv::Mat &relpose);

double VerifyLoop(int loop_edge_id, int curr_id, Map MapST, cv::Mat K_, cv::Mat& relpose);