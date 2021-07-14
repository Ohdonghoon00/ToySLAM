#pragma once
 
// #include <iostream>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
// #include "g2o/solvers/csparse/linear_solver_csparse.h"
// #include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/factory.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/edge_se3.h"
#include "g2o/types/sba/types_six_dof_expmap.h"



// // G2O_USE_TYPE_GROUP(slam2d);
// // G2O_USE_TYPE_GROUP(slam3d);    



int getNewID();


void addPoseVertex(g2o::SparseOptimizer* optimizer, g2o::SE3Quat& pose, bool set_fixed);


void addEdgePosePose(g2o::SparseOptimizer* optimizer, int id0, int id1, g2o::SE3Quat& relpose);

// int getNewID()
// {
//     static int vertex_id = 0;
//     return vertex_id++;
// }

// void addPoseVertex(g2o::SparseOptimizer* optimizer, g2o::SE3Quat& pose, bool set_fixed)
// {
//     std::cout << "add pose: t=" << pose.translation().transpose()
//               << " r=" << pose.rotation().coeffs().transpose() << std::endl;
//     g2o::VertexSE3* v_se3 = new g2o::VertexSE3;
//     v_se3->setId(getNewID());
//     if(set_fixed)
//         v_se3->setEstimate(pose);
//     v_se3->setFixed(set_fixed);
//     optimizer->addVertex(v_se3);
// }

// void addEdgePosePose(g2o::SparseOptimizer* optimizer, int id0, int id1, g2o::SE3Quat& relpose)
// {
//     std::cout << "add edge: id0=" << id0 << ", id1" << id1
//               << ", t=" << relpose.translation().transpose()
//               << ", r=" << relpose.rotation().coeffs().transpose() << std::endl;

//     g2o::EdgeSE3* edge = new g2o::EdgeSE3;
//     edge->setVertex(0, optimizer->vertices().find(id0)->second);
//     edge->setVertex(1, optimizer->vertices().find(id1)->second);
//     edge->setMeasurement(relpose);
//     Eigen::MatrixXd info_matrix = Eigen::MatrixXd::Identity(6,6) * 10.;
//     edge->setInformation(info_matrix);
//     optimizer->addEdge(edge);
// }