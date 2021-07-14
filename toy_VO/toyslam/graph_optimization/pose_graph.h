#pragma once

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




// G2O_USE_TYPE_GROUP(slam2d);
// G2O_USE_TYPE_GROUP(slam3d);



int getNewID();


void addPoseVertex(g2o::SparseOptimizer* optimizer, g2o::SE3Quat& pose, bool set_fixed);


void addEdgePosePose(g2o::SparseOptimizer* optimizer, int id0, int id1, g2o::SE3Quat& relpose);

