#include "gurobi_c++.h"
#include <vector>
#include <Eigen/Dense>
#include "types/Map.h"
#include "algorithm"

std::vector<GRBVar> CreateVariablesBinaryVector(int PointCloudNum, GRBModel& model_);

Eigen::Matrix<double, Eigen::Dynamic, 1> CalculateObservationCountWeight(Map& map_data);

void SetObjectiveILP(std::vector<GRBVar> x_, Eigen::Matrix<double, Eigen::Dynamic, 1> q_, GRBModel& model_);

Eigen::MatrixXd CalculateVisibilityMatrix(Map& map_data, std::vector<cv::Mat> Inlier_ST);

void AddConstraint(Map& map_data, GRBModel& model_, Eigen::MatrixXd A, std::vector<GRBVar> x);