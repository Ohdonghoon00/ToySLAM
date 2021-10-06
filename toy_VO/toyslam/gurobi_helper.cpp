#include "gurobi_helper.h"



std::vector<GRBVar> CreateVariablesBinaryVector(int PointCloudNum, GRBModel& model_)
{
    
    std::vector<GRBVar> x;
    x.resize(PointCloudNum);
    for(int i = 0; i < PointCloudNum; i++ )
    {
        x[i] = model_.addVar(-5.0, 3.0, 0.0, GRB_BINARY);
    }

    return x;
}

Eigen::Matrix<double, Eigen::Dynamic, 1> CalculateObservationCountWeight(Map& map_data)
{
    Eigen::Matrix<double, Eigen::Dynamic, 1> q;
    int PointCloudNum_ = map_data.InlierID.size();
    q.resize(PointCloudNum_);
    int KeyframeNum = map_data.keyframe.size();
    
    for(int i = 0; i < map_data.InlierID.size(); i++)
    {
        int PointCloudID = map_data.InlierID[i];
        q[i] = (double)map_data.MapToKF_ids[PointCloudID].size() / (double)KeyframeNum;
    }
    return q;
}

void SetObjectiveILP(std::vector<GRBVar> x_, Eigen::Matrix<double, Eigen::Dynamic, 1> q_, GRBModel& model_)
{
    GRBLinExpr obj = 0;
    for(int i = 0; i < x_.size(); i++)
    {
        obj += q_[i] * x_[i];
    } 
    model_.setObjective(obj);
    
}

Eigen::MatrixXd CalculateVisibilityMatrix(Map& map_data, std::vector<cv::Mat> Inlier_ST)
{
    Eigen::MatrixXd A(map_data.keyframe.size(), map_data.InlierID.size()); 
    A.setZero();
    for(int i = 0; i < A.rows(); i++ )
        {
            // std::cout << std::endl;
            for(int j = 0; j < Inlier_ST[i].rows; j++)
            {
                
                int id = map_data.keyframe[i].pts_id[Inlier_ST[i].at<int>(j, 0)];
                auto index = find(map_data.InlierID.begin(), map_data.InlierID.end(), id) - map_data.InlierID.begin();
                A(i, index) = 1.0;
            }
        }

    return A;
}

void AddConstraint(Map& map_data, GRBModel& model_, Eigen::MatrixXd A, std::vector<GRBVar> x)
{
    GRBLinExpr constraint = 0;
    double b = 35.0;
    for(int i = 0; i < map_data.keyframe.size(); i++)
    {
       constraint.clear();
       for(int j = 0; j < map_data.InlierID.size(); j++)
       {
        
        constraint += A(i, j) * x[j];

       }
       model_.addConstr(constraint >= b);
    }
}
