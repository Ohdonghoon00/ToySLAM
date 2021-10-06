

#include "gurobi_c++.h"
#include <vector>
#include <Eigen/Dense>
using namespace std;

int
main(int   argc,
     char *argv[])
{
  try {
    GRBEnv env = GRBEnv();

    GRBModel model = GRBModel(env);
    int cnt = 10;
    // Create variables
    std::vector<GRBVar> x;
    
    Eigen::Matrix<double, 10, 1> q;
    q << 0.4, 0.4, 0.2, 0.6, 0.4, 0.6, 0.4, 0.4, 0.4, 0.4;
    
    Eigen::Matrix<double, 5, 10> A;
    A <<    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0;
    
    Eigen::Matrix<double, 5, 1> b;
    b << 2.0, 2.0, 2.0, 2.0, 2.0;
    
    // std::vector<double> q(0.4, 0.4, 0.2, 0.6);
    x.resize(cnt);
    // q.resize(cnt);
    
    // for(int i =0; i < cnt; i++)
    // {
    //     q[i] = weight;
    //     weight += 0.1;
    // }
    
    for(int i = 0; i < cnt; i++ )
    {
        x[i] = model.addVar(-5.0, 3.0, 0.0, GRB_BINARY);
    }
    // GRBVar z = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "z");
    
    // Set objective
    // GRBQuadExpr obj = x*x + x*y + y*y + y*z + z*z + 2*x;
    GRBLinExpr obj = 0;
    for(int i = 0; i < x.size(); i++)
    {
        obj += q[i] * x[i];
    } 
    model.setObjective(obj);
    
    // Add constraint
    GRBLinExpr constraint = 0;
    for(int i = 0; i < 5; i++)
    {
       constraint.clear();
       for(int j = 0; j < cnt; j++)
       {
        
        constraint += A(i, j) * x[j];

       }
       model.addConstr(constraint >= b[i]);
    }


    // Optimize model

    model.optimize();

    // cout << x.get(GRB_StringAttr_VarName) << " "
    //      << x.get(GRB_DoubleAttr_X) << endl;
    // cout << y.get(GRB_StringAttr_VarName) << " "
    //      << y.get(GRB_DoubleAttr_X) << endl;
    // cout << z.get(GRB_StringAttr_VarName) << " "
    //      << z.get(GRB_DoubleAttr_X) << endl;

    cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
    cout << " x : "; 
    for(int i = 0; i < x.size(); i++) 
    {
        std::cout   << x[i].get(GRB_DoubleAttr_X) << "  ";
    }
    std::cout << std::endl;
    // // Change variable types to integer

    // x.set(GRB_CharAttr_VType, GRB_INTEGER);
    // y.set(GRB_CharAttr_VType, GRB_INTEGER);
    // z.set(GRB_CharAttr_VType, GRB_INTEGER);

    // // Optimize model

    // model.optimize();

    // cout << x.get(GRB_StringAttr_VarName) << " "
    //      << x.get(GRB_DoubleAttr_X) << endl;
    // cout << y.get(GRB_StringAttr_VarName) << " "
    //      << y.get(GRB_DoubleAttr_X) << endl;
    // cout << z.get(GRB_StringAttr_VarName) << " "
    //      << z.get(GRB_DoubleAttr_X) << endl;


  } catch(GRBException e) {
    cout << "Error code = " << e.getErrorCode() << endl;
    cout << e.getMessage() << endl;
  } catch(...) {
    cout << "Exception during optimization" << endl;
  }

  return 0;
}
