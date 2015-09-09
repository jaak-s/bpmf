#ifndef BPMF_H
#define BPMF_H

#define EIGEN_RUNTIME_NO_MALLOC 1
#define EIGEN_DONT_PARALLELIZE 1

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <random>

double randn(double);
auto nrandn(int n) -> decltype( Eigen::VectorXd::NullaryExpr(n, std::ptr_fun(randn)) );

std::pair<Eigen::VectorXd, Eigen::MatrixXd> CondNormalWishart(const Eigen::MatrixXd &U, const Eigen::VectorXd &mu, const double kappa, const Eigen::MatrixXd &T, const int nu);

#include <chrono>

inline double tick() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); 
}

/**
 * finds option from the list of command line arguments
 * returns 0 if not found
 */
char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

#endif
