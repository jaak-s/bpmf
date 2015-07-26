
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>

#include <unsupported/Eigen/SparseExtra>

#ifdef _OPENMP
#include <omp.h>
#else
#include <tbb/tbb.h>
#endif

#include "bpmf.h"

using namespace std;
using namespace Eigen;

int num_feat = 32;

double alpha = 2;
int nsims = 20;
int burnin = 5;

double mean_rating = .0;

typedef SparseMatrix<double> SparseMatrixD;
SparseMatrixD M, Mt, P;

//typedef Matrix<double, num_feat, 1> VectorNd;
//typedef Matrix<double, num_feat, num_feat> MatrixNNd;
//typedef Matrix<double, num_feat, Dynamic> MatrixNXd;

VectorXd mu_u;
VectorXd mu_m;
MatrixXd Lambda_u;
MatrixXd Lambda_m;
MatrixXd sample_u;
MatrixXd sample_m;

// parameters of Inv-Whishart distribution (see paper for details)
MatrixXd WI_u;
int b0_u = 2;
int df_u = num_feat;
VectorXd mu0_u;

MatrixXd WI_m;
int b0_m = 2;
int df_m = num_feat;
VectorXd mu0_m;

/**
 *  RNG for each thread.
 *  Adopted from 
 *    http://stackoverflow.com/questions/15918758/how-to-make-each-thread-use-its-own-rng-in-c11
 */
class RNG
{
public:
    typedef std::mt19937 Engine;
    typedef std::normal_distribution<double> Distribution;

    RNG() : engines(), distribution(0.0, 1.0)
    {
        int threads = std::max(1, omp_get_max_threads());
        for(int seed = 0; seed < threads; seed++)
        {
            unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
            seed1 = seed1 ^ seed;
            engines.push_back(Engine(seed1));
        }
    }

    double operator()()
    {
        int id = omp_get_thread_num();
        return distribution(engines[id]);
    }

    std::vector<Engine> engines;
    Distribution distribution;
};

// parallel normal random
RNG prandn;

void init() {
    mean_rating = M.sum() / M.nonZeros();
    Lambda_u.resize(num_feat, num_feat);
    Lambda_u.resize(num_feat, num_feat);
    Lambda_u.setIdentity();
    Lambda_m.setIdentity();

    sample_u.resize(num_feat, M.rows());
    sample_m.resize(num_feat, M.cols());
    sample_u.setZero();
    sample_m.setZero();

    // parameters of Inv-Whishart distribution (see paper for details)
    WI_u.resize(num_feat, num_feat);
    WI_u.setIdentity();
    mu0_u.resize(num_feat);
    mu0_u.setZero();

    WI_m.resize(num_feat, num_feat);
    WI_m.setIdentity();
    mu0_m.resize(num_feat);
    mu0_m.setZero();
}

inline double sqr(double x) { return x*x; }

std::pair<double,double> eval_probe_vec(int n, VectorXd & predictions, const MatrixXd &sample_m, const MatrixXd &sample_u, double mean_rating)
{
    double se = 0.0, se_avg = 0.0;
    unsigned idx = 0;
    for (int k=0; k<P.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(P,k); it; ++it) {
            const double pred = sample_m.col(it.col()).dot(sample_u.col(it.row())) + mean_rating;
            //se += (it.value() < log10(200)) == (pred < log10(200));
            se += sqr(it.value() - pred);

            const double pred_avg = (n == 0) ? pred : (predictions[idx] + (pred - predictions[idx]) / n);
            //se_avg += (it.value() < log10(200)) == (pred_avg < log10(200));
            se_avg += sqr(it.value() - pred_avg);
            predictions[idx++] = pred_avg;
        }
    }

    const unsigned N = P.nonZeros();
    const double rmse = sqrt( se / N );
    const double rmse_avg = sqrt( se_avg / N );
    return std::make_pair(rmse, rmse_avg);
}

void sample_movie(MatrixXd &s, int mm, const SparseMatrixD &mat, double mean_rating,
    const MatrixXd &samples, double alpha, const VectorXd &mu_u, const MatrixXd &Lambda_u)
{
    int i = 0;
    MatrixXd MM(num_feat, num_feat);
    MM.setZero();
    VectorXd rr(num_feat);
    rr.setZero();
    for (SparseMatrixD::InnerIterator it(mat,mm); it; ++it, ++i) {
        // cout << "M[" << it.row() << "," << it.col() << "] = " << it.value() << endl;
        auto col = samples.col(it.row());
        MM.noalias() += col * col.transpose();
        rr.noalias() += col * ((it.value() - mean_rating) * alpha);
    }

    Eigen::LLT<MatrixXd> chol = (Lambda_u + alpha * MM).llt();
    if(chol.info() != Eigen::Success) {
      throw std::runtime_error("Cholesky Decomposition failed!");
    }

    VectorXd tmp = rr + Lambda_u * mu_u;
    chol.matrixL().solveInPlace(tmp);
    for (int i = 0; i < num_feat; i++) {
      tmp[i] += prandn();
    }
    chol.matrixU().solveInPlace(tmp);
    s.col(mm) = tmp;

#ifdef TEST_SAMPLE
      cout << "movie " << mm << ":" << result.cols() << " x" << result.rows() << endl;
      cout << "mean rating " << mean_rating << endl;
      cout << "E = [" << E << "]" << endl;
      cout << "rr = [" << rr << "]" << endl;
      cout << "MM = [" << MM << "]" << endl;
      cout << "Lambda_u = [" << Lambda_u << "]" << endl;
      cout << "covar = [" << covar << "]" << endl;
      cout << "mu = [" << mu << "]" << endl;
      cout << "chol = [" << chol << "]" << endl;
      cout << "rand = [" << r <<"]" <<  endl;
      cout << "result = [" << result << "]" << endl;
#endif

}

#ifdef TEST_SAMPLE
void test() {
    MatrixXd sample_u(num_feat, M.rows());
    MatrixXd sample_m(num_feat, M.cols());

    mu_m.setZero();
    Lambda_m.setIdentity();
    sample_u.setConstant(2.0);
    Lambda_m *= 0.5;
    sample_m.col(0) = sample_movie(0, M, mean_rating, sample_u, alpha, mu_m, Lambda_m);
}

#else

void run() {
    auto start = tick();
    VectorXd predictions;
    predictions = VectorXd::Zero( P.nonZeros() );

    std::cout << "Sampling" << endl;
    for(int i=0; i<nsims; ++i) {

      // Sample from movie hyperparams
      tie(mu_m, Lambda_m) = CondNormalWishart(sample_m, mu0_m, b0_m, WI_m, df_m);

      // Sample from user hyperparams
      tie(mu_u, Lambda_u) = CondNormalWishart(sample_u, mu0_u, b0_u, WI_u, df_u);

      const int num_m = M.cols();
      const int num_u = M.rows();
#ifdef _OPENMP
#pragma omp parallel for
      for(int mm=0; mm<num_m; ++mm) {
        sample_movie(sample_m, mm, M, mean_rating, sample_u, alpha, mu_m, Lambda_m);
      }
#pragma omp parallel for
      for(int uu=0; uu<num_u; ++uu) {
        sample_movie(sample_u, uu, Mt, mean_rating, sample_m, alpha, mu_u, Lambda_u);
      }
#else
      tbb::parallel_for(0, num_m, [](int mm) {
        sample_movie(sample_m, mm, M, mean_rating, sample_u, alpha, mu_m, Lambda_m);
      });

      tbb::parallel_for(0, num_u, [](int uu) {
         sample_movie(sample_u, uu, Mt, mean_rating, sample_m, alpha, mu_u, Lambda_u);
       });
#endif

      auto eval = eval_probe_vec( (i < burnin) ? 0 : (i - burnin), predictions, sample_m, sample_u, mean_rating);
//      auto eval = std::make_pair(0.0, 0.0);
      double norm_u = sample_u.norm();
      double norm_m = sample_m.norm();
      auto end = tick();
      auto elapsed = end - start;
      double samples_per_sec = (i + 1) * (M.rows() + M.cols()) / elapsed;

      printf("Iteration %d:\t RMSE: %3.3f\tavg RMSE: %3.3f\tFU(%6.2f)\tFM(%6.2f)\tSamples/sec: %6.2f\n",
              i, eval.first, eval.second, norm_u, norm_m, samples_per_sec);
    }

  auto end = tick();
  auto elapsed = end - start;
  printf("Total time: %6.2f\n", elapsed);
}

#endif

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

/**
 * checks if command line option is present
 */
bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

void usage()
{
    printf("Usage:\n");
    printf("./bpmf [options] <train.data> <test.data>\n");
    printf("where options are:\n");
    printf(" -a alpha       precision (default 2.0)\n");
    printf(" -d num_latent  number of latent dimensions (default 32)\n");
}

int main(int argc, char *argv[])
{
    // args: bpmf [options] <train.matrix> <test.matrix>
    int optargc = argc - 3;
    if (optargc < 0) {
        printf("Need at least <train.matrix> and <test.matrix> files as arguments.\n");
        usage();
        exit(1);
    }
    // making sure matrix files exist
    char* train_matrix = argv[argc - 2];
    char* test_matrix  = argv[argc - 1];
    if ( access( train_matrix, F_OK) == -1 ) {
        printf("File for train.matrix '%s' does not exist.\n", train_matrix);
        usage();
        exit(1);
    }
    if ( access( test_matrix, F_OK) == -1 ) {
        printf("File for test.matrix '%s' does not exist.\n", test_matrix);
        usage();
        exit(1);
    }

    // fetching optional arguments
    char * tmp;
    tmp = getCmdOption(argv, argv + argc - 2, "-a");
    if (tmp) {
        alpha = atof(tmp);
    }
    tmp = getCmdOption(argv, argv + argc - 2, "-d");
    if (tmp) {
        num_feat = atoi(tmp);
    }

    printf("---- BPMF parameters ----\n");
    printf("alpha      = %1.2f\n", alpha);
    printf("num_latent = %d\n", num_feat);
    printf("train      = %s\n", train_matrix);
    printf("test       = %s\n", test_matrix);

    Eigen::initParallel();

    loadMarket(M, train_matrix);
    Mt = M.transpose();
    loadMarket(P, test_matrix);

    init();
#ifdef TEST_SAMPLE
    test();
#else
    run();
#endif

    return 0;
}
