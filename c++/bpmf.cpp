
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>

#include <unsupported/Eigen/SparseExtra>

#ifdef _OPENMP
#include <omp.h>
#else
#include <tbb/tbb.h>
#endif

#include "bpmf.h"
#include "prand.h"

using namespace std;
using namespace Eigen;

// parallel normal random
RNG prandn;

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

static MatrixXd* sm_MM;
static VectorXd* sm_rr;
static VectorXd* sm_tmp;
#pragma omp threadprivate(sm_MM, sm_rr, sm_tmp)

// parameters of Inv-Whishart distribution (see paper for details)
MatrixXd WI_u;
int b0_u = 2;
int df_u = num_feat;
VectorXd mu0_u;

MatrixXd WI_m;
int b0_m = 2;
int df_m = num_feat;
VectorXd mu0_m;

void init() {
    mean_rating = M.sum() / M.nonZeros();
    Lambda_u.resize(num_feat, num_feat);
    Lambda_m.resize(num_feat, num_feat);
    Lambda_u.setIdentity();
    Lambda_m.setIdentity();
    Lambda_u *= 10;
    Lambda_m *= 10;

    mu_u.resize(num_feat);
    mu_u.setZero();
    mu_m.resize(num_feat);
    mu_m.setZero();

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

    // initializing sample_movies variables
#pragma omp parallel
    {
      sm_MM = new MatrixXd(num_feat, num_feat);
      sm_MM->setZero();
      sm_rr = new VectorXd(num_feat);
      sm_rr->setZero();
      sm_tmp = new VectorXd(num_feat);
    }
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
//    MatrixXd MM(num_feat, num_feat);
    sm_MM->setZero();
//    VectorXd rr(num_feat);
    sm_rr->setZero();
    for (SparseMatrixD::InnerIterator it(mat,mm); it; ++it) {
        // cout << "M[" << it.row() << "," << it.col() << "] = " << it.value() << endl;
        auto col = samples.col(it.row());
        sm_MM->noalias() += col * col.transpose();
        sm_rr->noalias() += col * ((it.value() - mean_rating) * alpha);
    }

    Eigen::LLT<MatrixXd> chol = (Lambda_u + alpha * (*sm_MM)).llt();
    if(chol.info() != Eigen::Success) {
      throw std::runtime_error("Cholesky Decomposition failed!");
    }

    sm_tmp->noalias() = (*sm_rr) + Lambda_u * mu_u;
    chol.matrixL().solveInPlace(*sm_tmp);
    for (int i = 0; i < num_feat; i++) {
      (*sm_tmp)[i] += prandn();
    }
    chol.matrixU().solveInPlace(*sm_tmp);
    s.col(mm).noalias() = *sm_tmp;

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

      // Sample from movie hyperparams
      tie(mu_m, Lambda_m) = CondNormalWishart(sample_m, mu0_m, b0_m, WI_m, df_m);

      // Sample from user hyperparams
      tie(mu_u, Lambda_u) = CondNormalWishart(sample_u, mu0_u, b0_u, WI_u, df_u);

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

void usage()
{
    printf("Usage:\n");
    printf("./bpmf [options] <train.matrix> <test.matrix>\n");
    printf("where options are:\n");
    printf("  --alpha 2.0       precision (default 2.0)\n");
    printf("  --num-latent 32   number of latent dimensions (default 32)\n");
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
    tmp = getCmdOption(argv, argv + argc - 2, "--alpha");
    if (tmp) {
        alpha = atof(tmp);
    }
    tmp = getCmdOption(argv, argv + argc - 2, "--num-latent");
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
