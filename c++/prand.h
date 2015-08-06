#ifndef PRAND_H
#define PRAND_H

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

#endif
