#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>

#include <hpx/thrust/algorithms.hpp>
#include <hpx/thrust/policy.hpp>

#include <vector>

int hpx_main(int, char**)
{
    hpx::thrust::thrust_host_policy host{};
    std::vector<int> v(1024, 0);
    hpx::fill(host, v.begin(), v.end(), 42);
    return hpx::local::finalize();
}

int main(int argc, char** argv)
{
    hpx::local::init_params init_args;
    return hpx::local::init(hpx_main, argc, argv, init_args);
}
