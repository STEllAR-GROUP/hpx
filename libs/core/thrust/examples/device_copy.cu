#include <hpx/init.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>

#include <hpx/thrust/algorithms.hpp>
#include <hpx/thrust/policy.hpp>

#include <thrust/device_vector.h>

int hpx_main(int, char**)
{
    hpx::thrust::thrust_device_policy device{};
    thrust::device_vector<int> src(1024, 5);
    thrust::device_vector<int> dst(1024, 0);
    hpx::copy(device, src.begin(), src.end(), dst.begin());
    return hpx::local::finalize();
}

int main(int argc, char** argv)
{
    hpx::local::init_params init_args;
    return hpx::local::init(hpx_main, argc, argv, init_args);
}



