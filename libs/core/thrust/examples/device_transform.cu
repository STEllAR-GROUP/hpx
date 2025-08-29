#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>

#include <hpx/thrust/algorithms.hpp>
#include <hpx/thrust/policy.hpp>

#include <thrust/device_vector.h>

int hpx_main(int, char**)
{
    hpx::thrust::thrust_device_policy device{};
    thrust::device_vector<int> a(1024, 2);
    thrust::device_vector<int> b(1024, 3);
    thrust::device_vector<int> out(1024, 0);

    hpx::transform(device, a.begin(), a.end(), b.begin(), out.begin(),
        [] __device__(int x, int y) { return x + y; });
    return hpx::local::finalize();
}

int main(int argc, char** argv)
{
    hpx::local::init_params init_args;
    return hpx::local::init(hpx_main, argc, argv, init_args);
}
