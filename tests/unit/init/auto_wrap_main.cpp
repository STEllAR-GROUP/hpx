#include <hpx/modules/async_distributed.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/runtime_local/runtime_local.hpp>

int main(int, char**) {
    // The wrapper should have initialized HPX runtime before reaching main.
    HPX_TEST(hpx::get_runtime_ptr() != nullptr);

    // Verify that calling thread is an HPX thread by launching another 
    // HPX task and waiting for its result.
    auto f = hpx::async([] { return (hpx::get_runtime_ptr() != nullptr);});
    HPX_TEST(f.get());

    return hpx::util::report_errors();

}