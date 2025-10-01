#include <hpx/contracts.hpp>
#include <hpx/modules/testing.hpp>
#include <iostream>
int main()
{
    HPX_PRE(true);
    HPX_CONTRACT_ASSERT(true);
    HPX_POST(true);
    
    // Add a failing assertion to test WILL_FAIL behavior
    HPX_CONTRACT_ASSERT(false);  // This should abort in Debug mode

    HPX_TEST(true);

    return hpx::util::report_errors();
}