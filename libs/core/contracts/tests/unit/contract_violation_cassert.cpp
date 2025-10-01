#include <hpx/modules/contracts.hpp>
#include <hpx/modules/testing.hpp>

static bool handler_called = false;

void handle_contract_violation(const std::contracts::contract_violation& violation)
{
    if (violation.semantic() == std::contracts::evaluation_semantic::observe)
    {
        handler_called = true; // record call in observe mode
    }
    else
    {
        // fallback for enforce/quick_enforce/ignore modes
        invoke_default_contract_violation_handler(violation);
    }
}

int f(int x)
{
    HPX_CONTRACT_ASSERT(false); // This will cause a contract assertion violation when x <= 0
    return x;
}

int main()
{
    std::contracts::set_default_contract_violation_handler(&handle_contract_violation);

    if (std::contracts::get_default_contract_semantic() ==
        std::contracts::evaluation_semantic::observe)
    {
        handler_called = false;
        (void) f(0); // This should trigger contract assertion violation since -1 <= 0
        HPX_TEST(handler_called);
    }

    return hpx::util::report_errors();
}