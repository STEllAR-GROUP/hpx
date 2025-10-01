#include <hpx/contracts.hpp>
#include <hpx/modules/testing.hpp>

static bool assertion_handler_called = false;

// Custom assertion handler to detect when HPX_ASSERT is called
void test_assertion_handler(hpx::source_location const& /*loc*/,
    const char* /*expr*/, std::string const& /*msg*/)
{
    assertion_handler_called = true;
    // Don't abort - just record that the handler was called
}

int main()
{
    // Set our custom assertion handler to detect HPX_ASSERT calls
    hpx::assertion::set_assertion_handler(&test_assertion_handler);

    // Test 1: Should always pass (no assertion triggered)
    HPX_CONTRACT_ASSERT(true);
    HPX_TEST(!assertion_handler_called);  // Handler should NOT be called

    // Test 2: Should trigger HPX_ASSERT in fallback mode
    assertion_handler_called = false;
    HPX_CONTRACT_ASSERT(false);  // This should call our handler

    // Verify that the fallback to HPX_ASSERT actually happened
    HPX_TEST(assertion_handler_called);

    return hpx::util::report_errors();
}