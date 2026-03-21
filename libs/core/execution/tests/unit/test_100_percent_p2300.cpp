//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0

#include <hpx/config.hpp>

#if !defined(HPX_HAVE_STDEXEC)

#include <hpx/execution/algorithms/just.hpp>
#include <hpx/execution/algorithms/sync_wait.hpp>
#include <hpx/execution/algorithms/then.hpp>
#include <hpx/execution/algorithms/upon_error.hpp>
#include <hpx/execution/algorithms/upon_stopped.hpp>
#include <hpx/modules/testing.hpp>

#include <exception>

namespace ex = hpx::execution::experimental;

int main()
{
    {
        auto result = ex::sync_wait(
            ex::upon_error(ex::just(42), [](std::exception_ptr) { return 0; }));
        HPX_TEST(result.has_value());
        HPX_TEST_EQ(hpx::get<0>(*result), 42);
    }

    {
        auto result =
            ex::sync_wait(ex::upon_stopped(ex::just(42), []() { return 0; }));
        HPX_TEST(result.has_value());
        HPX_TEST_EQ(hpx::get<0>(*result), 42);
    }

    return hpx::util::report_errors();
}

#else

int main()
{
    return 0;
}

#endif
