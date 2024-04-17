//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#ifdef HPX_HAVE_STDEXEC
#include <hpx/execution/algorithms/just.hpp>
#else
#include <hpx/modules/execution.hpp>
#endif
#include <hpx/modules/testing.hpp>

#include "algorithm_test_utils.hpp"

#include <atomic>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace ex = hpx::execution::experimental;

int main()
{
    {
        std::atomic<bool> set_stopped_called{false};
        auto s = ex::just_stopped();

        static_assert(ex::is_sender_v<decltype(s)>);
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);

        check_value_types<hpx::variant<>>(s);
#ifdef HPX_HAVE_STDEXEC
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<true>(s);

        auto r = expect_stopped_receiver{set_stopped_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_stopped_called);
    }

    return hpx::util::report_errors();
}
