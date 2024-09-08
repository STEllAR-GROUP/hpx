//  Copyright (c) 2024 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/testing.hpp>

#include <hpx/config.hpp>
#include <utility>

#if defined(HPX_HAVE_STDEXEC)
#include <hpx/execution_base/stdexec_forward.hpp>

int main()
{
    auto x = hpx::execution::experimental::just(42);

    auto [a] = hpx::execution::experimental::sync_wait(std::move(x)).value();

    HPX_TEST(a == 42);

    return hpx::util::report_errors();
}
#else
int main() {}
#endif
