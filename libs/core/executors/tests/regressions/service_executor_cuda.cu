//  Copyright (c) 2021 Gregor Daiss
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution/execution.hpp>
#include <hpx/executors/service_executors.hpp>

int main()
{
    hpx::parallel::execution::detail::service_executor exec{nullptr};
    hpx::parallel::execution::async_execute(exec, []{}).get();
}
