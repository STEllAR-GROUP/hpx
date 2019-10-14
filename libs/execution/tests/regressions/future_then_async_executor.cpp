//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/testing.hpp>

#include <type_traits>
#include <utility>

struct test_async_executor
{
    using execution_category = hpx::parallel::execution::parallel_execution_tag;

    template <typename F, typename... Ts>
    static hpx::future<typename hpx::util::invoke_result<F, Ts...>::type>
    async_execute(F&& f, Ts&&... ts)
    {
        return hpx::dataflow(
            hpx::launch::async, std::forward<F>(f), std::forward<Ts>(ts)...);
    }
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<test_async_executor> : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

int main()
{
    test_async_executor exec;
    hpx::future<void> f = hpx::make_ready_future();
    f.then(exec, [](hpx::future<void>&& f) { f.get(); });

    return hpx::util::report_errors();
}
