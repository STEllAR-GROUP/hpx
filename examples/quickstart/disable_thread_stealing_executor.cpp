//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
// The purpose of this example is to show how to write an executor disables
// thread stealing for the duration of the execution of a parallel algorithm
// it is used with.

#include <hpx/hpx_main.hpp>

#include <hpx/algorithm.hpp>
#include <hpx/assert.hpp>
#include <hpx/execution.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace executor_example {

    template <typename BaseExecutor>
    class disable_thread_stealing_executor : public BaseExecutor
    {
    public:
        using execution_category = typename BaseExecutor::execution_category;
        using executor_parameters_type =
            typename BaseExecutor::executor_parameters_type;

        explicit disable_thread_stealing_executor(BaseExecutor& exec)
          : BaseExecutor(exec)
        {
        }

        disable_thread_stealing_executor const& context() const noexcept
        {
            return *this;
        }

        // Add two executor API functions that will be called before the
        // parallel algorithm starts executing and after it has finished
        // executing.
        //
        // Note that this method can cause problems if two parallel algorithms
        // are executed concurrently.
        template <typename Parameters>
        static void mark_begin_execution(Parameters&&)
        {
            hpx::threads::remove_scheduler_mode(
                hpx::threads::policies::enable_stealing);
        }

        template <typename Parameters>
        static void mark_end_execution(Parameters&&)
        {
            hpx::threads::add_scheduler_mode(
                hpx::threads::policies::enable_stealing);
        }
    };

    template <typename BaseExecutor>
    disable_thread_stealing_executor<BaseExecutor>
    make_disable_thread_stealing_executor(BaseExecutor& exec)
    {
        return disable_thread_stealing_executor<BaseExecutor>(exec);
    }
}    // namespace executor_example

///////////////////////////////////////////////////////////////////////////////
// simple forwarding implementations of executor traits
namespace hpx { namespace parallel { namespace execution {

    template <typename BaseExecutor>
    struct is_one_way_executor<
        executor_example::disable_thread_stealing_executor<BaseExecutor>>
      : is_one_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_never_blocking_one_way_executor<
        executor_example::disable_thread_stealing_executor<BaseExecutor>>
      : is_never_blocking_one_way_executor<
            typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_two_way_executor<
        executor_example::disable_thread_stealing_executor<BaseExecutor>>
      : is_two_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_bulk_one_way_executor<
        executor_example::disable_thread_stealing_executor<BaseExecutor>>
      : is_bulk_one_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_bulk_two_way_executor<
        executor_example::disable_thread_stealing_executor<BaseExecutor>>
      : is_bulk_two_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };
}}}    // namespace hpx::parallel::execution

int main()
{
    std::vector<double> v(1000);
    std::iota(v.begin(), v.end(), 0.0);

    // The following for_loop will be executed while thread stealing is disabled
    auto exec = executor_example::make_disable_thread_stealing_executor(
        hpx::execution::par.executor());

    hpx::for_loop(
        hpx::execution::par.on(exec), 0, v.size(), [](std::size_t) {});

    return 0;
}
