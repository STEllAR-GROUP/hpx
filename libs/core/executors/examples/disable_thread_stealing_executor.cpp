//  Copyright (c) 2020-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
// The purpose of this example is to show how to write an executor disables
// thread stealing for the duration of the execution of a parallel algorithm
// it is used with.

#include <hpx/algorithm.hpp>
#include <hpx/assert.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>

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

        template <typename Executor,
            typename Enable = std::enable_if_t<!std::is_same_v<
                std::decay_t<Executor>, disable_thread_stealing_executor>>>
        explicit disable_thread_stealing_executor(Executor&& exec)
          : BaseExecutor(std::forward<Executor>(exec))
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
                hpx::threads::policies::scheduler_mode::enable_stealing);
        }

        template <typename Parameters>
        static void mark_end_execution(Parameters&&)
        {
            hpx::threads::add_scheduler_mode(
                hpx::threads::policies::scheduler_mode::enable_stealing);
        }
    };

    // support all properties exposed by the wrapped executor
    template <typename Tag, typename BaseExecutor, typename Property>
        requires(hpx::execution::experimental::is_scheduling_property_v<Tag>)
    auto tag_invoke(Tag tag,
        disable_thread_stealing_executor<BaseExecutor> const& exec,
        Property&& prop)
        -> decltype(disable_thread_stealing_executor<BaseExecutor>(
            std::declval<Tag>()(
                std::declval<BaseExecutor>(), std::declval<Property>())))
    // clang-format on
    {
        return disable_thread_stealing_executor<BaseExecutor>(
            tag(static_cast<BaseExecutor const&>(exec),
                HPX_FORWARD(Property, prop)));
    }

    template <typename Tag, typename BaseExecutor>
        requires(hpx::execution::experimental::is_scheduling_property_v<Tag>)
    auto tag_invoke(
        Tag tag, disable_thread_stealing_executor<BaseExecutor> const& exec)
        -> decltype(std::declval<Tag>()(std::declval<BaseExecutor>()))
    {
        return tag(static_cast<BaseExecutor const&>(exec));
    }

    template <typename BaseExecutor>
    auto make_disable_thread_stealing_executor(BaseExecutor&& exec)
    {
        return disable_thread_stealing_executor<std::decay_t<BaseExecutor>>(
            std::forward<BaseExecutor>(exec));
    }
}    // namespace executor_example

///////////////////////////////////////////////////////////////////////////////
// simple forwarding implementations of executor traits
namespace hpx::execution::experimental {

    template <typename BaseExecutor>
    struct is_one_way_executor<
        executor_example::disable_thread_stealing_executor<BaseExecutor>>
      : is_one_way_executor<std::decay_t<BaseExecutor>>
    {
    };

    template <typename BaseExecutor>
    struct is_never_blocking_one_way_executor<
        executor_example::disable_thread_stealing_executor<BaseExecutor>>
      : is_never_blocking_one_way_executor<std::decay_t<BaseExecutor>>
    {
    };

    template <typename BaseExecutor>
    struct is_two_way_executor<
        executor_example::disable_thread_stealing_executor<BaseExecutor>>
      : is_two_way_executor<std::decay_t<BaseExecutor>>
    {
    };

    template <typename BaseExecutor>
    struct is_bulk_one_way_executor<
        executor_example::disable_thread_stealing_executor<BaseExecutor>>
      : is_bulk_one_way_executor<std::decay_t<BaseExecutor>>
    {
    };

    template <typename BaseExecutor>
    struct is_bulk_two_way_executor<
        executor_example::disable_thread_stealing_executor<BaseExecutor>>
      : is_bulk_two_way_executor<std::decay_t<BaseExecutor>>
    {
    };
}    // namespace hpx::execution::experimental

int hpx_main()
{
    std::vector<double> v(1000);
    std::iota(v.begin(), v.end(), 0.0);

    // The following for_loop will be executed while thread stealing is disabled
    auto exec = executor_example::make_disable_thread_stealing_executor(
        hpx::execution::par.executor());

    hpx::experimental::for_loop(
        hpx::execution::par.on(exec), 0, v.size(), [](std::size_t) {});

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::local::init(hpx_main, argc, argv);
}
