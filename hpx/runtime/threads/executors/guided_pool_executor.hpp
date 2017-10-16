//  Copyright (c)      2017 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_GUIDED_POOL_EXECUTOR
#define HPX_RUNTIME_THREADS_GUIDED_POOL_EXECUTOR

#include <hpx/async.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/runtime/threads/detail/thread_pool_base.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/util/thread_specific_ptr.hpp>
#include <hpx/lcos/dataflow.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <iostream>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace executors
{
    struct bitmap_storage
    {
        struct tls_tag {};
        static hpx::util::thread_specific_ptr<hwloc_bitmap_ptr, tls_tag> bitmap_storage_;
    };

    // --------------------------------------------------------------------
    // Template type for a numa domain scheduling hint
    template <typename... Args>
    struct HPX_EXPORT pool_numa_hint {};

    // Template type for a core scheduling hint
    template <typename... Args>
    struct HPX_EXPORT pool_core_hint {};

    // --------------------------------------------------------------------
    // helper : numa domain scheduling and then execution
    template <typename Executor, typename NumaFunction>
    struct pre_execution_domain_schedule
    {
        Executor     &executor_;
        NumaFunction &numa_function_;
        //
        template <typename F, typename ... Ts>
        auto operator()(F && f, Ts &&... ts) const
        {
            int domain = numa_function_(ts...);
            std::cout << "The numa domain is " << domain << "\n";

            // now we must forward the task on to the correct dispatch function
            typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;

            lcos::local::futures_factory<result_type()> p(
                executor_,
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...));
            p.apply(
                launch::async,
                threads::thread_priority_default,
                threads::thread_stacksize_default,
                threads::thread_schedule_hint(domain));
            return p.get_future();
        }
    };

    // --------------------------------------------------------------------
    struct HPX_EXPORT guided_pool_executor_base {
    public:
        guided_pool_executor_base(const std::string& pool_name)
            : pool_executor_(pool_name)
        {}

        guided_pool_executor_base(const std::string& pool_name,
                             thread_stacksize stacksize)
            : pool_executor_(pool_name, stacksize)
        {}

        guided_pool_executor_base(const std::string& pool_name,
                             thread_priority priority,
                             thread_stacksize stacksize = thread_stacksize_default)
            : pool_executor_(pool_name, priority, stacksize)
        {}

    protected:
        pool_executor pool_executor_;
    };


    // --------------------------------------------------------------------
    template <typename... Args>
    struct HPX_EXPORT guided_pool_executor {};

    // --------------------------------------------------------------------
    // this is a guided pool executor templated over a function type
    // the function type should be the one used for async calls
    template <>
    template <typename R, typename...Args>
    struct HPX_EXPORT guided_pool_executor<pool_numa_hint<R(*)(Args...)>>
        : guided_pool_executor_base
    {
    public:
        using guided_pool_executor_base::guided_pool_executor_base;

        template <typename F, typename ... Ts>
        hpx::future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F && f, Ts &&... ts)
        {
            // hold onto the function until all futures have become ready
            // by using a dataflow operation, then call the scheduling hint
            // before passing the task onwards to the real executor
            return hpx::dataflow(
                util::unwrapping(
                    pre_execution_domain_schedule<pool_executor,
                        pool_numa_hint<R(*)(Args...)>>{
                            pool_executor_, hint_
                        }
                ),
                std::forward<F>(f), std::forward<Ts>(ts)...);
        };

    private:
        pool_numa_hint<R(*)(Args...)> hint_;
    };

    // --------------------------------------------------------------------
    // this is a guided pool executor templated over args only
    // the args should be the same as those that would be called
    // for an async function or continuation. This makes it possible to
    // guide a lambda rather than a full function.
    template <>
    template <typename...Args>
    struct HPX_EXPORT guided_pool_executor<pool_numa_hint<Args...>>
        : guided_pool_executor_base
    {
    public:
        using guided_pool_executor_base::guided_pool_executor_base;

        template <typename F, typename ... Ts>
        hpx::future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F && f, Ts &&... ts)
        {
            // hold onto the function until all futures have become ready
            // by using a dataflow operation, then call the scheduling hint
            // before passing the task onwards to the real executor
            return hpx::dataflow(
                util::unwrapping(
                    pre_execution_domain_schedule<pool_executor,
                        pool_numa_hint<Args...>> {
                            pool_executor_, hint_
                        }
                ),
                std::forward<F>(f), std::forward<Ts>(ts)...);
        };

    private:
        pool_numa_hint<Args...> hint_;
    };

}}}

namespace hpx { namespace parallel { namespace execution
{
    template <typename Executor>
    struct executor_execution_category<
        threads::executors::guided_pool_executor<Executor> >
    {
        typedef parallel::execution::parallel_execution_tag type;
    };

    template <typename Executor>
    struct is_one_way_executor<
            threads::executors::guided_pool_executor<Executor> >
      : std::true_type
    {};

    template <typename Executor>
    struct is_two_way_executor<
            threads::executors::guided_pool_executor<Executor> >
      : std::true_type
    {};

    template <typename Executor>
    struct is_bulk_one_way_executor<
            threads::executors::guided_pool_executor<Executor> >
      : std::false_type
    {};

    template <typename Executor>
    struct is_bulk_two_way_executor<
            threads::executors::guided_pool_executor<Executor> >
      : std::false_type
    {};

}}}

#include <hpx/config/warnings_suffix.hpp>

#endif /*HPX_RUNTIME_THREADS_GUIDED_POOL_EXECUTOR*/
