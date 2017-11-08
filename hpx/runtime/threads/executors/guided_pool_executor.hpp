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
#include <hpx/util/invoke.hpp>
#include <hpx/util/pack_traversal.hpp>
#include <hpx/util/demangle_helper.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <iostream>

#include <hpx/config/warnings_prefix.hpp>

// #define GUIDED_EXECUTOR_DEBUG 1

// --------------------------------------------------------------------
// pool_numa_hint
// --------------------------------------------------------------------
namespace hpx { namespace threads { namespace executors
{
    // --------------------------------------------------------------------
    // helper structs to make future<tuple<f1, f2, f3, ...>>>
    // detection of futures simpler
    // --------------------------------------------------------------------
    template <typename TupleOfFutures>
    struct is_tuple_of_futures;

    template <typename...Futures>
    struct is_tuple_of_futures<util::tuple<Futures...>>
        : util::detail::all_of<traits::is_future<Futures>...>
    {};

    template <typename Future>
    struct is_future_of_tuple_of_futures
        : std::integral_constant<bool,
            traits::is_future<Future>::value &&
            is_tuple_of_futures<
                typename traits::future_traits<Future>::result_type>::value>
    {};

    // --------------------------------------------------------------------
    // function that returns a const ref to the contents of a future
    // without calling .get() on the future so that we can use the value
    // and then pass the original future on to the intended destination.
    // --------------------------------------------------------------------
    struct future_extract_value
    {
        template<typename T, template <typename> typename Future>
        const T& operator()(const Future<T> &el) const
        {
            const auto & state = traits::detail::get_shared_state(el);
            return *state->get_result();
        }
    };

    // --------------------------------------------------------------------
    // Template type for a numa domain scheduling hint
    template <typename... Args>
    struct HPX_EXPORT pool_numa_hint {};

    // Template type for a core scheduling hint
    template <typename... Args>
    struct HPX_EXPORT pool_core_hint {};

    // --------------------------------------------------------------------
    // helper : numa domain scheduling for async() execution
    // --------------------------------------------------------------------
    template <typename Executor, typename NumaFunction>
    struct pre_execution_async_domain_schedule
    {
        Executor     executor_;
        NumaFunction numa_function_;
        //
        template <typename F, typename ... Ts>
        auto operator()(F && f, Ts &&... ts) const
        {
            // call the numa hint function
            int domain = numa_function_(ts...);

            // now we must forward the task+hint on to the correct dispatch function
            typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;

            lcos::local::futures_factory<result_type()> p(
                const_cast<Executor&>(executor_),
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
    // helper : numa domain scheduling for .then() execution
    // this differs from the above because a future is unwrapped before
    // calling the numa_hint
    // --------------------------------------------------------------------
    template <typename Executor, typename NumaFunction>
    struct pre_execution_then_domain_schedule
    {
        Executor     executor_;
        NumaFunction numa_function_;
        //
        template <typename F, template <typename> typename Future, typename P, typename ... Ts>
        auto operator()(F && f, Future<P> && predecessor, Ts &&... ts)
        {
            // get the argument for the numa hint function from the predecessor future
            const auto & predecessor_value = future_extract_value().operator()(predecessor);

            // call the numa hint function
            int domain = numa_function_(predecessor_value, ts...);

            // now we must forward the task+hint on to the correct dispatch function
            typedef typename
                util::detail::invoke_deferred_result<F, Future<P>, Ts...>::type
                        result_type;

            lcos::local::futures_factory<result_type()> p(
                const_cast<Executor&>(executor_),
                util::deferred_call(std::forward<F>(f), std::move(predecessor),
                                    std::forward<Ts>(ts)...)
            );

            p.apply(
                launch::async,
                threads::thread_priority_default,
                threads::thread_stacksize_default,
                threads::thread_schedule_hint(domain));

            return p.get_future();
        }
    };

    // --------------------------------------------------------------------
    // base class for guided executors
    // these differ by the numa_hint fnuction type that is called
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

        scheduled_executor &get_scheduled_executor() {
            return pool_executor_;
        }
    protected:
        pool_executor pool_executor_;
    };

    // --------------------------------------------------------------------
    template <typename H>
    struct HPX_EXPORT guided_pool_executor {};

    // --------------------------------------------------------------------
    // this is a guided pool executor templated over args only
    // the args should be the same as those that would be called
    // for an async function or continuation. This makes it possible to
    // guide a lambda rather than a full function.
    template <typename H, typename Tag>
    struct HPX_EXPORT guided_pool_executor<pool_numa_hint<H,Tag>>
        : guided_pool_executor_base
    {
    public:
        // force usage of base class constructors
        using guided_pool_executor_base::guided_pool_executor_base;

        // --------------------------------------------------------------------
        // async execute specialized for simple arguments typical
        // of a normal async call with arbitrary arguments
        // --------------------------------------------------------------------
        template <typename F, typename ... Ts>
        future<typename util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F && f, Ts &&... ts)
        {
#ifdef GUIDED_EXECUTOR_DEBUG
            typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;

            std::cout << "async_execute : Function    : "
                      << debug::print_type<F>() << "\n";
            std::cout << "async_execute : Arguments   : "
                      << debug::print_type<Ts...>(" | ") << "\n";
            std::cout << "async_execute : Result      : "
                      << debug::print_type<result_type>() << "\n";
            std::cout << "async_execute : Numa Hint   : "
                      << debug::print_type<pool_numa_hint<H,Tag>>() << "\n";
            std::cout << "async_execute : Hint   : "
                      << debug::print_type<H>() << "\n";
#endif

            // hold onto the function until all futures have become ready
            // by using a dataflow operation, then call the scheduling hint
            // before passing the task onwards to the real executor
            return dataflow(
                util::unwrapping(
                    pre_execution_async_domain_schedule<pool_executor,
                        pool_numa_hint<H,Tag>> {
                            pool_executor_, hint_
                        }
                ),
                std::forward<F>(f), std::forward<Ts>(ts)...
            );
        }

        // --------------------------------------------------------------------
        // .then() execute specialized for a future<P> predecessor argument
        // note that future<> and shared_future<> are both supported
        // --------------------------------------------------------------------
        template <typename F,
                  typename Future,
                  typename ... Ts,
                  typename = typename std::enable_if_t<traits::is_future<Future>::value>
                  >
        auto
        then_execute(F && f, Future & predecessor, Ts &&... ts)
        ->  future<typename util::detail::invoke_deferred_result<
            F, Future, Ts...>::type>
        {
#ifdef GUIDED_EXECUTOR_DEBUG
            typedef typename util::detail::invoke_deferred_result<
                    F, Future, Ts...>::type result_type;

            std::cout << "then_execute : Function     : "
                      << debug::print_type<F>() << "\n";
            std::cout << "then_execute : Predecessor  : "
                      << debug::print_type<Future>() << "\n";
            std::cout << "then_execute : Future       : "
                      << debug::print_type<typename
                         traits::future_traits<Future>::result_type>() << "\n";
            std::cout << "then_execute : Arguments    : "
                      << debug::print_type<Ts...>(" | ") << "\n";
            std::cout << "then_execute : Result       : "
                      << debug::print_type<result_type>() << "\n";
#endif

            // Note : The Ts &&... args are not actually used in a continuation since
            // only the future becoming ready (predecessor) is actually passed onwards.
            // Note : we do not need to use unwrapping here, because continuations
            // are only invoked once the futures are already ready
            return dataflow(
                [&](Future && predecessor, Ts &&... ts)
                {
                    pre_execution_then_domain_schedule<pool_executor,
                        pool_numa_hint<H,Tag>>
                        pre_exec { pool_executor_, hint_ };

                    return pre_exec.operator ()(
                        std::forward<F>(f), std::forward<Future>(predecessor));
                },
                std::move(predecessor), std::forward<Ts>(ts)...
            );
        }

        // --------------------------------------------------------------------
        // .then() execute specialized for a when_all dispatch for any future types
        // future< tuple< is_future<a>::type, is_future<b>::type, ...> >
        // --------------------------------------------------------------------
        template <typename F,
                  template <typename> typename  OuterFuture,
                  typename ... InnerFutures,
                  typename ... Ts,
                  typename = typename std::enable_if_t<is_future_of_tuple_of_futures<
                    OuterFuture<util::tuple<InnerFutures...>>>::value>,
                  typename = typename std::enable_if_t<is_tuple_of_futures<
                    util::tuple<InnerFutures...>>::value>
                  >
        auto
        then_execute(F && f,
                     OuterFuture<util::tuple<InnerFutures... > > && predecessor,
                     Ts &&... ts)
        ->  future<typename util::detail::invoke_deferred_result<
            F, OuterFuture<util::tuple<InnerFutures... >>, Ts...>::type>
        {
            // get the tuple of futures from the predecessor future <tuple of futures>
            const auto & predecessor_value = future_extract_value().operator()(predecessor);

            // create a tuple of the unwrapped future values
            auto unwrapped_futures_tuple = util::map_pack(
                future_extract_value{},
                predecessor_value
            );

#ifdef GUIDED_EXECUTOR_DEBUG
            typedef typename util::detail::invoke_deferred_result<
                F, OuterFuture<util::tuple<InnerFutures... >>, Ts...>::type
                    result_type;

            std::cout << "when_all(fut) : Predecessor : "
                      << debug::print_type<OuterFuture<util::tuple<InnerFutures...>>>()
                      << "\n";
            std::cout << "when_all(fut) : unwrapped   : "
                      << debug::print_type<decltype(unwrapped_futures_tuple)>(" | ") << "\n";
            std::cout << "then_execute  : Arguments   : "
                      << debug::print_type<Ts...>(" | ") << "\n";
            std::cout << "when_all(fut) : Result      : "
                      << debug::print_type<result_type>() << "\n";
#endif

            return dataflow(
                [&](OuterFuture<util::tuple<InnerFutures...>> && predecessor, Ts &&... ts)
                {
                    pre_execution_then_domain_schedule<pool_executor,
                        pool_numa_hint<H,Tag>>
                        pre_exec { pool_executor_, hint_ };

                    return pre_exec.operator ()(
                        std::forward<F>(f),
                        std::forward<OuterFuture<util::tuple<InnerFutures...>>>
                            (predecessor));
                },
                std::move(predecessor), std::forward<Ts>(ts)...
            );
        }

        // --------------------------------------------------------------------
        // execute specialized for a dataflow dispatch
        // dataflow unwraps the outer future for us but passes a dataflowframe
        // function type, result type and tuple of futures as arguments
        // --------------------------------------------------------------------
        template <typename F,
                  typename DataFlowFrame,
                  typename Result,
                  typename ... InnerFutures,
                  typename = typename std::enable_if_t<
                      is_tuple_of_futures<util::tuple<InnerFutures...>>::value>
                  >
        auto
        async_execute(F && f,
                      DataFlowFrame && df,
                      Result && r,
                      util::tuple<InnerFutures... > && predecessor)
        ->  future<typename util::detail::invoke_deferred_result<
            F, DataFlowFrame, Result, util::tuple<InnerFutures... >>::type>
        {
            typedef typename util::detail::invoke_deferred_result<
                F, DataFlowFrame, Result, util::tuple<InnerFutures... >>::type
                    result_type;

            auto unwrapped_futures_tuple = util::map_pack(
                future_extract_value{},
                predecessor
            );

#ifdef GUIDED_EXECUTOR_DEBUG
            std::cout << "dataflow      : Predecessor : "
                      << debug::print_type<util::tuple<InnerFutures...>>()
                      << "\n";
            std::cout << "dataflow      : unwrapped   : "
                      << debug::print_type<decltype(unwrapped_futures_tuple)>(" | ") << "\n";
            std::cout << "dataflow-frame: Result      : "
                      << debug::print_type<Result>() << "\n";
#endif

            // invoke the hint function with the unwrapped tuple futures
            int domain = util::invoke_fused(hint_, unwrapped_futures_tuple);

            // forward the task execution on to the real internal executor
            lcos::local::futures_factory<result_type()> p(
                pool_executor_,
                util::deferred_call(
                    std::forward<F>(f),
                    std::forward<DataFlowFrame>(df),
                    std::forward<Result>(r),
                    std::forward<util::tuple<InnerFutures...>>(predecessor)
                )
            );

            p.apply(
                launch::async,
                threads::thread_priority_default,
                threads::thread_stacksize_default,
                threads::thread_schedule_hint(domain));

            return p.get_future();
        }

    private:
        pool_numa_hint<H,Tag> hint_;
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
