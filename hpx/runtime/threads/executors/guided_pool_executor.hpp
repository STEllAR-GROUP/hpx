//  Copyright (c) 2017-2018 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_GUIDED_POOL_EXECUTOR
#define HPX_RUNTIME_THREADS_GUIDED_POOL_EXECUTOR

#include <hpx/async.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/runtime/threads/thread_pool_base.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/util/thread_specific_ptr.hpp>
#include <hpx/lcos/dataflow.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/pack_traversal.hpp>
#include <hpx/util/debug/demangle_helper.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <iostream>
#include <type_traits>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

//#define GUIDED_EXECUTOR_DEBUG 1
//#define GUIDED_POOL_EXECUTOR_FAKE_NOOP

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
    // For C++11 compatibility
    template <bool B, typename T = void>
    using enable_if_t = typename std::enable_if<B, T>::type;

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
        Executor     &executor_;
        NumaFunction &numa_function_;
        bool          hp_sync_;
        //
        template <typename F, typename ... Ts>
        auto operator()(F && f, Ts &&... ts) const
        {
            // call the numa hint function
#ifdef GUIDED_POOL_EXECUTOR_FAKE_NOOP
            int domain = -1;
#else
            int domain = numa_function_(ts...);
#endif

#ifdef GUIDED_EXECUTOR_DEBUG
            std::cout << "pre_execution_async_domain_schedule : domain "
                      << domain << std::endl;
#endif
            // now we must forward the task+hint on to the correct dispatch function
            typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;

            lcos::local::futures_factory<result_type()> p(
                const_cast<Executor&>(executor_),
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...));

            if (hp_sync_ &&
                    executor_.get_priority() == hpx::threads::thread_priority_high) {
                p.apply(
                    hpx::launch::sync,
                    executor_.get_priority(),
                    executor_.get_stacksize(),
                    threads::thread_schedule_hint(
                                thread_schedule_hint_mode_numa, domain));
            }
            else {
                p.apply(
                    hpx::launch::async,
                    executor_.get_priority(),
                    executor_.get_stacksize(),
                    threads::thread_schedule_hint(
                                thread_schedule_hint_mode_numa, domain));
            }

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
        Executor     &executor_;
        NumaFunction &numa_function_;
        bool          hp_sync_;
        //
        template <typename F, typename Future, typename ... Ts>
        auto operator()(F && f, Future && predecessor, Ts &&... ts) const
        {
            // call the numa hint function
#ifdef GUIDED_POOL_EXECUTOR_FAKE_NOOP
            int domain = -1;
#else
            // get the argument for the numa hint function from the predecessor future
            const auto & predecessor_value = future_extract_value()(predecessor);
            int domain = numa_function_(predecessor_value, ts...);
#endif

#ifdef GUIDED_EXECUTOR_DEBUG
            std::cout << "pre_execution_then_domain_schedule : domain "
                      << domain << std::endl;
#endif
            // now we must forward the task+hint on to the correct dispatch function
            typedef typename
                util::detail::invoke_deferred_result<F, Future, Ts...>::type
                        result_type;

            lcos::local::futures_factory<result_type()> p(
                const_cast<Executor&>(executor_),
                util::deferred_call(std::forward<F>(f),
                                    std::forward<Future>(predecessor),
                                    std::forward<Ts>(ts)...)
            );

            if (hp_sync_ &&
                    executor_.get_priority() == hpx::threads::thread_priority_high) {
                p.apply(
                    hpx::launch::sync,
                    executor_.get_priority(),
                    executor_.get_stacksize(),
                    threads::thread_schedule_hint(
                                thread_schedule_hint_mode_numa, domain));
            }
            else {
                p.apply(
                    hpx::launch::async,
                    executor_.get_priority(),
                    executor_.get_stacksize(),
                    threads::thread_schedule_hint(
                                thread_schedule_hint_mode_numa, domain));
            }

            return p.get_future();
        }
    };

    // --------------------------------------------------------------------
    template <typename H>
    struct HPX_EXPORT guided_pool_executor {};

    // --------------------------------------------------------------------
    // this is a guided pool executor templated over args only
    // the args should be the same as those that would be called
    // for an async function or continuation. This makes it possible to
    // guide a lambda rather than a full function.
    template <typename Tag>
    struct HPX_EXPORT guided_pool_executor<pool_numa_hint<Tag>>
    {
      public:
        guided_pool_executor(const std::string& pool_name,
                             bool hp_sync = false)
            : pool_executor_(pool_name)
            , hp_sync_(hp_sync)
        {}

        guided_pool_executor(const std::string& pool_name,
                             thread_stacksize stacksize,
                             bool hp_sync = false)
            : pool_executor_(pool_name, stacksize)
            , hp_sync_(hp_sync)
        {}

        guided_pool_executor(const std::string& pool_name, thread_priority priority,
                             thread_stacksize stacksize = thread_stacksize_default,
                             bool hp_sync = false)
            : pool_executor_(pool_name, priority, stacksize)
            , hp_sync_(hp_sync)
        {}

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
                      << util::debug::print_type<F>() << "\n"
                      << "async_execute : Arguments   : "
                      << util::debug::print_type<Ts...>(" | ") << "\n"
                      << "async_execute : Result      : "
                      << util::debug::print_type<result_type>() << "\n"
                      << "async_execute : Numa Hint   : "
                      << util::debug::print_type<pool_numa_hint<Tag>>() << "\n";
#endif

            // hold onto the function until all futures have become ready
            // by using a dataflow operation, then call the scheduling hint
            // before passing the task onwards to the real executor
            return dataflow(launch::sync,
                util::unwrapping(
                    pre_execution_async_domain_schedule<pool_executor,
                        pool_numa_hint<Tag>> {
                            pool_executor_, hint_, hp_sync_
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
                  typename = enable_if_t<traits::is_future<Future>::value>
                  >
        auto
        then_execute(F && f, Future && predecessor, Ts &&... ts)
        ->  future<typename util::detail::invoke_deferred_result<
            F, Future, Ts...>::type>
        {
#ifdef GUIDED_EXECUTOR_DEBUG
            typedef typename util::detail::invoke_deferred_result<
                    F, Future, Ts...>::type result_type;

            std::cout << "then_execute : Function     : "
                      << util::debug::print_type<F>() << "\n"
                      << "then_execute : Predecessor  : "
                      << util::debug::print_type<Future>() << "\n"
                      << "then_execute : Future       : "
                      << util::debug::print_type<typename
                         traits::future_traits<Future>::result_type>() << "\n"
                      << "then_execute : Arguments    : "
                      << util::debug::print_type<Ts...>(" | ") << "\n"
                      << "then_execute : Result       : "
                      << util::debug::print_type<result_type>() << "\n";
#endif

            // Note 1 : The Ts &&... args are not actually used in a continuation since
            // only the future becoming ready (predecessor) is actually passed onwards.

            // Note 2 : we do not need to use unwrapping here, because dataflow
            // continuations are only invoked once the futures are already ready

            // Note 3 : launch::sync is used here to make wrapped task run on
            // the thread of the predecessor continuation coming ready.
            // the numa_hint_function will be evaluated on that thread and then
            // the real task will be spawned on a new task with hints - as intended
            return dataflow(launch::sync,
                [f{std::move(f)}, this](Future && predecessor, Ts &&... ts)
                {
                    pre_execution_then_domain_schedule<
                        pool_executor, pool_numa_hint<Tag>>
                            pre_exec { pool_executor_, hint_, hp_sync_};

                    return pre_exec(
                        std::move(f),
                        std::forward<Future>(predecessor));
                },
                std::forward<Future>(predecessor),
                std::forward<Ts>(ts)...
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
                  typename = enable_if_t<is_future_of_tuple_of_futures<
                    OuterFuture<util::tuple<InnerFutures...>>>::value>,
                  typename = enable_if_t<is_tuple_of_futures<
                    util::tuple<InnerFutures...>>::value>
                  >
        auto
        then_execute(F && f,
                     OuterFuture<util::tuple<InnerFutures... > > && predecessor,
                     Ts &&... ts)
        ->  future<typename util::detail::invoke_deferred_result<
            F, OuterFuture<util::tuple<InnerFutures... >>, Ts...>::type>
        {
#ifdef GUIDED_EXECUTOR_DEBUG
            // get the tuple of futures from the predecessor future <tuple of futures>
            const auto & predecessor_value = future_extract_value()(predecessor);

            // create a tuple of the unwrapped future values
            auto unwrapped_futures_tuple = util::map_pack(
                future_extract_value{},
                predecessor_value
            );

            typedef typename util::detail::invoke_deferred_result<
                F, OuterFuture<util::tuple<InnerFutures... >>, Ts...>::type
                    result_type;

            std::cout << "when_all(fut) : Predecessor : "
                      << util::debug::print_type<
                         OuterFuture<util::tuple<InnerFutures...>>>()
                      << "\n"
                      << "when_all(fut) : unwrapped   : "
                      << util::debug::print_type<
                         decltype(unwrapped_futures_tuple)>(" | ")
                      << "\n"
                      << "then_execute  : Arguments   : "
                      << util::debug::print_type<Ts...>(" | ") << "\n"
                      << "when_all(fut) : Result      : "
                      << util::debug::print_type<result_type>() << "\n";
#endif

            // Please see notes for previous then_execute function above
            return dataflow(launch::sync,
                [f{std::move(f)}, this]
                (OuterFuture<util::tuple<InnerFutures...>> && predecessor, Ts &&... ts)
                {
                    pre_execution_then_domain_schedule<pool_executor,
                        pool_numa_hint<Tag>>
                        pre_exec { pool_executor_, hint_, hp_sync_ };

                    return pre_exec(
                        std::move(f),
                        std::forward<OuterFuture<util::tuple<InnerFutures...>>>
                        (predecessor));
                },
                std::forward<OuterFuture<util::tuple<InnerFutures...>>>(predecessor),
                std::forward<Ts>(ts)...
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
                  typename = enable_if_t<
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

            // invoke the hint function with the unwrapped tuple futures
#ifdef GUIDED_POOL_EXECUTOR_FAKE_NOOP
            int domain = -1;
#else
            auto unwrapped_futures_tuple = util::map_pack(
                future_extract_value{},
                predecessor
            );

            int domain = util::invoke_fused(hint_, unwrapped_futures_tuple);
#endif

#ifdef GUIDED_EXECUTOR_DEBUG
            std::cout << "dataflow      : Predecessor : "
                      << util::debug::print_type<util::tuple<InnerFutures...>>()
                      << "\n"
                      << "dataflow      : unwrapped   : "
                      << util::debug::print_type<
                         decltype(unwrapped_futures_tuple)>(" | ")
                      << "\n"
                      << "dataflow-frame: Result      : "
                      << util::debug::print_type<Result>() << "\n";

            std::cout << "dataflow hint returning " << domain << "\n";
#endif

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

            if (hp_sync_ &&
                    pool_executor_.get_priority() == hpx::threads::thread_priority_high) {
                p.apply(
                    hpx::launch::sync,
                    pool_executor_.get_priority(),
                    pool_executor_.get_stacksize(),
                    threads::thread_schedule_hint(
                                thread_schedule_hint_mode_numa, domain));
            }
            else {
                p.apply(
                    hpx::launch::async,
                    pool_executor_.get_priority(),
                    pool_executor_.get_stacksize(),
                    threads::thread_schedule_hint(
                                thread_schedule_hint_mode_numa, domain));
            }
            return p.get_future();
        }

    private:
        pool_executor       pool_executor_;
        pool_numa_hint<Tag> hint_;
        bool                hp_sync_;
    };

    // --------------------------------------------------------------------
    // guided_pool_executor_shim
    // an executor compatible with scheduled executor API
    // --------------------------------------------------------------------
    template <typename H>
    struct HPX_EXPORT guided_pool_executor_shim {
    public:
        guided_pool_executor_shim(bool guided, const std::string& pool_name,
                                  bool hp_sync = false)
            : guided_(guided)
            , guided_exec_(pool_name, hp_sync)
            , pool_exec_(pool_name)
        {}

        guided_pool_executor_shim(bool guided, const std::string& pool_name,
                                  thread_stacksize stacksize,
                                  bool hp_sync = false)
            : guided_(guided)
            , guided_exec_(pool_name, hp_sync, stacksize)
            , pool_exec_(pool_name, stacksize)
        {}

        guided_pool_executor_shim(bool guided, const std::string& pool_name,
                                  thread_priority priority,
                                  thread_stacksize stacksize = thread_stacksize_default,
                                  bool hp_sync = false)
            : guided_(guided)
            , guided_exec_(pool_name, priority, stacksize, hp_sync)
            , pool_exec_(pool_name, priority, stacksize)
        {}

        // --------------------------------------------------------------------
        // async
        // --------------------------------------------------------------------
        template <typename F, typename ... Ts>
        future<typename util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F && f, Ts &&... ts)
        {
            if (guided_) return guided_exec_.async_execute(
                std::forward<F>(f), std::forward<Ts>(ts)...);
            else {
                typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
                    result_type;

                lcos::local::futures_factory<result_type()> p(
                    pool_exec_,
                    util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...)
                );
                p.apply(
                    hpx::launch::async,
                    pool_exec_.get_priority(),
                    pool_exec_.get_stacksize(),
                    threads::thread_schedule_hint()
                );
                return p.get_future();
            }
        }

        // --------------------------------------------------------------------
        // Continuation
        // --------------------------------------------------------------------
        template <typename F,
                  typename Future,
                  typename ... Ts,
                  typename = enable_if_t<traits::is_future<Future>::value>
                  >
        auto
        then_execute(F && f, Future && predecessor, Ts &&... ts)
        ->  future<typename util::detail::invoke_deferred_result<
            F, Future, Ts...>::type>
        {
            if (guided_) return guided_exec_.then_execute(
                std::forward<F>(f), std::forward<Future>(predecessor),
                std::forward<Ts>(ts)...);
            else {
                typedef typename hpx::util::detail::invoke_deferred_result<
                        F, Future, Ts...
                    >::type result_type;

                auto func = hpx::util::bind(
                    hpx::util::one_shot(std::forward<F>(f)),
                    hpx::util::placeholders::_1, std::forward<Ts>(ts)...);

                typename hpx::traits::detail::shared_state_ptr<result_type>::type p =
                    hpx::lcos::detail::make_continuation_exec<result_type>(
                        std::forward<Future>(predecessor), pool_exec_,
                        std::move(func));

                return hpx::traits::future_access<hpx::lcos::future<result_type>>::
                    create(std::move(p));
            }
        }

        // --------------------------------------------------------------------

        bool                    guided_;
        guided_pool_executor<H> guided_exec_;
        pool_executor           pool_exec_;
    };


}}}

namespace hpx { namespace parallel { namespace execution
{
    template <typename Hint>
    struct executor_execution_category<
        threads::executors::guided_pool_executor<Hint> >
    {
        typedef parallel::execution::parallel_execution_tag type;
    };

    template <typename Hint>
    struct is_two_way_executor<
            threads::executors::guided_pool_executor<Hint> >
      : std::true_type
    {};

    // ----------------------------
    template <typename Hint>
    struct executor_execution_category<
        threads::executors::guided_pool_executor_shim<Hint> >
    {
        typedef parallel::execution::parallel_execution_tag type;
    };

    template <typename Hint>
    struct is_two_way_executor<
            threads::executors::guided_pool_executor_shim<Hint> >
      : std::true_type
    {};

}}}

#include <hpx/config/warnings_suffix.hpp>

#endif /*HPX_RUNTIME_THREADS_GUIDED_POOL_EXECUTOR*/
