//  Copyright (c) 2017-2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/assert.hpp>
#include <hpx/debugging/demangle_helper.hpp>
#include <hpx/debugging/print.hpp>
#include <hpx/executors/dataflow.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/futures/traits/is_future_tuple.hpp>
#include <hpx/threading_base/thread_description.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

//#define GUIDED_POOL_EXECUTOR_FAKE_NOOP

#include <hpx/config/warnings_prefix.hpp>

#if !defined(GUIDED_POOL_EXECUTOR_DEBUG)
#define GUIDED_POOL_EXECUTOR_DEBUG false
#endif

namespace hpx {
    // cppcheck-suppress ConfigurationNotChecked
    static hpx::debug::enable_print<GUIDED_POOL_EXECUTOR_DEBUG> gpx_deb(
        "GP_EXEC");
}    // namespace hpx

// --------------------------------------------------------------------
// pool_numa_hint
// --------------------------------------------------------------------
namespace hpx { namespace parallel { namespace execution {
    namespace detail {
        // --------------------------------------------------------------------
        // helper struct for tuple of futures future<tuple<f1, f2, f3, ...>>>
        // --------------------------------------------------------------------
        template <typename Future>
        struct is_future_of_tuple_of_futures
          : std::integral_constant<bool,
                hpx::traits::is_future<Future>::value &&
                    hpx::traits::is_future_tuple<typename hpx::traits::
                            future_traits<Future>::result_type>::value>
        {
        };

        // --------------------------------------------------------------------
        // function that returns a const ref to the contents of a future
        // without calling .get() on the future so that we can use the value
        // and then pass the original future on to the intended destination.
        // --------------------------------------------------------------------
        struct future_extract_value
        {
            template <typename T, template <typename> class Future>
            const T& operator()(const Future<T>& el) const
            {
                const auto& state = hpx::traits::detail::get_shared_state(el);
                return *state->get_result();
            }
        };

        // --------------------------------------------------------------------
        // helper to get result from a ready future without invalidating it
        // --------------------------------------------------------------------
        template <typename T>
        T const& peek_future_result(T const& t)
        {
            return t;
        }

        template <typename T,
            typename Enable = std::enable_if_t<!std::is_void<T>::value>>
        T const& peek_future_result(hpx::future<T> const& f)
        {
            HPX_ASSERT(f.is_ready());
            auto shared_state =
                hpx::traits::future_access<hpx::future<T>>::get_shared_state(f);
            return *shared_state->get_result();
        }

        template <typename T,
            typename Enable = std::enable_if_t<!std::is_void<T>::value>>
        T const& peek_future_result(hpx::shared_future<T> const& f)
        {
            HPX_ASSERT(f.is_ready());
            return f.get();
        }

        // --------------------------------------------------------------------
        // helper : numa domain scheduling for async() execution
        // --------------------------------------------------------------------
        template <typename Executor, typename NumaFunction>
        struct pre_execution_async_domain_schedule
        {
            Executor& executor_;
            NumaFunction& numa_function_;
            bool hp_sync_;
            //
            template <typename F, typename... Ts>
            auto operator()(F&& f, Ts&&... ts) const
            {
                // call the numa hint function
#ifdef GUIDED_POOL_EXECUTOR_FAKE_NOOP
                int domain = -1;
#else
                int domain = numa_function_(peek_future_result(ts)...);
#endif

                gpx_deb.debug(
                    debug::str<>("async_schedule"), "domain ", domain);

                // now we must forward the task+hint on to the correct dispatch function
                typedef typename hpx::util::detail::invoke_deferred_result<F,
                    Ts...>::type result_type;

                lcos::local::futures_factory<result_type()> p(
                    hpx::util::deferred_call(
                        HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...));

                gpx_deb.debug(
                    debug::str<>("triggering apply"), "domain ", domain);
                if (hp_sync_ &&
                    executor_.priority_ == hpx::threads::thread_priority::high)
                {
                    p.apply(executor_.pool_, "guided sync",
                        hpx::launch::sync_policy(
                            hpx::threads::thread_priority::high,
                            executor_.stacksize_,
                            hpx::threads::thread_schedule_hint(
                                hpx::threads::thread_schedule_hint_mode::numa,
                                domain)));
                }
                else
                {
                    p.apply(executor_.pool_, "guided async",
                        hpx::launch::async_policy(executor_.priority_,
                            executor_.stacksize_,
                            hpx::threads::thread_schedule_hint(
                                hpx::threads::thread_schedule_hint_mode::numa,
                                domain)));
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
            Executor& executor_;
            NumaFunction& numa_function_;
            bool hp_sync_;
            //
            template <typename F, typename Future, typename... Ts>
            auto operator()(F&& f, Future&& predecessor, Ts&&... ts) const
            {
                // call the numa hint function
#ifdef GUIDED_POOL_EXECUTOR_FAKE_NOOP
                int domain = -1;
#else
                // get the argument for the numa hint function from the predecessor future
                const auto& predecessor_value =
                    detail::future_extract_value()(predecessor);
                int domain = numa_function_(predecessor_value, ts...);
#endif

                gpx_deb.debug(debug::str<>("then_schedule"), "domain ", domain);

                // now we must forward the task+hint on to the correct dispatch function
                typedef typename hpx::util::detail::invoke_deferred_result<F,
                    Future, Ts...>::type result_type;

                lcos::local::futures_factory<result_type()> p(
                    hpx::util::deferred_call(HPX_FORWARD(F, f),
                        HPX_FORWARD(Future, predecessor),
                        HPX_FORWARD(Ts, ts)...));

                if (hp_sync_ &&
                    executor_.priority_ == hpx::threads::thread_priority::high)
                {
                    p.apply(executor_.pool_, "guided then",
                        hpx::launch::sync_policy(
                            hpx::threads::thread_priority::high,
                            executor_.stacksize_,
                            hpx::threads::thread_schedule_hint(
                                hpx::threads::thread_schedule_hint_mode::numa,
                                domain)));
                }
                else
                {
                    p.apply(executor_.pool_, "guided then",
                        hpx::launch::async_policy(executor_.priority_,
                            executor_.stacksize_,
                            hpx::threads::thread_schedule_hint(
                                hpx::threads::thread_schedule_hint_mode::numa,
                                domain)));
                }

                return p.get_future();
            }
        };
    }    // namespace detail

    // --------------------------------------------------------------------
    // Template type for a numa domain scheduling hint
    template <typename... Args>
    struct pool_numa_hint
    {
    };

    // Template type for a core scheduling hint
    template <typename... Args>
    struct pool_core_hint
    {
    };

    // --------------------------------------------------------------------
    template <typename H>
    struct guided_pool_executor;

    template <typename H>
    struct guided_pool_executor_shim;

    // --------------------------------------------------------------------
    // this is a guided pool executor templated over args only
    // the args should be the same as those that would be called
    // for an async function or continuation. This makes it possible to
    // guide a lambda rather than a full function.
    template <typename Tag>
    struct guided_pool_executor<pool_numa_hint<Tag>>
    {
        template <typename Executor, typename NumaFunction>
        friend struct detail::pre_execution_async_domain_schedule;

        template <typename Executor, typename NumaFunction>
        friend struct detail::pre_execution_then_domain_schedule;

        template <typename H>
        friend struct guided_pool_executor_shim;

    public:
        guided_pool_executor(
            threads::thread_pool_base* pool, bool hp_sync = false)
          : pool_(pool)
          , priority_(threads::thread_priority::default_)
          , stacksize_(threads::thread_stacksize::default_)
          , hp_sync_(hp_sync)
        {
        }

        guided_pool_executor(threads::thread_pool_base* pool,
            threads::thread_stacksize stacksize, bool hp_sync = false)
          : pool_(pool)
          , priority_(threads::thread_priority::default_)
          , stacksize_(stacksize)
          , hp_sync_(hp_sync)
        {
        }

        guided_pool_executor(threads::thread_pool_base* pool,
            threads::thread_priority priority,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize::default_,
            bool hp_sync = false)
          : pool_(pool)
          , priority_(priority)
          , stacksize_(stacksize)
          , hp_sync_(hp_sync)
        {
        }

        // --------------------------------------------------------------------
        // async execute specialized for simple arguments typical
        // of a normal async call with arbitrary arguments
        // --------------------------------------------------------------------
        template <typename F, typename... Ts>
        future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F&& f, Ts&&... ts)
        {
            typedef typename hpx::util::detail::invoke_deferred_result<F,
                Ts...>::type result_type;

            gpx_deb.debug(debug::str<>("async execute"), "\n\t",
                "Function    : ", hpx::util::debug::print_type<F>(), "\n\t",
                "Arguments   : ", hpx::util::debug::print_type<Ts...>(" | "),
                "\n\t",
                "Result      : ", hpx::util::debug::print_type<result_type>(),
                "\n\t", "Numa Hint   : ",
                hpx::util::debug::print_type<pool_numa_hint<Tag>>());

            // hold onto the function until all futures have become ready
            // by using a dataflow operation, then call the scheduling hint
            // before passing the task onwards to the real executor
            return dataflow(launch::sync,
                detail::pre_execution_async_domain_schedule<
                    typename std::decay<typename std::remove_pointer<decltype(
                        this)>::type>::type,
                    pool_numa_hint<Tag>>{*this, hint_, hp_sync_},
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        // --------------------------------------------------------------------
        // .then() execute specialized for a future<P> predecessor argument
        // note that future<> and shared_future<> are both supported
        // --------------------------------------------------------------------
        template <typename F, typename Future, typename... Ts,
            typename = std::enable_if_t<hpx::traits::is_future<Future>::value>>
        auto then_execute(F&& f, Future&& predecessor, Ts&&... ts)
            -> future<typename hpx::util::detail::invoke_deferred_result<F,
                Future, Ts...>::type>
        {
            typedef typename hpx::util::detail::invoke_deferred_result<F,
                Future, Ts...>::type result_type;

            gpx_deb.debug(debug::str<>("then execute"), "\n\t",
                "Function    : ", hpx::util::debug::print_type<F>(), "\n\t",
                "Predecessor  : ", hpx::util::debug::print_type<Future>(),
                "\n\t", "Future       : ",
                hpx::util::debug::print_type<
                    typename hpx::traits::future_traits<Future>::result_type>(),
                "\n\t",
                "Arguments   : ", hpx::util::debug::print_type<Ts...>(" | "),
                "\n\t",
                "Result      : ", hpx::util::debug::print_type<result_type>(),
                "\n\t", "Numa Hint   : ",
                hpx::util::debug::print_type<pool_numa_hint<Tag>>());

            // Note 1 : The Ts &&... args are not actually used in a continuation since
            // only the future becoming ready (predecessor) is actually passed onwards.

            // Note 2 : we do not need to use unwrapping here, because dataflow
            // continuations are only invoked once the futures are already ready

            // Note 3 : launch::sync is used here to make wrapped task run on
            // the thread of the predecessor continuation coming ready.
            // the numa_hint_function will be evaluated on that thread and then
            // the real task will be spawned on a new task with hints - as intended
            return dataflow(
                launch::sync,
                [f = HPX_FORWARD(F, f), this](
                    Future&& predecessor, Ts&&... /* ts */) mutable {
                    detail::pre_execution_then_domain_schedule<
                        guided_pool_executor, pool_numa_hint<Tag>>
                        pre_exec{*this, hint_, hp_sync_};

                    return pre_exec(
                        HPX_MOVE(f), HPX_FORWARD(Future, predecessor));
                },
                HPX_FORWARD(Future, predecessor), HPX_FORWARD(Ts, ts)...);
        }

        // --------------------------------------------------------------------
        // .then() execute specialized for a when_all dispatch for any future types
        // future< tuple< is_future<a>::type, is_future<b>::type, ...> >
        // --------------------------------------------------------------------
        template <typename F, template <typename> class OuterFuture,
            typename... InnerFutures, typename... Ts,
            typename = std::enable_if_t<detail::is_future_of_tuple_of_futures<
                OuterFuture<hpx::tuple<InnerFutures...>>>::value>,
            typename = std::enable_if_t<hpx::traits::is_future_tuple<
                hpx::tuple<InnerFutures...>>::value>>
        auto then_execute(F&& f,
            OuterFuture<hpx::tuple<InnerFutures...>>&& predecessor, Ts&&... ts)
            -> future<typename hpx::util::detail::invoke_deferred_result<F,
                OuterFuture<hpx::tuple<InnerFutures...>>, Ts...>::type>
        {
#ifdef GUIDED_EXECUTOR_DEBUG
            // get the tuple of futures from the predecessor future <tuple of futures>
            const auto& predecessor_value =
                detail::future_extract_value()(predecessor);

            // create a tuple of the unwrapped future values
            auto unwrapped_futures_tuple = hpx::util::map_pack(
                detail::future_extract_value{}, predecessor_value);

            typedef typename hpx::util::detail::invoke_deferred_result<F,
                OuterFuture<hpx::tuple<InnerFutures...>>, Ts...>::type
                result_type;

            // clang-format off
            gpx_deb.debug(debug::str<>("when_all(fut) : Predecessor")
                , hpx::util::debug::print_type<
                       OuterFuture<hpx::tuple<InnerFutures...>>>()
                , "\n"
                , "when_all(fut) : unwrapped   : "
                , hpx::util::debug::print_type<decltype(unwrapped_futures_tuple)>(
                       " | ")
                , "\n"
                , "then_execute  : Arguments   : "
                , hpx::util::debug::print_type<Ts...>(" | ") , "\n"
                , "when_all(fut) : Result      : "
                , hpx::util::debug::print_type<result_type>() , "\n"
            );
            // clang-format on
#endif

            // Please see notes for previous then_execute function above
            return dataflow(
                launch::sync,
                [f = HPX_FORWARD(F, f), this](
                    OuterFuture<hpx::tuple<InnerFutures...>>&& predecessor,
                    Ts&&... /* ts */) mutable {
                    detail::pre_execution_then_domain_schedule<
                        guided_pool_executor, pool_numa_hint<Tag>>
                        pre_exec{*this, hint_, hp_sync_};

                    return pre_exec(HPX_MOVE(f),
                        std::forward<OuterFuture<hpx::tuple<InnerFutures...>>>(
                            predecessor));
                },
                std::forward<OuterFuture<hpx::tuple<InnerFutures...>>>(
                    predecessor),
                HPX_FORWARD(Ts, ts)...);
        }

        // --------------------------------------------------------------------
        // execute specialized for a dataflow dispatch
        // dataflow unwraps the outer future for us but passes a dataflowframe
        // function type, result type and tuple of futures as arguments
        // --------------------------------------------------------------------
        template <typename F, typename... InnerFutures,
            typename = std::enable_if_t<hpx::traits::is_future_tuple<
                hpx::tuple<InnerFutures...>>::value>>
        auto async_execute(F&& f, hpx::tuple<InnerFutures...>&& predecessor)
            -> future<typename hpx::util::detail::invoke_deferred_result<F,
                hpx::tuple<InnerFutures...>>::type>
        {
            typedef typename hpx::util::detail::invoke_deferred_result<F,
                hpx::tuple<InnerFutures...>>::type result_type;

            // invoke the hint function with the unwrapped tuple futures
#ifdef GUIDED_POOL_EXECUTOR_FAKE_NOOP
            int domain = -1;
#else
            auto unwrapped_futures_tuple = hpx::util::map_pack(
                detail::future_extract_value{}, predecessor);

            int domain =
                hpx::util::invoke_fused(hint_, unwrapped_futures_tuple);
#endif

#ifndef GUIDED_EXECUTOR_DEBUG
            // clang-format off
            gpx_deb.debug(debug::str<>("dataflow      : Predecessor")
                      , hpx::util::debug::print_type<hpx::tuple<InnerFutures...>>()
                      , "\n"
                      , "dataflow      : unwrapped   : "
                      , hpx::util::debug::print_type<
#ifdef GUIDED_POOL_EXECUTOR_FAKE_NOOP
                             int>(" | ")
#else
                             decltype(unwrapped_futures_tuple)>(" | ")
#endif
                      , "\n");

            gpx_deb.debug(debug::str<>("dataflow hint"), debug::dec<>(domain));
            // clang-format on
#endif

            // forward the task execution on to the real internal executor
            lcos::local::futures_factory<result_type()> p(
                hpx::util::deferred_call(HPX_FORWARD(F, f),
                    std::forward<hpx::tuple<InnerFutures...>>(predecessor)));

            if (hp_sync_ && priority_ == hpx::threads::thread_priority::high)
            {
                p.apply(pool_, "guided async",
                    hpx::launch::sync_policy(
                        hpx::threads::thread_priority::high, stacksize_,
                        hpx::threads::thread_schedule_hint(
                            hpx::threads::thread_schedule_hint_mode::numa,
                            domain)));
            }
            else
            {
                p.apply(pool_, "guided async",
                    hpx::launch::async_policy(priority_, stacksize_,
                        hpx::threads::thread_schedule_hint(
                            hpx::threads::thread_schedule_hint_mode::numa,
                            domain)));
            }
            return p.get_future();
        }

    private:
        threads::thread_pool_base* pool_;
        threads::thread_priority priority_;
        threads::thread_stacksize stacksize_;
        pool_numa_hint<Tag> hint_;
        bool hp_sync_;
    };

    // --------------------------------------------------------------------
    // guided_pool_executor_shim
    // an executor compatible with scheduled executor API
    // --------------------------------------------------------------------
    template <typename H>
    struct guided_pool_executor_shim
    {
    public:
        guided_pool_executor_shim(
            bool guided, threads::thread_pool_base* pool, bool hp_sync = false)
          : guided_(guided)
          , guided_exec_(pool, hp_sync)
        {
        }

        guided_pool_executor_shim(bool guided, threads::thread_pool_base* pool,
            threads::thread_stacksize stacksize, bool hp_sync = false)
          : guided_(guided)
          , guided_exec_(pool, hp_sync, stacksize)
        {
        }

        guided_pool_executor_shim(bool guided, threads::thread_pool_base* pool,
            threads::thread_priority priority,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize::default_,
            bool hp_sync = false)
          : guided_(guided)
          , guided_exec_(pool, priority, stacksize, hp_sync)
        {
        }

        // --------------------------------------------------------------------
        // async
        // --------------------------------------------------------------------
        template <typename F, typename... Ts>
        future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F&& f, Ts&&... ts)
        {
            if (guided_)
                return guided_exec_.async_execute(
                    HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            else
            {
                typedef typename hpx::util::detail::invoke_deferred_result<F,
                    Ts...>::type result_type;

                lcos::local::futures_factory<result_type()> p(
                    hpx::util::deferred_call(
                        HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...));

                p.apply(guided_exec_.pool_, "guided async",
                    hpx::launch::async_policy(
                        guided_exec_.priority_, guided_exec_.stacksize_));
                return p.get_future();
            }
        }

        // --------------------------------------------------------------------
        // Continuation
        // --------------------------------------------------------------------
        template <typename F, typename Future, typename... Ts,
            typename = std::enable_if_t<hpx::traits::is_future<Future>::value>>
        auto then_execute(F&& f, Future&& predecessor, Ts&&... ts)
            -> future<typename hpx::util::detail::invoke_deferred_result<F,
                Future, Ts...>::type>
        {
            if (guided_)
                return guided_exec_.then_execute(HPX_FORWARD(F, f),
                    HPX_FORWARD(Future, predecessor), HPX_FORWARD(Ts, ts)...);
            else
            {
                typedef typename hpx::util::detail::invoke_deferred_result<F,
                    Future, Ts...>::type result_type;

                auto func = hpx::util::one_shot(
                    hpx::bind_back(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...));

                typename hpx::traits::detail::shared_state_ptr<
                    result_type>::type p =
                    hpx::lcos::detail::make_continuation_exec<result_type>(
                        HPX_FORWARD(Future, predecessor), *this,
                        HPX_MOVE(func));

                return hpx::traits::future_access<
                    hpx::future<result_type>>::create(HPX_MOVE(p));
            }
        }

        // --------------------------------------------------------------------

        bool guided_;
        guided_pool_executor<H> guided_exec_;
    };

    template <typename Hint>
    struct executor_execution_category<guided_pool_executor<Hint>>
    {
        typedef hpx::execution::parallel_execution_tag type;
    };

    template <typename Hint>
    struct is_two_way_executor<guided_pool_executor<Hint>> : std::true_type
    {
    };

    // ----------------------------
    template <typename Hint>
    struct executor_execution_category<guided_pool_executor_shim<Hint>>
    {
        typedef hpx::execution::parallel_execution_tag type;
    };

    template <typename Hint>
    struct is_two_way_executor<guided_pool_executor_shim<Hint>> : std::true_type
    {
    };

}}}    // namespace hpx::parallel::execution

#include <hpx/config/warnings_suffix.hpp>
