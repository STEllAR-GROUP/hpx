//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_EXECUTORS_THREAD_TIMED_EXECUTOR_JAN_09_2017_1117AM)
#define HPX_PARALLEL_EXECUTORS_THREAD_TIMED_EXECUTOR_JAN_09_2017_1117AM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/steady_clock.hpp>
#include <hpx/traits/executor_traits.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/unwrapped.hpp>

#include <hpx/parallel/executors/execution.hpp>

#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { namespace execution
{
    template <typename ThreadExecutor>
    struct thread_timed_executor
    {
    private:
        typedef typename std::decay<ThreadExecutor>::type base_type;
        static_assert(
            hpx::traits::is_threads_executor<base_type>::value,
            "ThreadExecutor must be a type exposing the threads::executor interface");

        base_type base_;

        template <typename ... Ts>
        struct is_self
          : std::true_type
        {};

        template <typename T>
        struct is_self<T>
          : std::is_same<typename std::decay<T>::type, thread_timed_executor>
        {};

    public:
        // generic forwarding constructor
        template <typename ... Ts, typename Enable =
                typename std::enable_if<
                    (sizeof...(Ts) != 1) || !is_self<Ts...>::value
                >::type>
        thread_timed_executor(Ts &&... ts)
          : base_(std::forward<Ts>(ts)...)
        {}

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename ... Ts>
        HPX_FORCEINLINE
        hpx::lcos::future<
            typename hpx::util::detail::deferred_result_of<F(Ts...)>::type
        >
        async_execute(F && f, Ts &&... ts)
        {
            typedef typename hpx::util::detail::deferred_result_of<
                    F(Ts...)
                >::type result_type;

            lcos::local::packaged_task<
                    result_type(typename std::decay<Ts>::type...)
                > task(std::forward<F>(f));

            hpx::future<result_type> result = task.get_future();

            base_.add(
                hpx::util::deferred_call(
                    std::move(task), std::forward<Ts>(ts)...),
                "async_execute");

            return result;
        }

        template <typename F, typename ... Ts>
        HPX_FORCEINLINE
        typename hpx::util::detail::deferred_result_of<F(Ts...)>::type
        sync_execute(F && f, Ts &&... ts)
        {
            return async_execute(std::forward<F>(f), std::forward<Ts>(ts)...).get();
        }

        template <typename F, typename Future, typename ... Ts>
        HPX_FORCEINLINE
        hpx::lcos::future<
            typename hpx::util::detail::deferred_result_of<
                F(Future, Ts...)
            >::type
        >
        then_execute(F && f, Future& predecessor, Ts &&... ts)
        {
            typedef typename hpx::util::detail::deferred_result_of<
                    F(Future, Ts...)
                >::type result_type;

            auto func = hpx::util::bind(
                hpx::util::one_shot(std::forward<F>(f)),
                hpx::util::placeholders::_1, std::forward<Ts>(ts)...);

            typename hpx::traits::detail::shared_state_ptr<result_type>::type
                p = hpx::lcos::detail::make_continuation_thread_exec<result_type>(
                        predecessor, base_, std::move(func));

            return hpx::traits::future_access<hpx::lcos::future<result_type> >::
                create(std::move(p));
        }

        template <typename F, typename ... Ts>
        HPX_FORCEINLINE
        void apply_execute(F && f, Ts &&... ts)
        {
            base_.add(
                hpx::util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(ts)...),
                "post");
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Shape, typename ... Ts>
        std::vector<hpx::lcos::future<
            typename parallel::execution::detail::bulk_function_result<
                F, Shape, Ts...
            >::type
        > >
        async_bulk_execute(F && f, Shape const& shape, Ts &&... ts)
        {
            std::vector<hpx::future<
                    typename parallel::execution::detail::bulk_function_result<
                        F, Shape, Ts...
                    >::type
                > > results;

// Before Boost V1.56 boost::size() does not respect the iterator category of
// its argument.
#if BOOST_VERSION < 105600
            results.reserve(std::distance(boost::begin(shape), boost::end(shape)));
#else
            results.reserve(boost::size(shape));
#endif

            for (auto const& elem: shape)
            {
                results.push_back(async_execute(std::forward<F>(f),
                    elem, ts...));
            }

            return results;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Shape, typename ... Ts>
        typename parallel::execution::detail::bulk_execute_result<
            F, Shape, Ts...
        >::type
        sync_bulk_execute(F && f, Shape const& shape, Ts &&... ts)
        {
            return hpx::util::unwrapped(async_bulk_execute(
                std::forward<F>(f), shape, std::forward<Ts>(ts)...));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Shape, typename Future, typename ... Ts>
        HPX_FORCEINLINE
        hpx::future<
            typename parallel::execution::detail::then_bulk_execute_result<
                F, Shape, Future, Ts...
            >::type
        >
        then_bulk_execute(F && f, Shape const& shape, Future& predecessor,
            Ts &&... ts)
        {
            typedef typename parallel::execution::detail::then_bulk_function_result<
                    F, Shape, Future, Ts...
                >::type func_result_type;

            typedef std::vector<hpx::lcos::future<func_result_type> > result_type;
            typedef hpx::lcos::future<result_type> result_future_type;

            // older versions of gcc are not able to capture parameter
            // packs (gcc < 4.9)
            auto args = hpx::util::make_tuple(std::forward<Ts>(ts)...);
            auto& this_ = *this;
            auto func =
                [&this_, f, shape, args](Future predecessor) mutable
                ->  result_type
                {
                    return parallel::execution::detail::fused_async_bulk_execute(
                        this_, f, shape, predecessor,
                        typename hpx::util::detail::make_index_pack<
                            sizeof...(Ts)
                        >::type(), args);
                };

            typedef typename hpx::traits::detail::shared_state_ptr<
                    result_type
                >::type shared_state_type;

            shared_state_type p =
                lcos::detail::make_continuation_thread_exec<result_type>(
                    predecessor, base_, std::move(func));

            return hpx::traits::future_access<result_future_type>::
                create(std::move(p));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename ... Ts>
        void apply_execute_at(hpx::util::steady_time_point const& abs_time,
            F && f, Ts &&... ts)
        {
            base_.add_at(abs_time,
                hpx::util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(ts)...),
                "post_at");
        }

        template <typename F, typename ... Ts>
        void apply_execute_after(hpx::util::steady_duration const& rel_time,
            F && f, Ts &&... ts)
        {
            base_.add_at(rel_time,
                hpx::util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(ts)...),
                "post_at");
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename ... Ts>
        hpx::future<
            typename hpx::util::detail::deferred_result_of<F(Ts...)>::type
        >
        async_execute_at(std::chrono::steady_clock::time_point const& abs_time,
            F && f, Ts &&... ts)
        {
            typedef typename hpx::util::detail::deferred_result_of<
                    F(Ts...)
                >::type result_type;

            lcos::local::packaged_task<
                    result_type(typename std::decay<Ts>::type...)
                > task(std::forward<F>(f));

            hpx::future<result_type> result = task.get_future();

            base_.add_at(abs_time,
                hpx::util::deferred_call(
                    std::move(task), std::forward<Ts>(ts)...),
                "async_execute_at");

            return result;
        }

        template <typename F, typename ... Ts>
        hpx::future<
            typename hpx::util::detail::deferred_result_of<F(Ts...)>::type
        >
        async_execute_after(std::chrono::steady_clock::duration const& rel_time,
            F && f, Ts &&... ts)
        {
            typedef typename hpx::util::detail::deferred_result_of<
                    F(Ts...)
                >::type result_type;

            lcos::local::packaged_task<result_type(Ts...)>
                task(std::forward<F>(f));

            hpx::future<result_type> result = task.get_future();

            base_.add_after(rel_time,
                hpx::util::deferred_call(
                    std::move(task), std::forward<Ts>(ts)...),
                "async_execute_at");

            return result;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename ... Ts>
        typename hpx::util::detail::deferred_result_of<F(Ts...)>::type
        sync_execute_at(std::chrono::steady_clock::time_point const& abs_time,
            F && f, Ts &&... ts)
        {
            return async_execute_at(abs_time, std::forward<F>(f),
                std::forward<Ts>(ts)...).get();
        }

        template <typename F, typename ... Ts>
        typename hpx::util::detail::deferred_result_of<F(Ts...)>::type
        sync_execute_after(std::chrono::steady_clock::duration const& rel_time,
            F && f, Ts &&... ts)
        {
            return async_execute_after(rel_time,std::forward<F>(f),
                std::forward<Ts>(ts)...).get();
        }
    };
}}}

namespace hpx { namespace traits
{
    template <typename BaseExecutor>
    struct is_one_way_executor<
            parallel::execution::thread_timed_executor<BaseExecutor> >
      : std::true_type
    {};

    template <typename BaseExecutor>
    struct is_two_way_executor<
            parallel::execution::thread_timed_executor<BaseExecutor> >
      : std::true_type
    {};

    template <typename BaseExecutor>
    struct is_non_blocking_one_way_executor<
            parallel::execution::thread_timed_executor<BaseExecutor> >
      : std::true_type
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename BaseExecutor>
    struct executor_execution_category<
        parallel::execution::thread_timed_executor<BaseExecutor> >
    {
        typedef parallel::execution::parallel_execution_tag type;
    };
}}

#endif
