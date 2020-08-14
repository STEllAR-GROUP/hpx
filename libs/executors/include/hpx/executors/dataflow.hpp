//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/async_base/dataflow.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/coroutines/detail/get_stack_pointer.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/functional/traits/get_function_annotation.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/futures/detail/future_transforms.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/acquire_future.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/pack_traversal/pack_traversal_async.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/threading_base/thread_num_tss.hpp>
#include <hpx/type_support/always_void.hpp>

#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
// forward declare the type we will get function annotations from
namespace hpx { namespace lcos { namespace detail {
    template <typename Frame>
    struct dataflow_finalization;
}}}    // namespace hpx::lcos::detail

namespace hpx { namespace traits {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    ///////////////////////////////////////////////////////////////////////////
    // traits specialization to get annotation from dataflow_finalization
    template <typename Frame>
    struct get_function_annotation<lcos::detail::dataflow_finalization<Frame>>
    {
        using function_type = typename Frame::function_type;
        //
        static char const* call(
            lcos::detail::dataflow_finalization<Frame> const& f) noexcept
        {
            char const* annotation =
                hpx::traits::get_function_annotation<typename hpx::util::decay<
                    function_type>::type>::call(f.this_->func_);
            return annotation;
        }
    };
#endif
}}    // namespace hpx::traits

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail {
    template <typename Frame>
    struct dataflow_finalization
    {
        //
        explicit dataflow_finalization(Frame* df)
          : this_(df)
        {
        }
        using is_void = typename Frame::is_void;
        //
        template <typename Futures>
        void operator()(Futures&& futures) const
        {
            return this_->execute(is_void{}, std::forward<Futures>(futures));
        }

        // keep the dataflow frame alive with this pointer reference
        hpx::intrusive_ptr<Frame> this_;
    };

    template <typename F, typename Args>
    struct dataflow_not_callable
    {
        static auto error(F f, Args args)
        {
            hpx::util::invoke_fused(std::move(f), std::move(args));
        }

        using type = decltype(error(std::declval<F>(), std::declval<Args>()));
    };

    ///////////////////////////////////////////////////////////////////////
    template <bool IsAction, typename F, typename Args, typename Enable = void>
    struct dataflow_return_impl
    {
        typedef typename dataflow_not_callable<F, Args>::type type;
    };

    template <typename F, typename Args>
    struct dataflow_return_impl<
        /*IsAction=*/false, F, Args,
        typename hpx::util::always_void<typename hpx::util::detail::
                invoke_fused_result<F, Args>::type>::type>
      : util::detail::invoke_fused_result<F, Args>
    {
    };

    template <typename F, typename Args>
    struct dataflow_return
      : detail::dataflow_return_impl<traits::is_action<F>::value, F, Args>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Policy, typename Func, typename Futures>
    struct dataflow_frame    //-V690
      : hpx::lcos::detail::future_data<
            typename detail::dataflow_return<Func, Futures>::type>
    {
        typedef
            typename detail::dataflow_return<Func, Futures>::type result_type;
        typedef hpx::lcos::detail::future_data<result_type> base_type;

        typedef hpx::lcos::future<result_type> type;

        typedef std::is_void<result_type> is_void;

        typedef Func function_type;
        typedef dataflow_frame<Policy, Func, Futures> dataflow_type;

        friend struct dataflow_finalization<dataflow_type>;
        friend struct traits::get_function_annotation<
            dataflow_finalization<dataflow_type>>;

    private:
        // workaround gcc regression wrongly instantiating constructors
        dataflow_frame();
        dataflow_frame(dataflow_frame const&);

    public:
        typedef typename base_type::init_no_addref init_no_addref;

        /// A struct to construct the dataflow_frame in-place
        struct construction_data
        {
            Policy policy_;
            Func func_;
        };

        /// Construct the dataflow_frame from the given policy
        /// and callable object.
        static construction_data construct_from(Policy policy, Func func)
        {
            return construction_data{std::move(policy), std::move(func)};
        }

        explicit dataflow_frame(construction_data data)
          : base_type(init_no_addref{})
          , policy_(std::move(data.policy_))
          , func_(std::move(data.func_))
        {
        }

    private:
        ///////////////////////////////////////////////////////////////////////
        /// Passes the futures into the evaluation function and
        /// sets the result future.
        HPX_FORCEINLINE
        void execute(std::false_type, Futures&& futures)
        {
            std::exception_ptr p;

            try
            {
                Func func = std::move(func_);

                this->set_data(
                    util::invoke_fused(std::move(func), std::move(futures)));
                return;
            }
            catch (...)
            {
                p = std::current_exception();
            }

            // The exception is set outside the catch block since
            // set_exception may yield. Ending the catch block on a
            // different worker thread than where it was started may lead
            // to segfaults.
            this->set_exception(std::move(p));
        }

        /// Passes the futures into the evaluation function and
        /// sets the result future.
        HPX_FORCEINLINE
        void execute(std::true_type, Futures&& futures)
        {
            std::exception_ptr p;

            try
            {
                Func func = std::move(func_);

                util::invoke_fused(std::move(func), std::move(futures));

                this->set_data(util::unused_type());
                return;
            }
            catch (...)
            {
                p = std::current_exception();
            }

            // The exception is set outside the catch block since
            // set_exception may yield. Ending the catch block on a
            // different worker thread than where it was started may lead
            // to segfaults.
            this->set_exception(std::move(p));
        }

        ///////////////////////////////////////////////////////////////////////
        void finalize(hpx::detail::async_policy policy, Futures&& futures)
        {
            detail::dataflow_finalization<dataflow_type> this_f_(this);

            parallel::execution::parallel_policy_executor<launch::async_policy>
                exec{policy};

            exec.post(std::move(this_f_), std::move(futures));
        }

        void finalize(hpx::detail::fork_policy policy, Futures&& futures)
        {
            detail::dataflow_finalization<dataflow_type> this_f_(this);

            parallel::execution::parallel_policy_executor<launch::fork_policy>
                exec{policy};

            exec.post(std::move(this_f_), std::move(futures));
        }

        HPX_FORCEINLINE
        void finalize(hpx::detail::sync_policy, Futures&& futures)
        {
            // We need to run the completion on a new thread if we are on a
            // non HPX thread.
            bool recurse_asynchronously =
                hpx::threads::get_self_ptr() == nullptr;
#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
            recurse_asynchronously = !this_thread::has_sufficient_stack_space();
#else
            struct handle_continuation_recursion_count
            {
                handle_continuation_recursion_count()
                  : count_(threads::get_continuation_recursion_count())
                {
                    ++count_;
                }
                ~handle_continuation_recursion_count()
                {
                    --count_;
                }

                std::size_t& count_;
            } cnt;
            recurse_asynchronously = recurse_asynchronously ||
                cnt.count_ > HPX_CONTINUATION_MAX_RECURSION_DEPTH;
#endif
            if (!recurse_asynchronously)
            {
                hpx::util::annotate_function annotate(func_);
                execute(is_void{}, std::move(futures));
            }
            else
            {
                finalize(hpx::launch::async, std::move(futures));
            }
        }

        void finalize(launch policy, Futures&& futures)
        {
            if (policy == launch::sync)
            {
                finalize(launch::sync, std::move(futures));
            }
            else if (policy == launch::fork)
            {
                finalize(launch::fork, std::move(futures));
            }
            else
            {
                finalize(launch::async, std::move(futures));
            }
        }

        // The overload for hpx::dataflow taking an executor simply forwards
        // to the corresponding executor customization point.
        //
        // parallel::execution::executor
        // threads::executor
        template <typename Executor>
        HPX_FORCEINLINE typename std::enable_if<
            traits::is_one_way_executor<Executor>::value ||
            traits::is_two_way_executor<Executor>::value ||
            traits::is_threads_executor<Executor>::value>::type
        finalize(Executor&& exec, Futures&& futures)
        {
            detail::dataflow_finalization<dataflow_type> this_f_(this);

            parallel::execution::post(std::forward<Executor>(exec),
                std::move(this_f_), std::move(futures));
        }

    public:
        /// Check whether the current future is ready
        template <typename T>
        auto operator()(util::async_traverse_visit_tag, T&& current)
            -> decltype(async_visit_future(std::forward<T>(current)))
        {
            return async_visit_future(std::forward<T>(current));
        }

        /// Detach the current execution context and continue when the
        /// current future was set to be ready.
        template <typename T, typename N>
        auto operator()(util::async_traverse_detach_tag, T&& current, N&& next)
            -> decltype(async_detach_future(
                std::forward<T>(current), std::forward<N>(next)))
        {
            return async_detach_future(
                std::forward<T>(current), std::forward<N>(next));
        }

        /// Finish the dataflow when the traversal has finished
        HPX_FORCEINLINE void operator()(
            util::async_traverse_complete_tag, Futures futures)
        {
            finalize(policy_, std::move(futures));
        }

    private:
        Policy policy_;
        Func func_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Policy, typename Func, typename... Ts,
        typename Frame = dataflow_frame<typename std::decay<Policy>::type,
            typename std::decay<Func>::type,
            util::tuple<typename std::decay<Ts>::type...>>>
    typename Frame::type create_dataflow(
        Policy&& policy, Func&& func, Ts&&... ts)
    {
        // Create the data which is used to construct the dataflow_frame
        auto data = Frame::construct_from(
            std::forward<Policy>(policy), std::forward<Func>(func));

        // Construct the dataflow_frame and traverse
        // the arguments asynchronously
        hpx::intrusive_ptr<Frame> p = util::traverse_pack_async(
            util::async_traverse_in_place_tag<Frame>{}, std::move(data),
            std::forward<Ts>(ts)...);

        using traits::future_access;
        return future_access<typename Frame::type>::create(std::move(p));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Allocator, typename Policy, typename Func,
        typename... Ts,
        typename Frame = dataflow_frame<typename std::decay<Policy>::type,
            typename std::decay<Func>::type,
            util::tuple<typename std::decay<Ts>::type...>>>
    typename Frame::type create_dataflow_alloc(
        Allocator const& alloc, Policy&& policy, Func&& func, Ts&&... ts)
    {
        // Create the data which is used to construct the dataflow_frame
        auto data = Frame::construct_from(
            std::forward<Policy>(policy), std::forward<Func>(func));

        // Construct the dataflow_frame and traverse
        // the arguments asynchronously
        hpx::intrusive_ptr<Frame> p = util::traverse_pack_async_allocator(alloc,
            util::async_traverse_in_place_tag<Frame>{}, std::move(data),
            std::forward<Ts>(ts)...);

        using traits::future_access;
        return future_access<typename Frame::type>::create(std::move(p));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <bool IsAction, typename Policy, typename Enable = void>
    struct dataflow_dispatch_impl;

    // launch
    template <typename Policy>
    struct dataflow_dispatch_impl<false, Policy,
        typename std::enable_if<traits::is_launch_policy<Policy>::value>::type>
    {
        template <typename Allocator, typename Policy_, typename F,
            typename... Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            !traits::is_action<typename std::decay<F>::type>::value,
            lcos::future<
                typename detail::dataflow_return<typename std::decay<F>::type,
                    util::tuple<typename traits::acquire_future<Ts>::type...>>::
                    type>>::type
        call(Allocator const& alloc, Policy_&& policy, F&& f, Ts&&... ts)
        {
            return detail::create_dataflow_alloc(alloc,
                std::forward<Policy_>(policy), std::forward<F>(f),
                traits::acquire_future_disp()(std::forward<Ts>(ts))...);
        }
    };

    template <typename Policy>
    struct dataflow_dispatch<Policy,
        typename std::enable_if<traits::is_launch_policy<Policy>::value>::type>
    {
        template <typename Allocator, typename F, typename... Ts>
        HPX_FORCEINLINE static auto call(
            Allocator const& alloc, F&& f, Ts&&... ts)
            -> decltype(dataflow_dispatch_impl<false, Policy>::call(
                alloc, std::forward<F>(f), std::forward<Ts>(ts)...))
        {
            return dataflow_dispatch_impl<false, Policy>::call(
                alloc, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Allocator, typename P, typename F, typename... Ts>
        HPX_FORCEINLINE static auto call(Allocator const& alloc, P&& p, F&& f,
            typename std::enable_if<
                traits::is_action<typename std::decay<F>::type>::value,
                hpx::naming::id_type>::type const& id,
            Ts&&... ts)
            -> decltype(dataflow_dispatch_impl<
                traits::is_action<typename std::decay<F>::type>::value,
                Policy>::call(alloc, std::forward<P>(p), std::forward<F>(f), id,
                std::forward<Ts>(ts)...))
        {
            return dataflow_dispatch_impl<
                traits::is_action<typename std::decay<F>::type>::value,
                Policy>::call(alloc, std::forward<P>(p), std::forward<F>(f), id,
                std::forward<Ts>(ts)...);
        }
    };

    // executors
    template <typename Executor>
    struct dataflow_dispatch<Executor,
        typename std::enable_if<traits::is_one_way_executor<Executor>::value ||
            traits::is_two_way_executor<Executor>::value ||
            traits::is_threads_executor<Executor>::value>::type>
    {
        template <typename Allocator, typename Executor_, typename F,
            typename... Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            !traits::is_action<typename std::decay<F>::type>::value,
            lcos::future<
                typename detail::dataflow_return<typename std::decay<F>::type,
                    util::tuple<typename traits::acquire_future<Ts>::type...>>::
                    type>>::type
        call(Allocator const& alloc, Executor_&& exec, F&& f, Ts&&... ts)
        {
            return detail::create_dataflow_alloc(alloc,
                std::forward<Executor_>(exec), std::forward<F>(f),
                traits::acquire_future_disp()(std::forward<Ts>(ts))...);
        }
    };

    // any action, plain function, or function object
    template <typename FD>
    struct dataflow_dispatch_impl<false, FD,
        typename std::enable_if<!traits::is_launch_policy<FD>::value &&
            !(traits::is_one_way_executor<FD>::value ||
                traits::is_two_way_executor<FD>::value ||
                traits::is_threads_executor<FD>::value)>::type>
    {
        template <typename Allocator, typename F, typename... Ts,
            typename Enable = typename std::enable_if<
                !traits::is_action<typename std::decay<F>::type>::value>::type>
        HPX_FORCEINLINE static auto call(Allocator const& alloc, F&& f,
            Ts&&... ts) -> decltype(dataflow_dispatch<launch>::call(alloc,
            launch::async, std::forward<F>(f), std::forward<Ts>(ts)...))
        {
            return dataflow_dispatch<launch>::call(alloc, launch::async,
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }
    };

    template <typename FD>
    struct dataflow_dispatch<FD,
        typename std::enable_if<!traits::is_launch_policy<FD>::value &&
            !(traits::is_one_way_executor<FD>::value ||
                traits::is_two_way_executor<FD>::value ||
                traits::is_threads_executor<FD>::value)>::type>
    {
        template <typename Allocator, typename F, typename... Ts>
        HPX_FORCEINLINE static auto call(
            Allocator const& alloc, F&& f, Ts&&... ts)
            -> decltype(dataflow_dispatch_impl<
                traits::is_action<typename std::decay<F>::type>::value,
                launch>::call(alloc, launch::async, std::forward<F>(f),
                std::forward<Ts>(ts)...))
        {
            return dataflow_dispatch_impl<
                traits::is_action<typename std::decay<F>::type>::value,
                launch>::call(alloc, launch::async, std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }
    };
}}}    // namespace hpx::lcos::detail
